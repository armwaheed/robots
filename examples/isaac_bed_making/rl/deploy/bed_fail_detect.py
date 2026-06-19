"""Pull-failure detector: did the draw move a FREE sheet, or stall against an ANCHORED load
(the classic failure = the claw caught the fixed mattress cover, not just the sheet)?

Two robust signals, both from the existing public RobotState — no extra telemetry, no harness change:

  * follow_err — max |commanded - actual| over the arm joints (rad). Under the PD arm-overlay a joint
    held back by an immovable load lags its command (torque ≈ Kp·follow_err), so a stalled arm shows a
    large, SUSTAINED following error while a free sweep tracks closely. This is our torque proxy.
  * yaw_drift — base IMU yaw change from draw-start (deg). Pulling an anchored load feeds a reaction
    moment into the body and the vendor balancer counter-rotates the torso (observed ~15° on a gripped
    mattress cover, hardware 2026-06-18). A free pull barely twists the torso.

Optional THIRD signal, logged only (not in the verdict): tau_est on the arm joints. Direct motor
torque — the strongest "pulling hard" signal IF it is populated, but it reads 0/garbage on some G1 EDU
firmware (cf. power_v=0.0), so we measure it for calibration and do not yet depend on it.

Verdict = FAILED if follow_err OR yaw_drift stays above its threshold for >= sustain_s (sustained, not
a transient spike at the start of the sweep). Thresholds are first-pass — calibrate against one free
pull and one anchored pull, then tighten and enable --abort-on-stall.
"""
from __future__ import annotations

import math


def yaw_deg_from_quat(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def _ang_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


class DrawResistanceMonitor:
    """Watch the draw and decide free-vs-anchored. Call start() at draw start, update() every tick."""

    def __init__(self, arm_joints, dt, *, follow_thresh_rad=0.28, tau_thresh=11.0, yaw_thresh_deg=6.0,
                 sustain_s=0.5, warmup_s=0.6):
        self._arm = list(arm_joints)
        self._dt = dt
        self.follow_thresh = follow_thresh_rad
        self.tau_thresh = tau_thresh
        self.yaw_thresh = yaw_thresh_deg
        self._sustain_ticks = max(1, int(sustain_s / dt))
        self._warmup_ticks = max(0, int(warmup_s / dt))   # ignore the draw's onset acceleration transient
        self._yaw0 = None
        self._tick = 0
        self._over = 0
        self.peak_follow = 0.0          # steady-state peaks (post-warmup) — what the verdict uses
        self.peak_yaw = 0.0
        self.peak_tau = 0.0
        self.min_dq = float("inf")
        self.tripped = False

    def start(self, state) -> None:
        self._yaw0 = yaw_deg_from_quat(state.quat_wxyz)
        self._tick = 0
        self._over = 0
        self.peak_follow = self.peak_yaw = self.peak_tau = 0.0
        self.min_dq = float("inf")
        self.tripped = False

    def update(self, state, cmdq, tau=None) -> dict:
        self._tick += 1
        armed = self._tick > self._warmup_ticks            # past the startup transient?
        follow = max(abs(cmdq[j] - state.q[j]) for j in self._arm)
        yaw = _ang_diff_deg(yaw_deg_from_quat(state.quat_wxyz), self._yaw0) if self._yaw0 is not None else 0.0
        dq = sum(abs(state.dq[j]) for j in self._arm) / len(self._arm)   # mean |arm velocity| (logged only)
        tau_max = max(abs(v) for v in tau.values()) if tau else 0.0
        if armed:                                          # peaks for the VERDICT exclude the transient
            self.peak_follow = max(self.peak_follow, follow)
            self.peak_yaw = max(self.peak_yaw, yaw)
            self.peak_tau = max(self.peak_tau, tau_max)
            self.min_dq = min(self.min_dq, dq)
        # ANCHORED = the arm pushes hard against a load that won't move. The two signals that PERSIST
        # and separate ~2× free-vs-anchored (calibrated 2026-06-18): following error (torque proxy) and
        # tau_est (direct torque). dq does NOT separate — the deterministic ramp saturates so the arm
        # settles to dq≈0 in BOTH cases; yaw is dead (base IMU misses the upper-torso twist). Either
        # signal over threshold (sustained) flags it — sensitive on purpose (asking for help on a good
        # pull is cheaper than claiming success on a failed one).
        over = armed and (follow > self.follow_thresh or tau_max > self.tau_thresh)
        self._over = self._over + 1 if over else 0
        if self._over >= self._sustain_ticks:
            self.tripped = True
        return {"follow": follow, "yaw": yaw, "dq": dq, "tau": tau_max,
                "armed": armed, "over": over, "tripped": self.tripped}

    def summary(self) -> str:
        min_dq = self.min_dq if self.min_dq != float("inf") else 0.0
        verdict = "ANCHORED LOAD — pull FAILED" if self.tripped else "free pull — OK"
        return (f"peak_follow={self.peak_follow:.3f}rad (thr {self.follow_thresh}) "
                f"peak_tau={self.peak_tau:.2f} (thr {self.tau_thresh}) "
                f"[min_dq={min_dq:.3f} peak_yaw={self.peak_yaw:.1f}deg — logged only]  ->  {verdict}")


def read_arm_tau(io, arm_joints):
    """Best-effort estimated joint torque from the latest lowstate (None if unavailable). Private
    RobotIO internals; wrapped so a firmware that lacks tau_est just degrades to the two main signals."""
    try:
        ms = io._ls.motor_state
        sdk = io._sdk
        return {n: float(ms[sdk[n]].tau_est) for n in arm_joints}
    except Exception:
        return None
