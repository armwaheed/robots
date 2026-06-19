"""Grasp confirmation: did the claw close on FABRIC, or on NOTHING (an empty grab)?

This is the pull-failure detector's blind spot and the live-demo killer. `bed_fail_detect.py`
catches an ANCHORED stall ("pulled something that won't move" — grabbed the fixed cover). It does
NOT catch the opposite failure: the claw closes on AIR, the arm "draws" nothing, following-error and
tau read perfectly FREE, and the orchestrator reports SUCCESS while the sheet never moved. A confident
false-success is worse for the Device Connect story than an honest stall — so the grasp must confirm
fabric-in-hand BEFORE the draw, and emit an explicit empty-grab signal the orchestrator escalates on
(exactly like /tmp/pull_failed).

Signals — both already in the brainco_bridge `get` telemetry, no harness change:

  * touch_force rise — per-finger contact force above the open-claw baseline (u16). Fabric in the claw
    loads the fingertip; air does not. CAVEAT: thin fabric reads LOW — observed grabs are ~10-40 u16
    (cf. bed_grip_v1's force_threshold=40, which barely fires on a real grab). So touch alone is a
    WEAK separator and we deliberately confirm at a lower rise than the grip's contact gate.
  * proximity deviation — per-finger proximity sensor change from the open-claw baseline (u16). Fabric
    sitting in the claw sits in front of the fingertip sensors; an empty claw sees only background. This
    is the signal bed_grip never used. Sign is firmware-dependent (some sensors rise on approach, some
    fall), so we key on ABSOLUTE deviation from the captured baseline — direction-agnostic until
    calibration pins it down.

Verdict = FABRIC PRESENT if (max touch rise >= force_rise) OR (max proximity deviation >= prox_dev)
on at least `fingers_needed` fingers, sustained past a short settle. Otherwise EMPTY GRAB.

CALIBRATION STATUS (this G1): the TOUCH threshold (force_rise) is hardware-calibrated 2026-06-19 to 6
(empty floor ≤1 over n=7 closes; single-layer sheet 9-39 over n=6, min 9; 2-layer 81) and validated live
under --require-fabric (real grabs proceed, true empties write /tmp/grab_empty). Proximity is DEAD on
this hardware (the bridge never publishes left_proximity), so the verdict runs touch-only and prox_dev is
unused here — it stays a FIRST-PASS guess for any G1 whose bridge does publish proximity. The original
procedure (capture ~5 empty + ~5 fabric closes, set the threshold between the clusters, THEN enable
--require-fabric — the analog of the pull detector's --abort-on-stall) is automated in
bed_grasp_calibrate.py. Re-run it if the fabric or hand changes.

Bias on purpose: when the two signals disagree or proximity reads garbage, prefer to DECLARE EMPTY
(escalate to the human) over claiming a grab — a false "I have the sheet" is the failure we are here to
kill. (Proximity degrades to touch-only if it reads all-zero/garbage, mirroring read_arm_tau.)
"""
from __future__ import annotations


def _as_list(v, n):
    """Coerce a bridge sensor field to a length-n list of floats (missing/short → zeros)."""
    out = [0.0] * n
    if not v:
        return out
    for i in range(min(n, len(v))):
        try:
            out[i] = float(v[i])
        except (TypeError, ValueError):
            out[i] = 0.0
    return out


class GraspConfirmMonitor:
    """Capture an open-claw baseline, watch the close, decide fabric-vs-empty.

    Usage: baseline(reading) once with the claw open (before the digits close), update(reading) each
    sensor poll during the close, then read verdict()/summary() after the close completes.
    """

    def __init__(self, *, n_fingers=5, force_rise=6.0, prox_dev=150.0, fingers_needed=1):
        self.n = n_fingers
        # force_rise HARDWARE-CALIBRATED 2026-06-19 (touch-only; proximity dead on this G1): empty floor
        # ≤1 (n=7), single-layer sheet 9-39 (n=6, min 9), 2-layer 81. 6 sits 6x over the empty floor
        # (false-fabric ~impossible) and below the weakest real grab. See bed_grip_v1.py --confirm-force-rise.
        self.force_rise = force_rise            # u16 touch rise that counts as fabric contact (LOW: soft fabric)
        self.prox_dev = prox_dev                # u16 |proximity - baseline| that counts as something-in-claw
        self.fingers_needed = fingers_needed
        self.base_force = [0.0] * n_fingers
        self.base_prox = [0.0] * n_fingers
        self.peak_force_rise = [0.0] * n_fingers
        self.peak_prox_dev = [0.0] * n_fingers
        self._prox_seen_nonzero = False         # did proximity ever report a usable (non-zero) value?
        self._based = False

    @staticmethod
    def _force(reading):
        return reading.get("left_touch_force") if isinstance(reading, dict) else None

    @staticmethod
    def _prox(reading):
        return reading.get("left_proximity") if isinstance(reading, dict) else None

    def baseline(self, reading) -> None:
        self.base_force = _as_list(self._force(reading), self.n)
        self.base_prox = _as_list(self._prox(reading), self.n)
        self._based = True

    def update(self, reading) -> dict:
        if not self._based:                     # be forgiving: first update doubles as the baseline
            self.baseline(reading)
        force = _as_list(self._force(reading), self.n)
        prox = _as_list(self._prox(reading), self.n)
        if any(p != 0.0 for p in prox):
            self._prox_seen_nonzero = True
        for i in range(self.n):
            self.peak_force_rise[i] = max(self.peak_force_rise[i], force[i] - self.base_force[i])
            self.peak_prox_dev[i] = max(self.peak_prox_dev[i], abs(prox[i] - self.base_prox[i]))
        return {"force": force, "prox": prox,
                "force_rise": [force[i] - self.base_force[i] for i in range(self.n)],
                "prox_dev": [abs(prox[i] - self.base_prox[i]) for i in range(self.n)]}

    def _touch_fingers(self):
        return [i for i in range(self.n) if self.peak_force_rise[i] >= self.force_rise]

    def _prox_fingers(self):
        if not self._prox_seen_nonzero:         # proximity dead/garbage → don't let it vote
            return []
        return [i for i in range(self.n) if self.peak_prox_dev[i] >= self.prox_dev]

    def fabric_present(self) -> bool:
        fingers = set(self._touch_fingers()) | set(self._prox_fingers())
        return len(fingers) >= self.fingers_needed

    def verdict(self) -> dict:
        return {
            "fabric_present": self.fabric_present(),
            "touch_fingers": self._touch_fingers(),
            "prox_fingers": self._prox_fingers(),
            "peak_force_rise": [round(v, 1) for v in self.peak_force_rise],
            "peak_prox_dev": [round(v, 1) for v in self.peak_prox_dev],
            "prox_usable": self._prox_seen_nonzero,
        }

    def summary(self) -> str:
        v = self.verdict()
        prox_note = "" if v["prox_usable"] else " [proximity dead — touch-only verdict]"
        result = "FABRIC in hand" if v["fabric_present"] else "EMPTY GRAB — closed on nothing"
        return (f"peak_force_rise={v['peak_force_rise']} (thr {self.force_rise}, fired {v['touch_fingers']}) "
                f"peak_prox_dev={v['peak_prox_dev']} (thr {self.prox_dev}, fired {v['prox_fingers']})"
                f"{prox_note}  ->  {result}")
