"""Pull a real bedsheet — deterministic high lift (collision-free) → RL come-in + grip + draw.

Why the hybrid: the RL reach policy CANNOT raise the hand high/wide for a clean top-down approach —
its trained command box caps the hand at z=+0.10 (~0.83 m), and it is commanded by a single hand xyz,
so it cannot hold a "shoulders rolled wide" configuration. Coming straight in low, the forearm drags
across the mattress. So we do the lift DETERMINISTICALLY in joint space: raise the LEFT arm wide and
up, keep the shoulder rolled wide while the elbow extends (so the arm reaches OUT TO THE SIDE and
stays clear of the low mattress instead of sweeping forward through it), then hand off to the trained
RL policy to come in from shoulder height, descend onto the quilt, and draw — the part the user wants
the RL policy to own. Note: in this arm-overlay path the VENDOR balancer keeps the robot upright under
the draw load (the RL policy only generates arm targets here), so balance under the pull is covered
regardless.

The lift poses below are our own, authored for the G1 EDU left arm — the only thing borrowed is the
general principle (stay shoulder-wide while extending). Right arm is held fixed (its Brainco hand is
damaged; we never drive it).

Fall-safe throughout: legs on the vendor balancer, arms via rt/arm_sdk (weight-blended in/out),
rate-limited + clamped, abort polled every tick, SafeStop damps (now a SMOOTH blended release).

  --lift-only   do ONLY the deterministic lift, then blend out — validate clearance before adding RL
  (default)     lift → RL come-in + descend → hold-for-grip → RL draw → release
"""
from __future__ import annotations

import argparse
import os
import time

from bed_deploy_common import ARM_JOINTS, ARM_LIMITS, LEFT_ARM, RIGHT_ARM, load, rc_root

# Deterministic LEFT-arm lift waypoints (joint-space, rad). Principle: shoulder stays rolled WIDE
# (high +roll abducts the left arm out to the side) while the elbow extends, so the arm rises and
# reaches out to the side ABOVE the low mattress, never sweeping forward through it. Tune via
# --lift-only + eye-verify before trusting the RL come-in.
LIFT_WAYPOINTS = [
    ("ape_hanger", {  # raise wide + up, elbow bent — hand hangs high, clear of the mattress
        "left_shoulder_pitch_joint": -0.52, "left_shoulder_roll_joint": 1.65,
        "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 1.50, "left_wrist_roll_joint": 0.0}),
    ("extend_side", {  # keep shoulder wide, extend the elbow + roll the palm down — arm out to side
        "left_shoulder_pitch_joint": -0.45, "left_shoulder_roll_joint": 1.55,
        "left_shoulder_yaw_joint": -0.10, "left_elbow_joint": 0.45, "left_wrist_roll_joint": -0.35}),
]

# RL hand_target (base frame). Trained box x(.18,.55) y(±.40) z(-.16,.10). The robot faces the bed's
# left long edge, so +y (left) points toward the HEAD of the bed.
GRIP_TARGET = [0.45, 0.12, -0.14, 1.0, 0.0, 0.0, 0.0]
# DRAW: pull the cover LATERALLY toward the headboard — keep x out over the bed and slide +y toward
# the head, with a slight lift. (Pulling x INWARD toward the torso drags the cover off the side —
# the wrong direction for bed-making; observed on the cot, 2026-06-17.)
DRAW_TARGET = [0.45, 0.35, -0.10, 1.0, 0.0, 0.0, 0.0]


def run_lift_rl_pull(dep, io, SafeStop, *, lift_only, grip_target, draw_target, lift_s, comein_s,
                     draw_s, draw_hold_s, blend_s, vmax_rad_s, hold_timeout_s,
                     arm_reached_file, gripped_file, draw_done_file, log=print):
    if not io.arm_abort():
        log("[pull2] abort not armed (release all controller buttons) — refusing")
        return False
    gains = dep.gains()
    dt = dep.dt
    max_step = vmax_rad_s * dt
    n_blend = max(1, int(blend_s / dt))
    cmdq = {n: io.read_state().q[n] for n in ARM_JOINTS}   # all 10 arm joints; right held at start
    right_hold = {n: cmdq[n] for n in RIGHT_ARM}
    traj = [dict(cmdq)]  # recorded forward joint path (incl. start), replayed in reverse to retract

    def _clamp(n, v):
        lo, hi = ARM_LIMITS.get(n, (-3.14, 3.14))
        return max(lo, min(hi, v))

    def ramp_to(joint_goals, seconds, label):
        """Deterministic joint-space ramp of cmdq → joint_goals (rate-limited), publishing overlay."""
        n = max(1, int(seconds / dt))
        for k in range(n):
            if io.abort_tripped():
                log(f"[pull2] abort during {label}")
                return False
            for j, goal in joint_goals.items():
                cmdq[j] += max(-max_step, min(max_step, _clamp(j, goal) - cmdq[j]))
            io.publish_targets(cmdq, gains, weight=1.0)
            traj.append(dict(cmdq))
            time.sleep(dt)
        return True

    def rl_step(command, seconds, label, record=True):
        """Run the RL policy toward `command` (hand_target); apply LEFT arm only, hold right."""
        n = max(1, int(seconds / dt))
        for k in range(n):
            if io.abort_tripped():
                log(f"[pull2] abort during {label}")
                return False
            state = io.read_state()
            obs, _ = dep.obs.build(state, command, dep.last_action)
            tq = dep.targets(dep.infer(obs))
            for j in LEFT_ARM:
                cmdq[j] += max(-max_step, min(max_step, _clamp(j, tq[j]) - cmdq[j]))
            for j in RIGHT_ARM:
                cmdq[j] = right_hold[j]
            io.publish_targets(cmdq, gains, weight=1.0)
            if record:
                traj.append(dict(cmdq))
            time.sleep(dt)
        return True

    def retract():
        """Retrace the EXACT approach path in reverse (recorded joint poses, newest→oldest), under
        overlay control, so the hand follows the descent path back out and cannot catch on the
        mattress on the way to the side. Ends at the start pose; the caller then blends the overlay
        out from there (collision-free). A plain weight-blend would let the vendor pull the arm to
        side along a DIRECT path — which is what snagged the mattress on the last run."""
        log("[pull2] retracting along the reverse approach path")
        for pose in reversed(traj):
            if io.abort_tripped():
                log("[pull2] abort during retract")
                return False
            for j in ARM_JOINTS:
                cmdq[j] = pose[j]
            io.publish_targets(cmdq, gains, weight=1.0)
            time.sleep(dt)
        return True

    for f in (arm_reached_file, gripped_file, draw_done_file):
        try:
            os.remove(f)
        except OSError:
            pass

    with SafeStop(io.damp_once, name="bed_pull2"):
        for k in range(n_blend):                                   # blend overlay IN (hold pose)
            if io.abort_tripped():
                return log("[pull2] abort during blend-in") or False
            io.publish_targets(cmdq, gains, weight=(k + 1) / n_blend)
            time.sleep(dt)

        for name, goals in LIFT_WAYPOINTS:                         # DETERMINISTIC LIFT (left arm)
            log(f"[pull2] lift → {name}")
            if not ramp_to(goals, lift_s, f"lift:{name}"):
                return False

        if lift_only:                                              # validation: stop after the lift
            log("[pull2] --lift-only: holding the lifted pose 2 s, then releasing (eye-verify clearance)")
            t_end = time.time() + 2.0
            while time.time() < t_end:
                if io.abort_tripped():
                    return log("[pull2] abort during lift-hold") or False
                io.publish_targets(cmdq, gains, weight=1.0)
                time.sleep(dt)
        else:
            log("[pull2] RL come-in + descend onto the quilt")
            if not rl_step(grip_target, comein_s, "rl-comein"):    # RL COME-IN + DESCEND
                return False
            open(arm_reached_file, "w").close()                    # HOLD for grip
            log(f"[pull2] on the quilt — holding. Grip should create {gripped_file}")
            gripped = False
            t_end = time.time() + hold_timeout_s
            while not os.path.exists(gripped_file):
                if io.abort_tripped():
                    return log("[pull2] abort during grip hold") or False
                if time.time() > t_end:
                    log("[pull2] grip timeout — skipping the draw")
                    break
                if not rl_step(grip_target, dt, "rl-hold", record=False):
                    return False
            else:
                gripped = True
                log("[pull2] grip confirmed — drawing")
            if gripped:                                            # RL DRAW
                if not rl_step(draw_target, draw_s, "rl-draw"):
                    return False
                open(draw_done_file, "w").close()
                log(f"[pull2] draw complete — signaled {draw_done_file}")
                if not rl_step(draw_target, draw_hold_s, "rl-draw-hold"):
                    return False
            else:
                open(draw_done_file, "w").close()

        if not retract():                                          # EXACT REVERSE retraction
            return False
        for k in range(n_blend):                                   # blend overlay OUT (from start pose)
            if io.abort_tripped():
                return log("[pull2] abort during blend-out") or False
            io.publish_targets(cmdq, gains, weight=1.0 - (k + 1) / n_blend)
            time.sleep(dt)
    log("[pull2] complete (overlay blended out, damped)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--contract", default="/home/unitree/bedreach_deploy/deploy_contract_v2.json")
    ap.add_argument("--policy", default="/home/unitree/bedreach_deploy/policy_v2.pt")
    ap.add_argument("--grip-target", type=float, nargs=7, default=GRIP_TARGET)
    ap.add_argument("--draw-target", type=float, nargs=7, default=DRAW_TARGET)
    ap.add_argument("--lift-only", action="store_true", help="do ONLY the deterministic lift, then release")
    ap.add_argument("--lift-s", type=float, default=3.0, help="seconds per lift waypoint")
    ap.add_argument("--comein-s", type=float, default=5.0, help="seconds for the RL come-in + descend")
    ap.add_argument("--draw-s", type=float, default=4.0, help="seconds for the RL draw")
    ap.add_argument("--draw-hold", type=float, default=1.0)
    ap.add_argument("--blend", type=float, default=1.2)
    ap.add_argument("--vmax", type=float, default=1.0, help="arm joint rate limit (rad/s)")
    ap.add_argument("--hold-timeout", type=float, default=90.0)
    ap.add_argument("--assume-balancer-up", action="store_true",
                    help="skip the in-context loco liveness gate (false-negatives under DDS contention); "
                         "use ONLY after a clean-context GET_FSM_ID code 0. Fall-safe.")
    ap.add_argument("--arm-reached-file", default="/tmp/arm_reached")
    ap.add_argument("--gripped-file", default="/tmp/sheet_gripped")
    ap.add_argument("--draw-done-file", default="/tmp/draw_done")
    args = ap.parse_args()

    rc = rc_root()
    ss = load("safe_stop", os.path.join(rc, "lib", "safe_stop.py"))
    pd = load("robotics_connect_policy_deploy", os.path.join(rc, "lib", "policy_deploy.py"))
    g1io = load("robotics_connect_g1_robot_io", os.path.join(rc, "unitree", "g1", "deploy", "g1_robot_io.py"))

    contract = pd.DeployContract.load(args.contract)
    if not contract.gains:
        raise SystemExit(f"{args.contract} carries no gains — overlay would be zero-torque.")
    print(f"[pull2] rc={rc}  contract={os.path.basename(args.contract)} ({contract.n}j, {contract.obs_total_dim}D)")
    print(f"[pull2] mode={'LIFT-ONLY' if args.lift_only else 'lift→RL come-in→grip→draw'}  "
          f"grip={args.grip_target}  draw={args.draw_target}")

    io = g1io.G1RobotIO(iface=args.iface, names=contract.action_joint_names)
    io.connect()
    dep = pd.PolicyDeploy(contract, args.policy, io)
    try:
        if not args.assume_balancer_up:
            if io._high_level_service_alive() is False:
                print("[pull2] REFUSING: balance controller DOWN. Stand in Regular mode or pass "
                      "--assume-balancer-up after a clean GET_FSM_ID 0.")
                return
        else:
            print("[pull2] --assume-balancer-up: skipping in-context liveness gate (clean probe confirmed).")
        run_lift_rl_pull(
            dep, io, ss.SafeStop, lift_only=args.lift_only,
            grip_target=args.grip_target, draw_target=args.draw_target,
            lift_s=args.lift_s, comein_s=args.comein_s, draw_s=args.draw_s, draw_hold_s=args.draw_hold,
            blend_s=args.blend, vmax_rad_s=args.vmax, hold_timeout_s=args.hold_timeout,
            arm_reached_file=args.arm_reached_file, gripped_file=args.gripped_file,
            draw_done_file=args.draw_done_file,
        )
    finally:
        io.shutdown()


if __name__ == "__main__":
    main()
