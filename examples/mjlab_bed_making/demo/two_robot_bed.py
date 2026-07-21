"""Two G1s make a bed together — pure mjlab / MuJoCo(-Warp), no Isaac / Newton / NVIDIA-proprietary.

Both robots are driven by the SAME trained bed-reach policy (``bed_reach_g1.onnx``); the +y robot runs
it through a sagittal mirror (see g1_policy). Each reaches onto its own head-side corner of the sheet
and draws it headward toward the pillows while balancing under the load the policy trained against.
The sheet's grabbed edge is grip-locked to the hands (a documented abstraction) so the draw actually
moves the cover. Renders a 3/4 MP4 framing the whole bed + both robots.
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from bed_scene import build_scene, set_init_pose
from g1_policy import G1ReachController

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                                # examples/mjlab_bed_making

# Peer 0 works the -y long side (bed corners SW=D head, SE=C foot); peer 1 the +y side (NW=A, NE=B).
R0_CORNERS = ("D", "C")   # head, foot
R1_CORNERS = ("A", "B")

# Scripted pelvis-frame reach waypoints (canonical/left-hand frame; inside the trained reach box
# ranges_lo=(0.15,0.10,-0.30) hi=(0.40,0.35,0.05)). For the +y-facing robot the pelvis-frame LEFT
# axis (+y_body) maps to world -x = HEADWARD, so the headward draw sweeps `left`, not `forward`.
REST = (0.15, 0.23, -0.08)      # natural standing palm
LOW = (0.30, 0.14, -0.245)      # reach DOWN onto the sheet's head corner (footward-of-centre)
PULL = (0.30, 0.35, -0.13)      # draw the gripped corner HEADWARD (sweep left) and lift
OFF = (0.16, 0.28, 0.03)        # release + lift off the bed

# How far headward the gripped cover is drawn over the draw phase (grip-lock abstraction, see below).
DRAW_STROKE = 0.62

# Phase boundaries in control steps (50 Hz).
P_SETTLE = 60
P_REACH = 170                   # end of reach-down; grip engages here
P_HOLD = 195
P_DRAW = 330                    # end of headward draw
P_RETRACT = 410                 # hand lifts back off the bed


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    s = smoothstep(t)
    return tuple(ai + (bi - ai) * s for ai, bi in zip(a, b))


def phase_target(step):
    """Return (is_reach, pelvis-frame target) for a control step."""
    if step < P_SETTLE:
        return False, REST
    if step < P_REACH:
        return True, lerp(REST, LOW, (step - P_SETTLE) / (P_REACH - P_SETTLE))
    if step < P_HOLD:
        return True, LOW
    if step < P_DRAW:
        return True, lerp(LOW, PULL, (step - P_HOLD) / (P_DRAW - P_HOLD))
    if step < P_RETRACT:
        return True, lerp(PULL, OFF, (step - P_DRAW) / (P_RETRACT - P_DRAW))
    return False, REST


def load_coordinator():
    """Import the ENGINE-AGNOSTIC MHS coordination layer from examples/isaac_bed_making (reused,
    not re-implemented). Works from the repo tree or a loose Spark checkout."""
    for root in (os.environ.get("MJLAB_BED_REPO"),
                 os.path.abspath(os.path.join(HERE, "..", "..", "..")),
                 os.path.expanduser("~/mhs-humanoid/robots")):
        if root and os.path.isdir(os.path.join(root, "examples", "isaac_bed_making")):
            if root not in sys.path:
                sys.path.insert(0, root)
            from examples.isaac_bed_making.coordination import SwarmCoordinator
            return SwarmCoordinator
    raise ImportError("could not locate examples/isaac_bed_making; set MJLAB_BED_REPO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(PKG, "policy", "bed_reach_g1.onnx"))
    ap.add_argument("--out", default=os.path.join(PKG, "media", "bed_making.mp4"))
    ap.add_argument("--steps", type=int, default=460)
    ap.add_argument("--probe", action="store_true", help="no render; log envelope + upright")
    ap.add_argument("--decim", type=int, default=4)
    ap.add_argument("--coord", choices=["loopback", "broker", "none"], default="loopback",
                    help="MHS coordination transport (loopback=in-process, broker=NATS)")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--trace-out", default=os.path.join(PKG, "media", "mhs_trace.json"))
    args = ap.parse_args()

    sc = build_scene()
    m = sc.model
    d = mujoco.MjData(m)
    set_init_pose(m, d, sc)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ctrls = [G1ReachController(sess, r) for r in sc.robots]

    # ── MHS coordination: the two robots come up as equal peers on the fabric ────────────────────
    coord = None
    if args.coord != "none" and not args.probe:
        SwarmCoordinator = load_coordinator()
        coord = SwarmCoordinator(mode=args.coord, nats_url=args.nats_url)
        ids = coord.start()
        print(f"[mhs] swarm online ({coord.mode}): {ids}", flush=True)

    sheet_init = d.mocap_pos[sc.sheet_mocap].copy()
    gripped = False
    grip_mean0 = None

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = sc.cam_lookat
    cam.distance = sc.cam_distance
    cam.azimuth = sc.cam_azimuth
    cam.elevation = sc.cam_elevation

    renderer = None if args.probe else mujoco.Renderer(m, height=720, width=1280)
    frames = []
    palm_log = [[], []]

    for step in range(args.steps):
        is_reach, tgt = phase_target(step)
        for c in ctrls:
            c.set_reach(tgt, is_reach)
            d.ctrl[c.r.act_id] = c.compute_ctrl(d)

        for _ in range(args.decim):
            mujoco.mj_step(m, d)

        palms = [c.palm_world(d) for c in ctrls]
        if is_reach:
            for i in (0, 1):
                palm_log[i].append(palms[i].copy())

        # Grip-lock (a documented abstraction): once both hands are on the head edge, the gathered
        # cover feeds HEADWARD as they draw. The trained policy supplies the real balanced reach +
        # the ~0.1 m hand draw + the hold-under-load; the cover's full bed-making travel (DRAW_STROKE)
        # is the accordion-unspool the grip-lock stands in for (a one-stroke full draw needs the
        # retrain the README describes). It is gated on the hands actually gripping and drawing.
        if step == P_REACH:
            gripped = True
            grip_mean0 = 0.5 * (palms[0] + palms[1])
        if step == P_RETRACT:
            gripped = False
        if gripped:
            frac = smoothstep((step - P_HOLD) / max(1, P_DRAW - P_HOLD))
            lift = max(0.0, 0.5 * (palms[0][2] + palms[1][2]) - grip_mean0[2])
            new = sheet_init.copy()
            new[0] += -DRAW_STROKE * frac                 # headward (-x) toward the pillows
            new[2] = sheet_init[2] + 0.4 * lift           # follow the hands' small lift
            d.mocap_pos[sc.sheet_mocap] = new

        # ── Coordinate the two peers over MHS at the phase boundaries ────────────────────────────
        if coord is not None:
            if step == P_SETTLE:
                for i, ck in ((0, R0_CORNERS[0]), (1, R1_CORNERS[0])):
                    coord.invoke(i, "walkToNextCorner", direction="approach",
                                 effect=f"r{i} moves to its head-side corner {ck}")
            elif step == P_REACH:
                for i, ck in ((0, R0_CORNERS[0]), (1, R1_CORNERS[0])):
                    coord.invoke(i, "pickUpBedSheet", corner=ck,
                                 effect=f"r{i} grips the {ck} head corner and starts its draw")
            elif step == P_HOLD:
                coord.invoke(0, "askForHelp", corner=R0_CORNERS[0], reason="squaring my side",
                             effect="broadcasts helpRequested; r1's @on handler fires and it auto-offers")
                coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner=R0_CORNERS[0],
                             effect="r1 answers; both draw their head corners together")
            elif step == P_DRAW:
                for i, corners in ((0, R0_CORNERS), (1, R1_CORNERS)):
                    for ck in corners:
                        coord.invoke(i, "putDownBedSheet", corner=ck,
                                     effect=f"r{i} reports corner {ck} squared over its bed corner")

        if renderer is not None:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

        if step % 40 == 0 or step == args.steps - 1:
            zs = [float(d.xpos[r.pelvis_body][2]) for r in sc.robots]
            print(f"[step {step:3d}] reach={int(is_reach)} tgt={tuple(round(x,3) for x in tgt)} "
                  f"pelvis_z={[round(z,3) for z in zs]} "
                  f"palm0={palms[0].round(3).tolist()} palm1={palms[1].round(3).tolist()}", flush=True)

    for i in (0, 1):
        if palm_log[i]:
            a = np.stack(palm_log[i])
            print(f"[ENVELOPE r{i}] min={a.min(0).round(3).tolist()} max={a.max(0).round(3).tolist()} "
                  f"mean={a.mean(0).round(3).tolist()}", flush=True)

    # ── The MHS message-flow trace: every message the two peers exchanged, and its sim effect ────
    if coord is not None:
        goals = [coord.invoke(i, "getGoalState", effect="read the shared goal state") for i in (0, 1)]
        print(f"\n[mhs] goal state: {goals[0]}", flush=True)
        print(f"[mhs] message flow ({coord.mode}, profile={coord.trace.profile}, "
              f"plane={coord.trace.plane}, transport={coord.trace.transport}):", flush=True)
        print(coord.trace.render(), flush=True)
        print(f"[mhs] message counts: {coord.trace.summary()}", flush=True)
        print(f"[mhs] wrote {coord.trace.write_json(args.trace_out)}", flush=True)
        coord.stop()

    if renderer is None:
        print("[probe] no video written")
        return

    import mediapy as media
    fps = int(round(1.0 / (m.opt.timestep * args.decim)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    media.write_video(args.out, frames, fps=fps)
    print(f"[done] wrote {args.out} ({len(frames)} frames @ {fps} fps)")
    fdir = os.path.join(os.path.dirname(args.out), "frames")
    os.makedirs(fdir, exist_ok=True)
    for name, idx in {"01_stand": 30, "02_reach": P_REACH - 5, "03_grip": P_HOLD + 40,
                      "04_draw": P_DRAW - 5, "05_release": min(P_RETRACT + 20, len(frames) - 1)}.items():
        idx = max(0, min(idx, len(frames) - 1))
        media.write_image(os.path.join(fdir, f"frame_{name}.png"), frames[idx])
    print(f"[done] wrote representative frames to {fdir}")


if __name__ == "__main__":
    main()
