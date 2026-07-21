"""Two G1s make a bed together — pure mjlab / MuJoCo(-Warp), no Isaac / Newton / NVIDIA-proprietary.

Both robots are driven by the SAME trained bed-reach policy (``bed_reach_g1.onnx``); the +y robot runs
it through a sagittal mirror (see g1_policy). The sheet is a real deformable flex cloth (cloth_sheet),
authored rumpled toward the foot of the bed with ~0.5 m of slack in it.

The draw is hand-over-hand, exactly because of the policy's measured limit: one trained stroke moves
the palm ~0.10 m headward, so each robot grips its side of the sheet, draws 0.10 m, releases, reaches
back footward and re-grips the cloth that is now under its palm — five times. Every centimetre the
sheet moves is produced by the policy's own hand motion pulling on the cloth; nothing translates the
sheet kinematically. Renders a 3/4 MP4 framing the whole bed + both robots.
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort

from bed_scene import (DECIMATION, GRIP_PATCH, HAND_CAPSULE_RADIUS, SHEET_RADIUS,
                       build_scene, set_init_pose)
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
LOW = (0.30, 0.14, -0.245)      # reach DOWN onto the sheet, footward of the grab line
PULL = (0.30, 0.35, -0.13)      # draw the gripped cloth HEADWARD (sweep left) and lift
OFF = (0.16, 0.28, 0.03)        # release + lift off the bed

# Hand-over-hand draw: one stroke per grip, each worth the policy's measured ~0.10 m of palm travel.
STROKES = 6
SETTLE = 90                     # let the cloth drape before anyone touches it
APPROACH = 80                   # REST/OFF -> LOW (first stroke gets FIRST_APPROACH)
FIRST_APPROACH = 110
HOLD = 15                       # settle onto the cloth, then the grip closes
DRAW = 130                      # LOW -> PULL: the trained headward draw
LIFT = 45                       # PULL -> OFF: release and clear the bed
GRIP_REACH = 0.28               # max palm->vertex distance that counts as a grab (m)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    s = smoothstep(t)
    return tuple(ai + (bi - ai) * s for ai, bi in zip(a, b))


def stroke_len(k):
    return (FIRST_APPROACH if k == 0 else APPROACH) + HOLD + DRAW + LIFT


def schedule(strokes=STROKES):
    """Absolute control-step index of each stroke's (approach, grip, draw-end, release)."""
    marks, t = [], SETTLE
    for k in range(strokes):
        appr = (FIRST_APPROACH if k == 0 else APPROACH)
        marks.append({"start": t, "grip": t + appr + HOLD,
                      "draw_end": t + appr + HOLD + DRAW, "end": t + stroke_len(k)})
        t += stroke_len(k)
    return marks, t + 60          # a beat at the end to show the made bed


def phase_target(step, marks):
    """Return (is_reach, pelvis-frame target) for a control step."""
    if step < SETTLE:
        return False, REST
    for k, mk in enumerate(marks):
        if step >= mk["end"]:
            continue
        appr = (FIRST_APPROACH if k == 0 else APPROACH)
        home = REST if k == 0 else OFF
        if step < mk["start"] + appr:
            return True, lerp(home, LOW, (step - mk["start"]) / appr)
        if step < mk["grip"]:
            return True, LOW
        if step < mk["draw_end"]:
            return True, lerp(LOW, PULL, (step - mk["grip"]) / DRAW)
        return True, lerp(PULL, OFF, (step - mk["draw_end"]) / LIFT)
    return False, REST


def grip(model, data, sc, r):
    """Close the grip: wire this robot's connect equalities to the handful of cloth under its palm.

    The mjlab G1 has no fingers — its hand is one rigid capsule — and a fingerless palm cannot drag
    flex cloth by friction (measured: <=3% of palm travel at mu=10, see README). So the hold is a
    constraint: the cloth under the palm is connected to the hand's underside for as long as the grip
    is closed, and released the instant it opens. Everything the sheet then does — stretching,
    gathering, wrinkling, dragging over the mattress, springing back — is the solver's, not a script's.
    """
    palm = data.site_xpos[r.hand_site]
    hold = palm + np.array([0.0, 0.0, -(HAND_CAPSULE_RADIUS + SHEET_RADIUS)])   # palm underside
    verts = sc.cloth(data)
    order = np.argsort(np.linalg.norm(verts - hold, axis=1))
    near = int(order[0])
    dist = float(np.linalg.norm(verts[near] - hold))
    if dist > GRIP_REACH:
        return None                                  # nothing under the hand: the grab missed
    rot = data.xmat[r.hand_body].reshape(3, 3)
    held = 0
    for eq, vi in zip(r.eq_ids, order):
        vi = int(vi)
        if np.linalg.norm(verts[vi] - verts[near]) > GRIP_PATCH:
            break                                    # the handful ends where the cloth is out of reach
        # Hold each vertex where it already is relative to the hand, so closing the grip does not
        # snap the patch to a point; the first one is pulled up onto the palm's underside.
        target = hold if vi == near else verts[vi]
        model.eq_obj1id[eq] = sc.vert_bodies[vi]
        model.eq_obj2id[eq] = r.hand_body
        model.eq_data[eq][:3] = 0.0                  # anchor at the vertex body's own origin
        model.eq_data[eq][3:6] = rot.T @ (target - data.xpos[r.hand_body])
        data.eq_active[eq] = 1
        held += 1
    return near, dist, held


def release(data, r):
    data.eq_active[r.eq_ids] = 0


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
    ap.add_argument("--strokes", type=int, default=STROKES)
    ap.add_argument("--probe", action="store_true", help="no render; log envelope + sheet advance")
    ap.add_argument("--decim", type=int, default=DECIMATION)
    ap.add_argument("--coord", choices=["loopback", "broker", "none"], default="loopback",
                    help="MHS coordination transport (loopback=in-process, broker=NATS)")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--trace-out", default=os.path.join(PKG, "media", "mhs_trace.json"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    sc = build_scene()
    m = sc.model
    d = mujoco.MjData(m)
    set_init_pose(m, d, sc)
    marks, total = schedule(args.strokes)
    print(f"[scene] {sc.vert_num} cloth vertices, {m.nflexelem} elements; "
          f"{args.strokes} strokes over {total} control steps", flush=True)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ctrls = [G1ReachController(sess, r) for r in sc.robots]

    # ── MHS coordination: the two robots come up as equal peers on the fabric ────────────────────
    coord = None
    if args.coord != "none" and not args.probe:
        SwarmCoordinator = load_coordinator()
        coord = SwarmCoordinator(mode=args.coord, nats_url=args.nats_url)
        ids = coord.start()
        print(f"[mhs] swarm online ({coord.mode}): {ids}", flush=True)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = sc.cam_lookat
    cam.distance = sc.cam_distance
    cam.azimuth = sc.cam_azimuth
    cam.elevation = sc.cam_elevation

    renderer = None if args.probe else mujoco.Renderer(m, height=args.height, width=args.width)
    writer = None
    if renderer is not None:
        import mediapy as media
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fps = int(round(1.0 / (m.opt.timestep * args.decim)))
        # Stream to the encoder: 1800 frames of 720p held in a list is ~5 GB of RAM for no reason.
        writer = media.VideoWriter(args.out, shape=(args.height, args.width), fps=fps).__enter__()
    shots = {"01_unmade": SETTLE - 1, "02_reach": marks[0]["grip"] - 20,
             "03_grip": marks[0]["grip"] + 10, "04_draw": marks[0]["draw_end"] - 5,
             "05_regrip": marks[min(2, len(marks) - 1)]["grip"] + 10, "06_made": total - 5}
    keep = {}
    for name, idx in shots.items():
        keep.setdefault(idx, []).append(name)
    frames = {}
    palm_log = [[], []]
    head_x0 = None
    cover0 = 0.0
    grips = 0

    for step in range(total):
        is_reach, tgt = phase_target(step, marks)
        for c in ctrls:
            c.set_reach(tgt, is_reach)
            d.ctrl[c.r.act_id] = c.compute_ctrl(d)

        for _ in range(args.decim):
            mujoco.mj_step(m, d)

        palms = [c.palm_world(d) for c in ctrls]
        if is_reach:
            for i in (0, 1):
                palm_log[i].append(palms[i].copy())

        verts = sc.cloth(d)
        if step == SETTLE - 1:
            head_x0 = float(verts[:, 0].min())
            cover0 = sc.coverage(d)
            print(f"[settle] cloth draped: head edge x={head_x0:.3f} "
                  f"z {verts[:,2].min():.3f}..{verts[:,2].max():.3f}; "
                  f"mattress covered {100*cover0:.0f}%", flush=True)

        for k, mk in enumerate(marks):
            if step == mk["grip"]:
                for i, r in enumerate(sc.robots):
                    got = grip(m, d, sc, r)
                    grips += got is not None
                    print(f"[stroke {k}] r{i} grip "
                          + (f"{got[2]} vertices, nearest {got[0]} at {got[1]*100:.1f} cm"
                             if got else "MISSED (no cloth under the palm)"), flush=True)
            elif step == mk["draw_end"]:
                for r in sc.robots:
                    release(d, r)
                print(f"[stroke {k}] release: cloth head edge x={verts[:,0].min():.3f} "
                      f"(advanced {100*(head_x0 - verts[:,0].min()):+.1f} cm), "
                      f"mean x={verts[:,0].mean():.3f}", flush=True)

        # ── Coordinate the two peers over MHS across the draw ────────────────────────────────────
        if coord is not None:
            if step == SETTLE:
                for i, ck in ((0, R0_CORNERS[0]), (1, R1_CORNERS[0])):
                    coord.invoke(i, "walkToNextCorner", direction="approach",
                                 effect=f"r{i} moves to its head-side corner {ck}")
            elif step == marks[0]["grip"]:
                for i, ck in ((0, R0_CORNERS[0]), (1, R1_CORNERS[0])):
                    coord.invoke(i, "pickUpBedSheet", corner=ck,
                                 effect=f"r{i} grips the {ck} head corner of the cloth")
            elif len(marks) > 1 and step == marks[1]["grip"]:
                coord.invoke(0, "askForHelp", corner=R0_CORNERS[0], reason="squaring my side",
                             effect="broadcasts helpRequested; r1's @on handler fires and it auto-offers")
                coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner=R0_CORNERS[0],
                             effect="r1 answers; both draw their head corners together")
            elif step == marks[-1]["draw_end"]:
                for i, corners in ((0, R0_CORNERS), (1, R1_CORNERS)):
                    for ck in corners:
                        coord.invoke(i, "putDownBedSheet", corner=ck,
                                     effect=f"r{i} reports corner {ck} squared over its bed corner")

        if renderer is not None:
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            writer.add_image(img)
            for name in keep.get(step, ()):
                frames[name] = img.copy()

        if step % 60 == 0 or step == total - 1:
            zs = [float(d.xpos[r.pelvis_body][2]) for r in sc.robots]
            print(f"[step {step:4d}] reach={int(is_reach)} pelvis_z={[round(z,3) for z in zs]} "
                  f"palm0={palms[0].round(3).tolist()} "
                  f"cloth head_x={verts[:,0].min():.3f} mean_x={verts[:,0].mean():.3f}", flush=True)

    verts = sc.cloth(d)
    print(f"\n[result] {grips}/{2*args.strokes} grips landed; cloth head edge "
          f"{head_x0:.3f} -> {verts[:,0].min():.3f} m "
          f"({100*(head_x0 - verts[:,0].min()):+.1f} cm headward, all of it hand-driven)", flush=True)
    warn = {mujoco.mjtWarning(w).name: int(d.warning[w].number)
            for w in range(len(d.warning)) if d.warning[w].number}
    print(f"[result] mattress covered {100*cover0:.0f}% -> {100*sc.coverage(d):.0f}%; "
          f"nan={bool(np.isnan(verts).any())}; solver warnings: {warn or 'none'}", flush=True)
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
    writer.__exit__(None, None, None)
    print(f"[done] wrote {args.out} ({total} frames @ "
          f"{int(round(1.0 / (m.opt.timestep * args.decim)))} fps)")
    fdir = os.path.join(os.path.dirname(args.out), "frames")
    os.makedirs(fdir, exist_ok=True)
    for name, img in frames.items():
        media.write_image(os.path.join(fdir, f"frame_{name}.png"), img)
    print(f"[done] wrote {len(frames)} representative frames to {fdir}")


if __name__ == "__main__":
    main()
