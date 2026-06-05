"""Two Unitree G1 humanoids make a bed in NVIDIA Isaac Sim, coordinating as equal
peers over Arm Device Connect.

End to end, each G1:

1. **Walks in** from ~0.5 m off its side of the bed under a **learned RL
   locomotion policy** (Unitree's open ``unitree_rl_gym`` G1 walk policy — issue #2
   item #4), balancing on its legs while tracking a velocity command.
2. **Settles** at the bedside (its base is then held steady), and
3. **Makes the bed** by replaying **real recorded arm motion** from Unitree's
   teleoperated bed-making dataset (waist + both arms, with Inspire 5-finger
   hands), gripping the cloth by friction.

The bed has a **headboard + pillows** (static; never touched) and a
**particle-cloth sheet** sized to overhang the sides + foot by ~9 inches; it
starts **folded over at the foot** and the robots arrange it.

Coordination is real Device Connect: each robot claims work, emits events, and
asks for / offers help. With ``--broker`` both peers register live on the
dashboard.

Run with the Isaac Lab Python on the DGX Spark::

    cd ~/workspaces/git/IsaacLab
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    ./isaaclab.sh -p ~/workspaces/git/robots/examples/isaac_bed_making/demo.py --loopback --render

Rendering note: the demo runs with ``use_fabric=True`` so the robot articulations
render their true motion (Isaac Lab only pushes link poses to the renderer when
Fabric is on). PhysX does not auto-sync particle-cloth deformation to Fabric, so we
read the live cloth positions from a PhysX tensor cloth-view and write them into the
Fabric mesh points each render (mesh updates propagate through Fabric — see cloth.py).

Self-contained: does not modify ``strands_robots`` or Arm's Device Connect; reuses
the swarm driver from ``examples/unitree_g1_bed_making_g1_driver.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--loopback", action="store_true", help="Coordinate via in-process bus (offline, default).")
    mode.add_argument("--broker", action="store_true", help="Register both peers on the real Device Connect NATS fabric.")
    p.add_argument("--no-device-connect", action="store_true", help="Skip Device Connect entirely.")
    p.add_argument("--gui", action="store_true",
                   help="Open the Isaac Sim window to watch live (instead of headless).")
    p.add_argument("--render", action="store_true", help="Capture frames and encode an mp4.")
    p.add_argument("--frames-dir", default=str(REPO_ROOT / "artifacts" / "isaac_bed_making"))
    p.add_argument("--nats-url", default=os.environ.get("DEVICE_CONNECT_NATS_URL", "nats://fabric.deviceconnect.dev:4222"))
    p.add_argument("--max-seconds", type=float, default=120.0, help="Safety cap on sim wall-time.")
    p.add_argument("--no-walk", action="store_true", help="Skip the learned approach-walk (spawn at the bedside).")
    p.add_argument("--walk-only", action="store_true", help="Stop after the approach-walk (debug the locomotion).")
    p.add_argument("--replay", action="store_true",
                   help="Use the open-loop dataset arm replay instead of closed-loop corner manipulation.")
    p.add_argument("--replay-speed", type=float, default=1.0, help="Playback speed of the recorded arm motion.")
    p.add_argument("--lag", type=float, default=0.4, help="Seconds robot 1 trails robot 0 in the replay.")
    return p.parse_known_args()[0]


ARGS = parse_args()

# Import eigenpy + pinocchio BEFORE the Isaac app launches so eigenpy registers its
# StdVec_StdString converter first. Launching the app afterwards otherwise shadows
# that registration and Pinocchio's ``model.names.tolist()`` throws inside the Pink
# IK controller (manipulation.PinkArmIK). Harmless if pinocchio isn't installed.
try:
    import eigenpy  # noqa: F401
    import pinocchio  # noqa: F401
except Exception:
    pass

# ── Launch the simulator first (Isaac Lab requires this before other imports) ──
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": not ARGS.gui, "enable_cameras": bool(ARGS.render or ARGS.gui)})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # noqa: E402

from examples.isaac_bed_making import cloth as clothmod  # noqa: E402
from examples.isaac_bed_making import locomotion as locomod  # noqa: E402
from examples.isaac_bed_making import scene as scenemod  # noqa: E402
from examples.isaac_bed_making.manipulation import PinkArmIK, apply_hand_friction  # noqa: E402
from examples.isaac_bed_making.replay import TrajectoryReplay  # noqa: E402

SIM_DT = 0.002            # the rl_gym walk policy's native sim rate (500 Hz)
DECIMATION = 10           # policy/control runs every 10 sim steps (50 Hz)


def main() -> int:
    frames_dir = Path(ARGS.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    def mark(msg):
        print(f"[setup] {msg}", flush=True)

    mark("building scene cfg (Inspire 5-finger G1s, headboard + pillows)")
    SceneCfg = scenemod.build_scene_cfg(G1_INSPIRE_FTP_CFG)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0", use_fabric=True))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=8.0))
    mark("scene built")

    pre_stage = omni.usd.get_context().get_stage()
    # (1) Turn each Inspire G1 into a FLOATING-base articulation so the policy can
    # walk it (the USD ships fixed-base; see locomotion.make_floating_base), and
    # (2) rubberize the hands so they grip the cloth by friction. Both edit the
    # stage and MUST happen before sim.reset().
    for i in (0, 1):
        rp = f"/World/envs/env_0/Robot_{i}"
        locomod.make_floating_base(pre_stage, rp)
    # Moderate (not sticky) hand friction: the hands *smooth/press* the draped
    # sheet rather than grabbing and lifting it (high friction yanks it off the bed).
    nb = [apply_hand_friction(pre_stage, f"/World/envs/env_0/Robot_{i}", "right",
                              static_friction=1.2, dynamic_friction=1.2) for i in (0, 1)]
    mark(f"floating base set; hand friction bound to {len(nb[0])}+{len(nb[1])} colliders")

    cam = None
    if ARGS.render or ARGS.gui:
        from isaaclab.sensors.camera import Camera, CameraCfg
        cam = Camera(cfg=CameraCfg(
            prim_path="/World/CameraSensor", update_period=0,
            height=scenemod.CAM_RES[1], width=scenemod.CAM_RES[0], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=400.0,
                                             horizontal_aperture=20.955, clipping_range=(0.05, 1.0e5))))

    mark("calling sim.reset()")
    sim.reset()
    stage = omni.usd.get_context().get_stage()
    robots = [scene["robot_0"], scene["robot_1"]]
    sim_dt = sim.get_physics_dt()
    if cam is not None:
        cam.set_world_poses_from_view(torch.tensor([scenemod.CAM_EYE], device=sim.device),
                                      torch.tensor([scenemod.CAM_TARGET], device=sim.device))

    # ── learned locomotion: two policies, each used for what it's good at ──
    # WALK-IN: Unitree's open rl_gym G1 walk policy strides across the floor (one
    # stateful LSTM instance per robot — sharing one would mix their hidden states).
    # STANCE: Isaac Lab's agile policy actively balances each robot in place while
    # the arms work. The walk policy is a gait-clock walker — at zero command it keeps
    # marching and drifts/topples, so it travels but does not hold a manipulation stance.
    walk = {i: locomod.RLGymWalker(robots[i], sim.device, SIM_DT * DECIMATION) for i in (0, 1)}
    agile = locomod.load_agile_policy(sim.device)
    stance = {i: locomod.LocomotionPolicy(robots[i], sim.device, agile) for i in (0, 1)}
    locos = walk  # phase 1 (the approach-walk) drives the robots with the walkers
    mark("locomotion ready (rl_gym walk-in + agile balance stance)")

    # ── folded bedsheet ──
    scene_path = clothmod.find_physics_scene_path(stage)
    clothmod.enable_gpu_dynamics(stage, scene_path)
    sheet = clothmod.build_bedsheet(
        stage, scene_path, "/World/Sheet",
        size=scenemod.SHEET_SIZE, resolution=scenemod.SHEET_RES,
        origin=scenemod.SHEET_ORIGIN, color=(0.86, 0.86, 0.92),
        thickness=scenemod.SHEET_THICKNESS, fold=True,
        fold_start=scenemod.SHEET_FOLD_START)
    mark("folded bedsheet built")

    # ── manipulation: closed-loop Pink IK (arm + waist) + optional dataset replay ──
    arms = {i: PinkArmIK(robots[i], "right", sim.device, SIM_DT * DECIMATION) for i in (0, 1)}
    reps = {i: TrajectoryReplay(robots[i], sim.device) for i in (0, 1)} if ARGS.replay else {}

    # ── Device Connect swarm ──
    coord = None
    if not ARGS.no_device_connect:
        from examples.isaac_bed_making.coordination import SwarmCoordinator
        coord = SwarmCoordinator(mode="broker" if ARGS.broker else "loopback", nats_url=ARGS.nats_url)
        print(f"[demo] Device Connect swarm online ({coord.mode}): {coord.start()}")

    state = {"frame": 0, "t": 0.0, "pin": None}
    cloth_view = None
    fabric_points = None
    zero_vel = torch.zeros((1, 6), device=sim.device)

    def step(n=1):
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step(render=False)
            # Once planted (after the walk-in), the pelvis is held kinematically at
            # its arrived pose EVERY sim step. A free-floating G1 cannot balance
            # through a bed-making reach/lean with the policies we have — it topples;
            # planting the base (the robot "stands firmly") lets the arms do the real
            # recorded motion while the legs/feet stay put. Writing every sim step
            # (not just every control step) is what actually holds it.
            if state["pin"] is not None:
                for i in (0, 1):
                    robots[i].write_root_pose_to_sim(state["pin"][i])
                    robots[i].write_root_velocity_to_sim(zero_vel)
            scene.update(sim_dt)
            state["t"] += sim_dt

    def capture():
        if cam is None:
            return
        if cloth_view is not None and fabric_points is not None:
            clothmod.sync_fabric_from_view(cloth_view, fabric_points)
        sim.render()  # updates the live GUI viewport too, when --gui
        cam.update(dt=sim_dt)
        if not ARGS.render:
            return  # GUI-only: shown live, nothing to save
        out = cam.data.output["rgb"]
        if out is None or out.shape[0] == 0:
            return
        rgb = out[0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb).save(frames_dir / f"frame_{state['frame']:04d}.png")
        state["frame"] += 1
        if state["frame"] % 20 == 0:
            print(f"[demo] captured {state['frame']} frames (t={state['t']:.1f}s)", flush=True)

    def control(cmd_fn, n_ctl, cap_every=2, pols=None):
        """Run a locomotion policy for n_ctl control steps (each = DECIMATION sim
        steps). ``cmd_fn(i)`` returns (command, arrived) for robot i. ``pols`` selects
        which policy set drives the legs (walk vs. stance); defaults to the walkers.
        The command must match the chosen policy (``command_to``/``stand`` of the same
        set). Stops early when both robots report arrived."""
        pols = pols or locos
        for c in range(n_ctl):
            arrived = []
            for i in (0, 1):
                cmd, arr = cmd_fn(i)
                pols[i].act(cmd)
                arrived.append(arr)
            step(DECIMATION)
            if c % cap_every == 0:
                capture()
            if all(arrived) or state["t"] > ARGS.max_seconds:
                return c
        return n_ctl

    # Register the cloth in physics, open a tensor view for live positions, and grab
    # the Fabric points handle we blit the deformation into each render.
    step(1)
    cloth_view = clothmod.make_cloth_view("/World/Sheet")
    fabric_points = clothmod.make_fabric_points("/World/Sheet")
    if fabric_points is None:
        print("[demo] WARNING: cloth not in Fabric — sheet deformation may not render", flush=True)

    body_ids = {}

    def _bid(i, name):
        key = (i, name)
        if key not in body_ids:
            bid, _ = robots[i].find_bodies([name])
            body_ids[key] = int(bid[0])
        return body_ids[key]

    def hand_pos(i):
        p = robots[i].data.body_pose_w[0, _bid(i, "right_wrist_yaw_link"), :3]
        return float(p[0]), float(p[1]), float(p[2])

    def foot_z(i):
        zl = float(robots[i].data.body_pose_w[0, _bid(i, "left_ankle_roll_link"), 2])
        zr = float(robots[i].data.body_pose_w[0, _bid(i, "right_ankle_roll_link"), 2])
        return min(zl, zr)

    def diag(label):
        pts = clothmod.view_positions(cloth_view) if cloth_view is not None else np.zeros((0, 3))
        parts = [f"[diag] {label}"]
        if pts.shape[0]:
            cen = pts.mean(axis=0)
            parts.append(f"sheet_cen=({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f}) z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        for i in (0, 1):
            bx, by = locos[i].base_xy()
            hx, hy, hz = hand_pos(i)
            gap = float(np.min(np.linalg.norm(pts - np.array([hx, hy, hz]), axis=1))) if pts.shape[0] else -1.0
            # foot_z<0 means the foot is through the floor
            parts.append(f"r{i}_base=({bx:.2f},{by:.2f},z{float(robots[i].data.root_pos_w[0,2]):.2f}) "
                         f"hand=({hx:.2f},{hy:.2f},{hz:.2f}) gap={gap:.2f} foot_z={foot_z(i):.2f}")
        print("  ".join(parts), flush=True)

    # ── PHASE 0: stand + let the folded sheet settle (agile balance) ──
    print("[demo] robots find their footing; the folded sheet settles…")
    control(lambda i: (stance[i].stand(), False), n_ctl=int(1.5 / (DECIMATION * sim_dt)),
            cap_every=4, pols=stance)
    diag("after stand-settle")

    # ── PHASE 1: learned approach-walk ──
    if not ARGS.no_walk:
        if coord:
            for i in (0, 1):
                coord.invoke(i, "walkToNextCorner", direction="approach")
        print("[demo] both G1s walk to the bed under the learned policy…")
        steps = control(lambda i: locos[i].command_to(scenemod.ROBOTS[i]["manip"]),
                        n_ctl=int(8.0 / (DECIMATION * sim_dt)), cap_every=2)
        diag(f"arrived after {steps} control steps")

    # ── PHASE 2: make the bed (the policy holds the stance THROUGHOUT) ──
    # The locomotion policy actively balances each robot (feet planted) the entire
    # time the arms work. We do NOT kinematically pin the floating base — writing the
    # pelvis pose each step did not hold against gravity, so the robots fell forward
    # onto the bed. Holding the stance with the policy keeps them upright on their
    # feet while the arms reach (verified: pelvis steady ~0.72 m, hand reaches a
    # commanded point to <1 mm when well-conditioned).
    if not ARGS.walk_only:
        # Settle the gait into a double-support stand (both feet down) before planting.
        print("[demo] settling into a stance at the bedside…", flush=True)
        control(lambda i: (stance[i].stand(), False),
                n_ctl=int(1.0 / (DECIMATION * sim_dt)), cap_every=4, pols=stance)
        # Plant: hold each pelvis at its arrived pose so the arms can make the bed
        # without the free base toppling. Freeze the legs at the settled pose too, so
        # they stay planted (the stance policy is no longer needed once pinned).
        state["pin"] = [robots[i].data.root_pose_w[:1].clone() for i in (0, 1)]
        for i in (0, 1):
            robots[i].set_joint_position_target(
                robots[i].data.joint_pos[:, stance[i].leg_ids].clone(), joint_ids=stance[i].leg_ids)
        step(2)
        diag("planted firmly at the bedside")
        if ARGS.replay:
            run_replay(reps, robots, stance, coord, step, capture, diag, sim_dt)
        else:
            run_bedmaking(arms, stance, coord, step, capture, diag, stage, cloth_view, sheet)
    else:
        diag("walk-only: standing at the bedside")

    # ── report ──
    if coord:
        print("\n[demo] Device Connect event history (peer 0):")
        for e in coord.event_history(0, limit=40):
            print(f"   [{e['ts']}] {e['kind']:<14} {e['summary']}")
        goals = [coord.invoke(i, "getGoalState") for i in (0, 1)]
        print(f"[demo] goal state: {goals}")
        coord.stop()

    if ARGS.render and state["frame"] > 0:
        encode(frames_dir, state["frame"])
    print("[demo] done.")
    return 0


def run_replay(reps, robots, stance, coord, step, capture, diag, sim_dt):
    """Replay the real recorded bed-making arm + waist motion while the agile
    balance policy holds each robot's stance (legs balance, the upper body follows
    the dataset). Only waist/arms/fingers are replayed — the legs are left to the
    policy, so this also runs on the floating walk-in base. Robot 1's 180° spawn
    makes it a mirrored peer; ``--lag`` desyncs them."""
    N = reps[0].n_frames
    spf = max(1, round((1.0 / sim_dt) / reps[0].fps / max(0.1, ARGS.replay_speed)))
    ctl_per_frame = max(1, round(spf / DECIMATION))
    lag = int(ARGS.lag * reps[0].fps)
    cap_every = max(1, N // 160)

    def hold_steps(n_ctl):
        # Base is pinned and the legs are frozen at the planted pose, so the legs need
        # no per-step control here — only the waist/arms/fingers move (driven by apply).
        for _ in range(n_ctl):
            step(DECIMATION)

    print(f"[demo] arranging the sheet — real motion (episode {reps[0].source_episode}, "
          f"{N} frames, {spf} sim steps/frame)", flush=True)
    if coord:
        coord.invoke(0, "pickUpBedSheet", corner="A")
        coord.invoke(1, "pickUpBedSheet", corner="B")
    # Ease arms from rest into the first recorded pose.
    for w in range(40):
        reps[0].warmup((w + 1) / 40.0)
        reps[1].warmup((w + 1) / 40.0)
        hold_steps(1)
        if w % 8 == 0:
            capture()
    mid_done = False
    for j in range(N + lag):
        reps[0].apply(j)
        reps[1].apply(j - lag)
        hold_steps(ctl_per_frame)
        if j % cap_every == 0:
            capture()
        if coord and not mid_done and j >= N // 2:
            coord.invoke(0, "askForHelp", corner="A", reason="squaring my side")
            coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner="A")
            mid_done = True
    if coord:
        coord.invoke(0, "putDownBedSheet", corner="A")
        coord.invoke(1, "putDownBedSheet", corner="B")
    diag("after arranging")
    capture()


def run_bedmaking(arms, stance, coord, step, capture, diag, stage, cloth_view, sheet):
    """Closed-loop bed-making with the agile policy holding each robot's stance the
    whole time. Each G1 finds its near edge of the sheet from the LIVE cloth positions,
    reaches it with world-frame arm IK (floating-base jacobian + per-step clamp),
    grasps it, and pulls it out + down to spread that side taut with the 9-inch
    overhang. Robot 0 works the -y side, robot 1 the +y side. (Planted robots reach
    their near edge but not the far corners — that needs corner-to-corner walking,
    a later step.)"""
    SIDE, TOP, MX = scenemod.SIDE_Y, scenemod.BED_TOP_Z, scenemod.MANIP_X
    sign = {0: -1.0, 1: 1.0}

    def hold_stance():
        pass  # base is pinned and the legs are frozen at the planted pose

    def reach_both(t0, t1, tol=0.06, max_steps=300):
        arms[0].set_target(list(t0))
        arms[1].set_target(list(t1))
        d0 = d1 = None
        for s in range(max_steps):
            hold_stance()
            d0 = arms[0].tick()
            d1 = arms[1].tick()
            step(DECIMATION)
            if s % 6 == 0:
                capture()
            if (d0 or 9) < tol and (d1 or 9) < tol:
                break
        return d0, d1

    def settle(n_ctl):
        for s in range(n_ctl):
            hold_stance()
            step(DECIMATION)
            if s % 4 == 0:
                capture()

    def near_cloth(ref):
        pts = clothmod.view_positions(cloth_view)
        if pts.shape[0] == 0:
            return ref
        k = int(np.argmin(np.linalg.norm(pts - np.array(ref), axis=1)))
        return float(pts[k, 0]), float(pts[k, 1]), float(pts[k, 2])

    # 1) reach down to the live near edge of the sheet (find the cloth, don't guess)
    if coord:
        coord.invoke(0, "pickUpBedSheet", corner="A")
        coord.invoke(1, "pickUpBedSheet", corner="B")
    g = {i: near_cloth((MX, sign[i] * (SIDE - 0.05), TOP + 0.04)) for i in (0, 1)}
    print(f"[demo] reaching the near sheet edge: r0->{tuple(round(v,2) for v in g[0])} "
          f"r1->{tuple(round(v,2) for v in g[1])}", flush=True)
    # Hover the hand just ABOVE the near edge — reaching all the way down to the bed
    # surface (z≈0.69) over-extends the arm and the IK stalls ~10 cm short. Hovering
    # at a comfortable height keeps the reach in-workspace; the grasp's bind offset
    # (0.14 m) catches the draped cloth below the hand.
    d = reach_both((g[0][0], g[0][1], max(g[0][2] + 0.10, 0.78)),
                   (g[1][0], g[1][1], max(g[1][2] + 0.10, 0.78)),
                   tol=0.05, max_steps=300)
    diag(f"reached edge (ik dist {d})")

    # 2) grasp the cloth at each hand (PhysX attachment binds the overlapping cloth)
    for i in (0, 1):
        clothmod.grasp(stage, "/World/Sheet", f"/World/envs/env_0/Robot_{i}/right_wrist_yaw_link",
                       f"/World/grasp_{i}", bind_offset=0.14)
    settle(12)
    diag("grasped the sheet edge")

    # 3) gentle outward + down drag FROM where each hand actually grasped, to spread
    # that side taut with the overhang. A relative tug (rather than a far world
    # target) keeps the pull inside the arm's comfortable workspace so the reaction
    # force does not yank the policy-balanced base off its feet.
    print("[demo] pulling the sheet out to spread it over the bed…", flush=True)
    h = {i: arms[i].ee_pos() for i in (0, 1)}
    s = {i: (h[i][0], h[i][1] + sign[i] * 0.14, h[i][2] - 0.12) for i in (0, 1)}
    reach_both(s[0], s[1], tol=0.06, max_steps=240)
    diag("spread to the side")

    # 4) release + a help exchange over Device Connect, then settle
    for i in (0, 1):
        clothmod.release(stage, f"/World/grasp_{i}")
    if coord:
        coord.invoke(0, "askForHelp", corner="A", reason="squaring my side")
        coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner="A")
        coord.invoke(0, "putDownBedSheet", corner="A")
        coord.invoke(1, "putDownBedSheet", corner="B")
    settle(60)
    diag("after bed-making")
    capture()


def encode(frames_dir, n):
    import shutil
    import subprocess
    out = frames_dir / "isaac_bed_making.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-pattern_type", "glob",
                        "-i", str(frames_dir / "frame_*.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", str(out)],
                       capture_output=True)
        print(f"[demo] wrote {out} ({n} frames)")
    else:
        print(f"[demo] {n} frames in {frames_dir} (install ffmpeg for mp4)")


if __name__ == "__main__":
    import traceback

    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if ARGS.gui:
        # Keep the window open so you can orbit/inspect after the run; close it
        # (or Ctrl-C) to exit.
        print("[demo] GUI open — close the window or press Ctrl-C to exit.", flush=True)
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass
        simulation_app.close()
    else:
        # Isaac's replicator orchestrator can hang inside simulation_app.close() on
        # headless shutdown with cameras; artifacts are already written, so hard-exit.
        os._exit(rc)
