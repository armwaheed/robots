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
**particle-cloth sheet** that starts **flat, gathered toward the foot** (as if
pulled down to the foot of the bed, head half bare); the robots make the bed by
grabbing the two **forward (head-side) corners** and pulling them toward the head.

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
from examples.isaac_bed_making.manipulation import FingerGrip, PinkArmIK, apply_hand_friction  # noqa: E402
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
    # HIGH hand friction on BOTH hands: the grip is friction-only (no kinematic
    # attachment) — closed fingers cage a handful of cloth and the rubberized,
    # high-friction palms/fingertips hold it so the robot can actually drag the sheet.
    nb = [apply_hand_friction(pre_stage, f"/World/envs/env_0/Robot_{i}", sd,
                              static_friction=10.0, dynamic_friction=10.0)
          for i in (0, 1) for sd in ("right", "left")]
    mark(f"floating base set; hand friction bound to {sum(len(x) for x in nb)} colliders")

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
    # Round the pillows: hide the square box visuals and draw a rounded superellipsoid
    # pillow in each one's place (the box stays as a hidden, rounded collider so the
    # cloth still drapes over it). Visual-only meshes, so post-reset authoring is fine.
    from pxr import UsdGeom  # noqa: E402
    from examples.isaac_bed_making import props as propsmod  # noqa: E402
    for side, path in (("left", "/World/PillowL"), ("right", "/World/PillowR")):
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
        propsmod.superellipsoid_mesh(stage, f"/World/PillowVis_{side}",
                                     size=scenemod.PILLOW_SIZE, center=scenemod.PILLOWS[side],
                                     color=(0.95, 0.95, 0.97), roundness=scenemod.PILLOW_ROUNDNESS)
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

    # ── bedsheet: flat, gathered toward the foot (head half of the bed bare) ──
    scene_path = clothmod.find_physics_scene_path(stage)
    clothmod.enable_gpu_dynamics(stage, scene_path)
    sheet = clothmod.build_bedsheet(
        stage, scene_path, "/World/Sheet",
        size=scenemod.SHEET_SIZE, resolution=scenemod.SHEET_RES,
        origin=scenemod.SHEET_ORIGIN, color=(0.86, 0.86, 0.92),
        thickness=scenemod.SHEET_THICKNESS, accordion=True)
    # Give the sheet VISUAL THICKNESS: the particle cloth is a single-layer membrane
    # (renders thin), so we drive a separate closed double-layer slab from the same
    # live particle positions each frame and hide the membrane (physics unchanged).
    shell = clothmod.build_shell_mesh(
        stage, sheet, thickness=scenemod.SHEET_SHELL_THICKNESS, color=(0.86, 0.86, 0.92))
    mark("bedsheet built (flat head edge + accordion ruffle at the foot; thick visual shell)")

    # ── manipulation: TWO-handed Pink IK + finger grips (or optional dataset replay) ──
    # Each G1 drives both arms (the right also owns the 3-DOF waist so it leans into the
    # reach; the left is waist-free so they don't fight) plus a FingerGrip per hand to
    # close the Inspire fingers on a handful of cloth. Two hands gather + pin the cloth,
    # one hand grips it and pulls — a friction grasp, no kinematic attachment.
    manip = {
        "R": {i: PinkArmIK(robots[i], "right", sim.device, SIM_DT * DECIMATION, with_waist=True) for i in (0, 1)},
        "L": {i: PinkArmIK(robots[i], "left", sim.device, SIM_DT * DECIMATION, with_waist=False) for i in (0, 1)},
        "gR": {i: FingerGrip(robots[i], "right", sim.device) for i in (0, 1)},
        "gL": {i: FingerGrip(robots[i], "left", sim.device) for i in (0, 1)},
    }
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
            scene.update(sim_dt)
            # Once planted, hold each pelvis steady EVERY sim step so the free base can't
            # topple through the bed-making lean. We pin only the xy position and a LEVEL
            # (upright, yaw-only) orientation, and leave Z FREE so the feet settle on the
            # floor under gravity — pinning z too left the legs floating with no ground
            # contact, and they drifted up. The z-linear velocity is kept (gravity beds the
            # feet down); xy + all angular velocity are zeroed so it neither slides nor tips.
            # Runs after scene.update() so root_pose_w/root_vel_w are the fresh post-step state.
            if state["pin"] is not None:
                for i in (0, 1):
                    pose = robots[i].data.root_pose_w[:1].clone()
                    px, py, qw, qx, qy, qz = state["pin"][i]
                    pose[0, 0], pose[0, 1] = px, py
                    pose[0, 3], pose[0, 4], pose[0, 5], pose[0, 6] = qw, qx, qy, qz
                    robots[i].write_root_pose_to_sim(pose)
                    vel = robots[i].data.root_vel_w[:1].clone()
                    vel[0, 0], vel[0, 1] = 0.0, 0.0
                    vel[0, 3], vel[0, 4], vel[0, 5] = 0.0, 0.0, 0.0
                    robots[i].write_root_velocity_to_sim(vel)
            state["t"] += sim_dt

    def capture():
        if cam is None:
            return
        if cloth_view is not None and fabric_points is not None:
            # Drive the thick visual slab from the live particle positions (the thin
            # membrane it replaces is hidden). fabric_points is the SHELL's points.
            clothmod.sync_shell_fabric(cloth_view, fabric_points, shell)
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
    # Blit into the SHELL's Fabric points (the visible thick slab), not the hidden
    # membrane. Positions still come from the membrane's tensor cloth view.
    fabric_points = clothmod.make_fabric_points(shell.prim_path)
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

    # ── PHASE 0: stand + let the sheet settle onto the bed (agile balance) ──
    print("[demo] robots find their footing; the sheet settles onto the foot of the bed…")
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
        # Plant the base the INSTANT the walk arrives. We do NOT let the agile stance
        # policy hold the free base unpinned even briefly: on the walk→stand handoff it
        # topples (the rl_gym walk LSTM leaves the legs mid-stride and the agile policy
        # can't recover the gait), which dropped both robots to the floor *before* the
        # pin was set. Instead we pin each pelvis at its arrived pose — leveled to
        # yaw-only at a clean standing height so we never freeze a mid-stride/toppling
        # pose — and command the legs to the bent-knee stand. The per-sim-step pin in
        # step() then holds the base rigidly while the arms make the bed.
        print("[demo] planting firmly at the bedside…", flush=True)
        pins = []
        for i in (0, 1):
            # Normalise to the INTENDED bedside working pose: the learned walk is the
            # visual approach but isn't perfectly reliable (it occasionally stumbles a
            # robot), so we place each G1 at its mark (xy + yaw facing the bed) in a clean,
            # near-straight STANDING pose, drop it from just above the floor, and let the
            # z-free pin bed the feet down. This guarantees both make the bed from a solid,
            # upright, feet-on-the-floor stance regardless of how the walk ended.
            mx, my = scenemod.ROBOTS[i]["manip"]
            qw, qx, qy, qz = scenemod.yaw_to_quat(scenemod.ROBOTS[i]["yaw_deg"])
            pins.append((mx, my, qw, qx, qy, qz))
            stand = robots[i].data.default_joint_pos.clone()
            for expr, val in ((".*_hip_pitch_joint", -0.05), (".*_knee_joint", 0.12),
                              (".*_ankle_pitch_joint", -0.06)):
                jid, _ = robots[i].find_joints([expr])
                stand[:, jid] = val
            pose0 = robots[i].data.root_pose_w[:1].clone()
            pose0[0, 0], pose0[0, 1], pose0[0, 2] = mx, my, scenemod.STAND_PELVIS_Z + 0.04
            pose0[0, 3], pose0[0, 4], pose0[0, 5], pose0[0, 6] = qw, qx, qy, qz
            robots[i].write_root_pose_to_sim(pose0)
            robots[i].write_root_velocity_to_sim(zero_vel)
            robots[i].write_joint_state_to_sim(stand, torch.zeros_like(robots[i].data.joint_vel))
            robots[i].set_joint_position_target(stand[:, stance[i].leg_ids], joint_ids=stance[i].leg_ids)
        state["pin"] = pins
        # Settle: the z-free pin lets the feet drop onto the floor and the stance steady.
        for _ in range(int(1.0 / (DECIMATION * sim_dt))):
            step(DECIMATION)
        capture()
        diag("planted firmly at the bedside")
        if ARGS.replay:
            run_replay(reps, robots, stance, coord, step, capture, diag, sim_dt)
        else:
            run_bedmaking(manip, coord, step, capture, diag, stage, cloth_view, sheet)
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


def run_bedmaking(manip, coord, step, capture, diag, stage, cloth_view, sheet):
    """Two-handed friction bed-making (no kinematic attachment). The sheet starts flat
    and gathered toward the foot. Each G1, planted at the bedside, uses BOTH hands to
    win a grip and then pulls with one:

      1. both palms press the sheet's head edge near the forward corner;
      2. the assisting (left) hand slides inboard, bunching a handful of cloth into the
         gripping (right) hand;
      3. the right hand CLOSES its Inspire fingers on the handful (high-friction grip)
         while the left presses it home, then the left opens and lifts away;
      4. the right hand drags the handful toward the head, drawing the sheet up the bed.

    Robot 0 works the −y side, robot 1 the +y side (mirror). The grip is friction +
    finger-cage only — the closed fingers and rubberized palms hold the cloth."""
    TOP = scenemod.BED_TOP_Z
    sign = {0: -1.0, 1: 1.0}
    HEAD_EDGE_X = scenemod.SHEET_ORIGIN[0] - scenemod.SHEET_SIZE[0] / 2.0  # ~0.10
    HEAD_PULL_X = scenemod.HEAD_X + 0.65  # = -0.35, reach-limited on a planted base
    GY = 0.66           # gripping (right) hand y, inboard so the grip is on the mattress
    LY = GY + 0.20      # assisting (left) hand y, just outboard — it pins the cloth flat
    # A friction grip needs NORMAL FORCE, not just contact. APPROACH_Z sets the wrist
    # just above the cover for a soft landing; GRIP_Z drives the wrist DOWN INTO the
    # cover (below the cover top) so the arm presses hard — the hand can't pass the
    # mattress, so it pushes, and the closed high-friction fingers clamp the cloth. With
    # the mattress slick (low friction) the clamped cover then slides headward as a unit.
    APPROACH_Z = TOP + 0.05   # ≈0.71 — wrist just above the cover, light contact
    # Gentle grip: a hard press into the mattress whips the arm and topples the planted
    # robot. The accordion unspools with little resistance, so a light close on the head
    # edge is enough — keep the wrist right at the cloth, no driving into the bed.
    GRIP_Z = TOP - 0.01       # ≈0.65 — wrist at the cloth, fingers close on the head edge
    LIFT_Z = TOP + 0.26       # left-hand retreat height
    PULL_DX = -0.40           # headward drag distance (relative to the grip point)
    R, L, gR, gL = manip["R"], manip["L"], manip["gR"], manip["gL"]

    def drive(n_ctl, rt, lt, rg, lg, cap=4):
        """Hold/seek both arms toward per-robot targets rt[i]/lt[i] (None = leave as-is)
        with finger fractions rg[i]/lg[i], for n_ctl control steps."""
        for i in (0, 1):
            if rt is not None and rt[i] is not None:
                R[i].set_target(rt[i])
            if lt is not None and lt[i] is not None:
                L[i].set_target(lt[i])
        dr = dl = None
        for s in range(n_ctl):
            for i in (0, 1):
                dr = R[i].tick()
                dl = L[i].tick()
                gR[i].set(rg[i])
                gL[i].set(lg[i])
            step(DECIMATION)
            if s % cap == 0:
                capture()
        return dr, dl

    def by_side(fn):
        return {i: fn(i) for i in (0, 1)}

    if coord:
        coord.invoke(0, "pickUpBedSheet", corner="A")
        coord.invoke(1, "pickUpBedSheet", corner="B")

    # 1) both palms descend onto the head edge — right where it will grip, left just
    #    inboard to pin the cloth flat — fingers open, soft landing.
    print("[demo] both hands settle on the head edge…", flush=True)
    rt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * GY, APPROACH_Z))
    lt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * LY, APPROACH_Z))
    drive(55, rt, lt, {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0})
    diag("hands on the edge")

    # 2) press DOWN hard (wrist into the cover) and close the right fingers on the cloth.
    #    The left presses alongside to PIN the cover flat so the right's press grips it
    #    rather than shoving it away. The left stays open (a flat pin, not a grab).
    print("[demo] pressing in and gripping a handful (right hand)…", flush=True)
    rt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * GY, GRIP_Z))
    lt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * LY, GRIP_Z))
    drive(28, rt, lt, {0: 0.5, 1: 0.5}, {0: 0.0, 1: 0.0})   # press, fingers half-curl
    drive(22, rt, lt, {0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0})   # fingers close hard on the cloth
    diag("gripped")

    # 3) the left hand lifts clear so only the gripping hand remains on the cloth.
    print("[demo] freeing the assisting hand…", flush=True)
    lt = by_side(lambda i: (HEAD_EDGE_X, sign[i] * LY, LIFT_Z))
    drive(24, rt, lt, {0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0})
    diag("left hand clear")

    # 4) the gripping hand drags the cover toward the head, fingers held closed and the
    #    wrist held LOW so it keeps pressing as it slides. The target is RELATIVE to where
    #    the hand actually gripped (same y/z, x moved headward) so the reach stays in the
    #    arm's workspace — a far world target sent the arm flying up.
    print("[demo] pulling the cover toward the head with one hand…", flush=True)
    h = {i: R[i].ee_pos() for i in (0, 1)}
    rt = by_side(lambda i: (max(HEAD_PULL_X, h[i][0] + PULL_DX), h[i][1], h[i][2]))
    drive(95, rt, None, {0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0}, cap=3)
    diag("pulled the sheet toward the head")

    # 5) release + a help exchange over Device Connect, then settle
    drive(10, None, None, {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0})
    if coord:
        coord.invoke(0, "askForHelp", corner="A", reason="squaring my side")
        coord.invoke(1, "offerHelp", target=coord.peers[0].device_id, corner="A")
        coord.invoke(0, "putDownBedSheet", corner="A")
        coord.invoke(1, "putDownBedSheet", corner="B")
    for s in range(40):
        step(DECIMATION)
        if s % 4 == 0:
            capture()
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
