"""Two Unitree G1 humanoids autonomously make a bed in NVIDIA Isaac Sim,
coordinating as equal peers over Arm Device Connect.

Run it with the Isaac Lab Python on the DGX Spark::

    cd ~/IsaacLab
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    ./isaaclab.sh -p ~/workspaces/git/robots/examples/isaac_bed_making/demo.py \
        --loopback --render

What it shows:

* Two G1s (Dex3 hands) planted at opposite long sides of a bed, with a PhysX
  particle-cloth sheet draped over it.
* Each G1 runs its own loop: it reaches its near edge of the sheet with
  world-frame differential IK, grasps the cloth (a PhysX attachment to its
  palm), and squares/smooths its side — both peers working in parallel, exactly
  like the Figure Helix bedroom-tidy clip.
* Coordination is real Device Connect: each robot claims its work and emits
  events; when one finishes and its peer is still ``stuck`` on a corner, it
  **offers help** and reaches across to hold it. With ``--broker`` both peers
  register live on the dashboard (callable functions, event stream).

This is a self-contained example: it does not modify ``strands_robots`` or Arm's
Device Connect. It reuses the swarm driver from
``examples/unitree_g1_bed_making_g1_driver.py``.
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
    p.add_argument("--render", action="store_true", help="Capture frames and encode an mp4.")
    p.add_argument("--frames-dir", default=str(REPO_ROOT / "artifacts" / "isaac_bed_making"))
    p.add_argument("--nats-url", default=os.environ.get("DEVICE_CONNECT_NATS_URL", "nats://fabric.deviceconnect.dev:4222"))
    p.add_argument("--max-seconds", type=float, default=60.0, help="Safety cap on sim wall-time.")
    return p.parse_known_args()[0]


ARGS = parse_args()

# ── Launch the simulator first (Isaac Lab requires this before other imports) ──
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "enable_cameras": bool(ARGS.render)})
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: E402
import omni.usd  # noqa: E402

from examples.isaac_bed_making import cloth as clothmod  # noqa: E402
from examples.isaac_bed_making import scene as scenemod  # noqa: E402
from examples.isaac_bed_making.manipulation import ArmIK  # noqa: E402


def main() -> int:
    frames_dir = Path(ARGS.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── scene ──
    SceneCfg = scenemod.build_scene_cfg(G1_29DOF_CFG)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=6.0))

    cam = None
    if ARGS.render:
        from isaaclab.sensors.camera import Camera, CameraCfg
        cam = Camera(cfg=CameraCfg(
            prim_path="/World/CameraSensor", update_period=0,
            height=scenemod.CAM_RES[1], width=scenemod.CAM_RES[0], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=400.0,
                                             horizontal_aperture=20.955, clipping_range=(0.05, 1.0e5))))

    sim.reset()
    stage = omni.usd.get_context().get_stage()
    robots = [scene["robot_0"], scene["robot_1"]]
    sim_dt = sim.get_physics_dt()

    if cam is not None:
        cam.set_world_poses_from_view(torch.tensor([scenemod.CAM_EYE], device=sim.device),
                                      torch.tensor([scenemod.CAM_TARGET], device=sim.device))

    # ── cloth bedsheet ──
    scene_path = clothmod.find_physics_scene_path(stage)
    clothmod.enable_gpu_dynamics(stage, scene_path)
    sheet = clothmod.build_bedsheet(stage, scene_path, "/World/Sheet",
                                    size=scenemod.SHEET_SIZE, resolution=scenemod.SHEET_RES,
                                    origin=scenemod.SHEET_ORIGIN, color=(0.86, 0.86, 0.92))

    # ── arms ──
    arms = {0: ArmIK(robots[0], scene, "right", sim.device),
            1: ArmIK(robots[1], scene, "right", sim.device)}
    robot_prims = {0: "/World/envs/env_0/Robot_0", 1: "/World/envs/env_0/Robot_1"}

    # ── Device Connect swarm ──
    coord = None
    if not ARGS.no_device_connect:
        from examples.isaac_bed_making.coordination import SwarmCoordinator
        coord = SwarmCoordinator(mode="broker" if ARGS.broker else "loopback", nats_url=ARGS.nats_url)
        ids = coord.start()
        print(f"[demo] Device Connect swarm online ({coord.mode}): {ids}")

    # ── helpers ──
    state = {"frame": 0, "t": 0.0}

    def step(n=1):
        # Step physics WITHOUT rendering every frame (rendering 2 robots + cloth
        # each step is the bottleneck); we render only when capturing.
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            state["t"] += sim_dt

    def capture():
        if cam is None:
            return
        sim.render()
        cam.update(dt=sim_dt)
        out = cam.data.output["rgb"]
        if out is None or out.shape[0] == 0:
            return
        rgb = out[0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb).save(frames_dir / f"frame_{state['frame']:04d}.png")
        state["frame"] += 1
        if state["frame"] % 10 == 0:
            print(f"[demo] captured {state['frame']} frames (t={state['t']:.1f}s)", flush=True)

    def reach(idx, target, tol=0.06, max_steps=300, cap_every=10):
        """Drive robot idx's arm to a world target until within tol or timeout."""
        arms[idx].set_target(list(target))
        d = None
        for s in range(max_steps):
            d = arms[idx].tick()
            step()
            if cap_every and s % cap_every == 0:
                capture()
            if d is not None and d < tol:
                break
            if state["t"] > ARGS.max_seconds:
                break
        return d

    def reach_both(t0, t1, tol=0.06, max_steps=300):
        """Drive both arms toward their targets in parallel."""
        arms[0].set_target(list(t0)); arms[1].set_target(list(t1))
        d0 = d1 = None
        for s in range(max_steps):
            d0 = arms[0].tick(); d1 = arms[1].tick()
            step()
            if s % 10 == 0:
                capture()
            if (d0 or 9) < tol and (d1 or 9) < tol:
                break
            if state["t"] > ARGS.max_seconds:
                break
        return d0, d1

    # ── choreography (sequence gated by IK convergence + DC, not a fixed clock) ──
    print("[demo] settling the bedsheet onto the bed…")
    for _ in range(160):
        step()
        if _ % 12 == 0:
            capture()

    # Each robot's near grab point + its smoothing pull point (from scene geometry).
    grab = scenemod.GRAB_POINTS
    smooth = scenemod.SMOOTH_POINTS

    # 1) Both peers claim and reach their near sheet edge in parallel.
    if coord:
        coord.invoke(0, "pickUpBedSheet", corner="A")
        coord.invoke(1, "pickUpBedSheet", corner="B")
    print("[demo] both peers reach + grasp their side of the sheet…")
    reach_both(grab[0], grab[1], tol=0.05, max_steps=320)
    capture()

    # 2) Grasp the cloth with each palm.
    clothmod.grasp(stage, "/World/Sheet", arms[0].palm_path(robot_prims[0]), "/World/Sheet/grasp_0")
    clothmod.grasp(stage, "/World/Sheet", arms[1].palm_path(robot_prims[1]), "/World/Sheet/grasp_1")
    step(12)
    capture()

    # 3) Both pull their edge outward to square the cover (bilateral, in sync).
    if coord:
        coord.invoke(0, "walkToNextCorner", direction="counterclockwise")
        coord.invoke(1, "walkToNextCorner", direction="counterclockwise")
    print("[demo] both peers square the cover…")
    reach_both(smooth[0], smooth[1], tol=0.06, max_steps=260)
    if coord:
        coord.invoke(0, "putDownBedSheet", corner="A")
        coord.invoke(1, "putDownBedSheet", corner="B")
    capture()

    # 4) Robot 0 is "stuck": its corner drifts as it lets go -> asks for help.
    #    Robot 1 (auto_offer) responds and reaches across to hold it.
    if coord:
        print("[demo] robot 0 is stuck (a corner is lifting) — asking the swarm for help…")
        coord.invoke(0, "askForHelp", corner="A", reason="placed corner A is lifting")
        # robot 1 releases its own and crosses to hold robot 0's corner
        clothmod.release(stage, "/World/Sheet/grasp_1")
        hold_pt = (grab[0][0] + 0.0, grab[0][1] + 0.10, grab[0][2])
        reach(1, hold_pt, tol=0.07, max_steps=240)
        clothmod.grasp(stage, "/World/Sheet", arms[1].palm_path(robot_prims[1]), "/World/Sheet/grasp_help")
        step(20); capture()

    # settle + final frames
    for _ in range(80):
        step()
        if _ % 10 == 0:
            capture()

    # ── report ──
    if coord:
        print("\n[demo] Device Connect event history (peer 0):")
        for e in coord.event_history(0, limit=40):
            print(f"   [{e['ts']}] {e['kind']:<14} {e['summary']}")
        print("\n[demo] help history (peer 0):")
        for h in coord.help_history(0, limit=20):
            print(f"   [{h['ts']}] {h['kind']:<8} {h['summary']}")
        goals = [coord.invoke(i, "getGoalState") for i in (0, 1)]
        print(f"\n[demo] goal state: {goals}")
        coord.stop()

    # ── encode mp4 ──
    if ARGS.render and state["frame"] > 0:
        import shutil
        import subprocess
        out = frames_dir / "isaac_bed_making.mp4"
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-pattern_type", "glob",
                            "-i", str(frames_dir / "frame_*.png"),
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", str(out)],
                           capture_output=True)
            print(f"[demo] wrote {out} ({state['frame']} frames)")
        else:
            print(f"[demo] {state['frame']} frames in {frames_dir} (install ffmpeg for mp4)")

    print("[demo] done.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        simulation_app.close()
    raise SystemExit(rc)
