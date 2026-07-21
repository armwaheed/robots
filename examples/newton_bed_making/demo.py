"""Two Unitree G1s make a bed together on the Isaac Lab 3.0 Newton (MJWarp) backend, coordinating
over MHS.

This is a bespoke single-scene render, NOT the RL play harness (which replicates one-robot envs and
so can only ever show "two robots near two grey boxes"). Two G1 articulations flank one correctly
proportioned bed + headboard + two propped pillows, each driven by the trained whole-body bed-reach
policy; they reach onto the bed and draw the cover headward toward the pillows while exchanging MHS
procedure calls. Both robots balance on their own two feet — the free base is the default; the reach
policy holds balance the same way it does in its training env (see ``--base-hold`` for the gantry
fallback). Headless capture is via Newton's own GL viewer (this host has no Omniverse Kit / Isaac
Sim), so the render is Newton-native end to end.

    MUJOCO_GL=egl ./isaaclab.sh -p demo.py --headless --device cuda:0 \
        --policy /path/to/optionA_work/checkpoints/exported/policy.pt \
        --mhs loopback --seconds 8 --out newton_bed_making.mp4

Cloth note: the issue's real soft sheet uses Isaac Lab's coupled MJWarp+VBD solver, whose deformable
spawner hard-requires ``omni.physx`` (Kit), which is unavailable here without a risky Isaac Sim
install. Per the issue's stated fallback, the sheet is a VISIBLE PROXY: a thin, bed-width cover the
robots draw headward. Everything else (bed, headboard, pillows, two policy-driven robots, MHS) is real.
"""

from __future__ import annotations

import argparse

from isaaclab_tasks.utils import add_launcher_args

parser = argparse.ArgumentParser(description="Two-G1 Newton bed-making demo.")
parser.add_argument("--policy", type=str, default="", help="Path to the exported bed-reach policy.pt. Empty = hold the default pose (scene smoke).")
parser.add_argument("--out", type=str, default="newton_bed_making.mp4")
parser.add_argument("--trace-out", type=str, default="mhs_trace.json")
parser.add_argument("--seconds", type=float, default=8.0, help="Total demo length (s).")
parser.add_argument("--fps", type=int, default=30, help="Output video fps (frames are sub-sampled to this).")
parser.add_argument("--mhs", choices=["loopback", "none"], default="loopback")
parser.add_argument("--no-mirror", action="store_true", help="Drive the +y robot with the raw left-hand policy instead of the bilateral mirror.")
parser.add_argument("--base-hold", action="store_true", help="Pin each pelvis at its bedside pose (a gantry) and freeze the legs — a fallback if the free-base reach drifts.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets import G1_MINIMAL_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import geometry as geo

DEVICE = args_cli.device or "cuda:0"
SIM_DT = 0.005
DECIMATION = 4                       # 50 Hz control (200 Hz sim), matching training
CONTROL_DT = SIM_DT * DECIMATION     # 0.02 s

# Friction 1.0/1.0 matches the training terrain's material; the Newton default is slippery enough to
# slide the balancing policy's feet out from under it.
_GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)

sim_cfg = sim_utils.SimulationCfg(
    dt=SIM_DT,
    device=DEVICE,
    physics_material=_GROUND_MATERIAL,
    physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=95, nconmax=40, cone="pyramidal", impratio=1, integrator="implicitfast"
        ),
        num_substeps=1,
    ),
)

# Per-prop render colours. Kitless, USD PreviewSurface/OmniPBR materials are skipped, so props render
# neutral grey; Newton's viewer instead reads ``model.shape_color``, which we set by prim path (see
# ``color_newton_shapes``) so the sheet reads distinctly from the mattress.
SHAPE_COLORS = {
    "/World/Bed": (0.46, 0.33, 0.24),        # tan mattress
    "/World/Headboard": (0.34, 0.24, 0.17),  # darker wood headboard
    "/World/PillowL": (0.88, 0.86, 0.90),    # pale pillows
    "/World/PillowR": (0.88, 0.86, 0.90),
    "/World/Cover": (0.93, 0.94, 0.99),      # white sheet, distinct from the mattress
}

# Reach targets in the pelvis / canonical left-hand frame. The trained box is forward 0.15..0.45,
# lateral(left) 0..0.30, up/down -0.20..0.15; targets near the forward edge make the policy lean past
# its support and topple, so we stay mid-box — the native play env holds ~(0.20, 0.24, -0.04) upright.
GRAB_TARGET = (0.22, 0.10, -0.08)    # ease onto the near corner of the cover: forward + slightly left + down
DRAW_TARGET = (0.24, 0.30, -0.02)    # sweep headward (lateral -> +0.30) and lift toward the pillows


# ── scene ─────────────────────────────────────────────────────────────────────
def _spawn_static_box(prim_path, size, center, rot=None):
    """Spawn a static collidable cuboid (a Newton collision shape, so the Newton GL viewer draws it).
    ``rot`` is an (x, y, z, w) quaternion used to prop the pillows on their edge."""
    spawn = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
    )
    spawn.func(prim_path, spawn, translation=center, orientation=rot)


def _robot(idx):
    """One G1 at its bedside mark, facing the bed."""
    spec = geo.ROBOTS[idx]
    cfg = G1_MINIMAL_CFG.replace(
        prim_path=f"/World/Robot{idx}",
        init_state=G1_MINIMAL_CFG.init_state.replace(pos=spec["pos"], rot=geo.yaw_to_quat(spec["yaw_deg"])),
    )
    return Articulation(cfg)


def build_scene():
    """Spawn ground + lights + bed furniture + the cover proxy + two G1s. Returns the robots, the
    cover rigid body, and the cover's rest centre (the cover slides in -x from there during the draw)."""
    for path, cfg in (
        ("/World/ground", sim_utils.GroundPlaneCfg(physics_material=_GROUND_MATERIAL)),
        ("/World/light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.95))),
        ("/World/key", sim_utils.DistantLightCfg(intensity=2000.0, angle=2.0)),
    ):
        cfg.func(path, cfg)

    _spawn_static_box("/World/Bed", geo.BED_SIZE, geo.BED_CENTER)
    _spawn_static_box("/World/Headboard", geo.HEADBOARD_SIZE, geo.HEADBOARD_CENTER)
    _spawn_static_box("/World/PillowL", geo.PILLOW_SIZE, geo.PILLOWS["left"], rot=geo.roty_to_quat(geo.PILLOW_PROP_DEG))
    _spawn_static_box("/World/PillowR", geo.PILLOW_SIZE, geo.PILLOWS["right"], rot=geo.roty_to_quat(geo.PILLOW_PROP_DEG))

    # Cover: a thin kinematic proxy the demo slides headward by hand. It needs a collider to exist as a
    # Newton body (a colliderless rigid body is not registered), so its footprint is kept off the
    # robots' stances (geometry keeps the cover half-width below the robots' |y|).
    cover_center = (geo.SHEET_HEAD_X + geo.SHEET_LEN / 2.0, 0.0, geo.SHEET_REST_Z)
    cover = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Cover",
            spawn=sim_utils.CuboidCfg(
                size=(geo.SHEET_LEN, geo.SHEET_WIDTH, geo.SHEET_THICKNESS),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=cover_center),
        )
    )
    robots = [_robot(0), _robot(1)]
    return robots, cover, cover_center


def color_newton_shapes():
    """Override the Newton model's per-shape colours by prim path (best-effort; see SHAPE_COLORS)."""
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager.get_model()
        colors = model.shape_color.numpy()
        for i, label in enumerate(model.shape_label):
            for prefix, rgb in SHAPE_COLORS.items():
                if label.startswith(prefix):
                    colors[i] = rgb
                    break
        model.shape_color = wp.array(colors, dtype=wp.vec3, device=model.device)
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] shape colouring skipped: {exc}", flush=True)


# ── choreography ────────────────────────────────────────────────────────────
def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def reach_schedule(tick, n_ticks):
    """Map a control tick to (pelvis-frame reach target, draw fraction 0..1, MHS beat name | None).

    Phases (fractions of the run): settle -> reach onto the cover -> draw headward -> hold. is_reach
    is always 1 (the policy balances-and-reaches, as in its play env)."""
    t_reach, t_draw = int(0.35 * n_ticks), int(0.80 * n_ticks)
    if tick < t_reach:
        return GRAB_TARGET, 0.0, ("pick" if tick == int(0.30 * n_ticks) else None)
    if tick < t_draw:
        s = (tick - t_reach) / max(1, t_draw - t_reach)
        return _lerp(GRAB_TARGET, DRAW_TARGET, s), s, ("help" if tick == int(0.60 * n_ticks) else None)
    return DRAW_TARGET, 1.0, ("down" if tick == t_draw else None)


def run_mhs_beat(coord, name):
    """Emit the MHS procedure calls for a choreography beat (no-op if coordination is disabled)."""
    if coord is None:
        return
    peer1 = coord.peers[1].device_id
    if name == "pick":
        coord.invoke(0, "pickUpBedSheet", corner="SW", effect="robot 0 grips its head corner")
        coord.invoke(1, "pickUpBedSheet", corner="NW", effect="robot 1 grips its head corner")
    elif name == "help":
        coord.invoke(0, "askForHelp", corner="SW", reason="squaring the head edge")
        coord.invoke(1, "offerHelp", target=peer1, corner="SW", effect="robot 1 joins the head edge")
    elif name == "down":
        coord.invoke(0, "putDownBedSheet", corner="SW")
        coord.invoke(1, "putDownBedSheet", corner="NW")


def make_drivers(robots):
    """Load the exported policy and wrap one BedReachPolicy per robot (None if no --policy)."""
    if not args_cli.policy:
        return None
    policy = torch.jit.load(args_cli.policy, map_location=DEVICE)
    policy.eval()
    from bed_reach import BedReachPolicy
    return [
        BedReachPolicy(policy, r, DEVICE, mirror=(geo.ROBOTS[i]["mirror"] and not args_cli.no_mirror))
        for i, r in enumerate(robots)
    ]


def make_gantry(robots):
    """(held_pose, leg_ids) for the opt-in base-hold gantry, or (None, None).

    The reach policy, reading a permanently-pinned (zero-velocity) base, winds the lower body into
    extreme poses, so under the gantry the legs are held at the stance default and the policy drives
    only the upper body."""
    if not args_cli.base_hold:
        return None, None
    held_pose = [
        torch.tensor([[*geo.ROBOTS[i]["pos"], *geo.yaw_to_quat(geo.ROBOTS[i]["yaw_deg"])]],
                     dtype=torch.float32, device=DEVICE)
        for i in range(len(robots))
    ]
    leg_ids = torch.tensor(
        [i for i, n in enumerate(robots[0].joint_names) if any(t in n for t in ("hip", "knee", "ankle"))],
        dtype=torch.long, device=DEVICE,
    )
    return held_pose, leg_ids


def make_coordinator():
    """Start the in-process MHS swarm (loopback), or None if disabled/unavailable."""
    if args_cli.mhs != "loopback":
        return None
    try:
        from coordination import SwarmCoordinator
        coord = SwarmCoordinator(mode="loopback")
        print("[demo] MHS swarm online:", coord.start(), flush=True)
        return coord
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] MHS unavailable ({exc}); continuing without coordination", flush=True)
        return None


def make_recorder():
    """A headless Newton-GL RGB recorder framed on the whole bed + both robots."""
    from isaaclab_newton.video_recording.newton_gl_perspective_video import create_newton_gl_perspective_video
    from isaaclab_newton.video_recording.newton_gl_perspective_video_cfg import NewtonGlPerspectiveVideoCfg
    return create_newton_gl_perspective_video(
        NewtonGlPerspectiveVideoCfg(
            window_width=geo.CAM_RES[0], window_height=geo.CAM_RES[1], eye=geo.CAM_EYE, lookat=geo.CAM_LOOKAT
        )
    )


def drive_robots(drivers, robots, target_b, leg_ids):
    """Set this tick's joint targets — the policy reach, or the default pose for a scene smoke."""
    if drivers is None:
        for r in robots:
            r.set_joint_position_target(r.data.default_joint_pos.torch)
    else:
        for drv in drivers:
            tgt = drv.compute_targets(target_b, is_reach=1.0)
            if leg_ids is not None:  # under the gantry the legs carry no balance load
                tgt[:, leg_ids] = drv.robot.data.default_joint_pos.torch[:, leg_ids]
            drv.robot.set_joint_position_target(tgt)
    for r in robots:
        r.write_data_to_sim()


def slide_cover(cover, cover_center, draw_t, zero_vel):
    """Kinematically slide the cover headward (-x) by the draw fraction."""
    cover_pose = cover.data.root_pose_w.torch.clone()
    cover_pose[0, 0] = cover_center[0] - geo.SHEET_DRAW_DX * draw_t
    cover.write_root_pose_to_sim(cover_pose)
    cover.write_root_velocity_to_sim(zero_vel)


def step_physics(sim, robots, cover, held_pose, zero_vel):
    """Advance DECIMATION sim substeps, pinning the pelvises each substep under the gantry."""
    for _ in range(DECIMATION):
        if held_pose is not None:  # pin EVERY substep (once per control tick drifts)
            for i, r in enumerate(robots):
                r.write_root_pose_to_sim(held_pose[i])
                r.write_root_velocity_to_sim(zero_vel)
        sim.step()
    for r in robots:
        r.update(SIM_DT)
    cover.update(SIM_DT)


def main():
    sim = SimulationContext(sim_cfg)
    robots, cover, cover_center = build_scene()
    sim.reset()
    color_newton_shapes()

    zero_vel = torch.zeros(1, 6, device=DEVICE)
    drivers = make_drivers(robots)
    held_pose, leg_ids = make_gantry(robots)
    coord = make_coordinator()
    cap = make_recorder()

    n_ticks = int(args_cli.seconds / CONTROL_DT)
    capture_every = max(1, round((1.0 / args_cli.fps) / CONTROL_DT))
    frames = []

    for tick in range(n_ticks):
        target, draw_t, beat = reach_schedule(tick, n_ticks)
        if beat:
            run_mhs_beat(coord, beat)
        target_b = torch.tensor([target], dtype=torch.float32, device=DEVICE)

        drive_robots(drivers, robots, target_b, leg_ids)
        slide_cover(cover, cover_center, draw_t, zero_vel)
        step_physics(sim, robots, cover, held_pose, zero_vel)

        if tick % capture_every == 0:
            frames.append(cap.render_rgb_array())

    _encode(frames)
    if coord is not None:
        try:
            print(f"[demo] wrote MHS trace {coord.trace.write_json(args_cli.trace_out)}", flush=True)
            coord.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[demo] MHS trace/stop failed: {exc}", flush=True)
    print("[demo] DONE", flush=True)


def _encode(frames):
    try:
        import imageio.v2 as imageio
        imageio.mimsave(args_cli.out, frames, fps=args_cli.fps)
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] imageio mimsave failed ({exc}); writing frames as PNGs", flush=True)
        import os

        from PIL import Image
        d = os.path.splitext(args_cli.out)[0] + "_frames"
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(np.asarray(f)).save(f"{d}/frame_{i:04d}.png")
    print(f"[demo] wrote {args_cli.out} ({len(frames)} frames @ {args_cli.fps} fps)", flush=True)


main()
