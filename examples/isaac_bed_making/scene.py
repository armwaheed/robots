"""Scene geometry + builders for the two-G1 Isaac Sim bed-making demo.

Layout (world frame, metres). The bed's long axis is **x**: the **head** is at
-x (headboard + pillows) and the **foot** is at +x.

* Bed (mattress): 2.0 (x) x 1.8 (y) x 0.66 (z) box, top at z=0.66.
* Headboard: a board standing at the head (-x), rising above the mattress.
* Two pillows: resting at the head, in place (the robots never touch them).
* Bedsheet: a particle-cloth sheet sized to **overhang each long side and the
  foot by ~9 inches (0.23 m)** and fold back from the headboard at the head —
  the made-bed target from the Figure Helix clip and the Unitree dataset. It
  **starts folded over-and-back on itself at the foot** (an accordion fold); the
  robots then arrange it.
* Robot 0 starts ~0.5 m off the -y long side and **walks in** under a learned
  policy; robot 1 mirrors on the +y side. (They used to be planted in reach;
  now they step in — issue #2 item #4.)

The G1 USD spawns rotated +90 deg about Z (facing +y); robot 1 gets -90 deg.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

# ── Bed (mattress) ──────────────────────────────────────────────────────────
BED_SIZE = (2.0, 1.8, 0.66)
BED_CENTER = (0.0, 0.0, 0.33)
BED_TOP_Z = BED_CENTER[2] + BED_SIZE[2] / 2.0  # 0.66
HEAD_X = -BED_SIZE[0] / 2.0  # -1.0
FOOT_X = +BED_SIZE[0] / 2.0  # +1.0
SIDE_Y = BED_SIZE[1] / 2.0   # 0.9

OVERHANG = 0.2286  # 9 inches: how far the made sheet hangs past each side + foot

# ── Headboard + pillows (static props at the head) ──────────────────────────
HEADBOARD_SIZE = (0.12, 1.9, 0.95)
HEADBOARD_CENTER = (HEAD_X - 0.06, 0.0, 0.55)  # just behind the head, rises to ~1.0
PILLOW_SIZE = (0.5, 0.72, 0.16)
PILLOWS = {
    "left": (HEAD_X + 0.42, -0.43, BED_TOP_Z + PILLOW_SIZE[2] / 2.0),
    "right": (HEAD_X + 0.42, 0.43, BED_TOP_Z + PILLOW_SIZE[2] / 2.0),
}

# ── Robots: walk in from ~0.5 m off their long side, work the foot half ──────
# (pos, yaw_deg). yaw about Z: +90 faces +y, -90 faces -y. They flank the
# foot-half of the bed (x>0) where the folded sheet starts.
ROBOT_Z = 0.75
STAND_PELVIS_Z = 0.80  # clean upright standing pelvis height for manipulation (feet on floor)
MANIP_X = 0.20      # flank the bed; hands sweep over the draped sheet
APPROACH_Y = 1.85   # spawn here (~0.5 m off the y=+/-0.9 side, plus body width)
MANIP_Y = 1.00      # walk-in target (they converge ~0.1 m short, ~1.12, just off the bed side)
ROBOTS = {
    0: {"approach": (MANIP_X, -APPROACH_Y, ROBOT_Z), "manip": (MANIP_X, -MANIP_Y),
        "yaw_deg": 90.0, "side": "-y"},
    1: {"approach": (MANIP_X, APPROACH_Y, ROBOT_Z), "manip": (MANIP_X, MANIP_Y),
        "yaw_deg": -90.0, "side": "+y"},
}
# Bent-knee standing default the agile locomotion policy expects (its output is an
# offset from this pose). Unmatched joints (arms/waist/fingers) default to 0.
WALK_DEFAULT_JOINTS = {
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": -0.20,
}

# Bed corners in the world (compass labels match the swarm driver). +x=foot/East.
BED_CORNERS: Dict[str, Tuple[float, float, float]] = {
    "NE": (FOOT_X, SIDE_Y, BED_TOP_Z),
    "NW": (HEAD_X, SIDE_Y, BED_TOP_Z),
    "SE": (FOOT_X, -SIDE_Y, BED_TOP_Z),
    "SW": (HEAD_X, -SIDE_Y, BED_TOP_Z),
}

# ── Bedsheet ────────────────────────────────────────────────────────────────
# Sized for the made-bed target: full width + 9 in overhang on both long sides
# and the foot. Drapes onto the bed from the foot half, pulled back from the
# headboard so the pillows stay exposed (head folded back). The robots smooth it;
# the foot end is folded back over itself (fold=True) so it starts unmade.
SHEET_SIZE = (1.7, BED_SIZE[1] + 2 * OVERHANG)  # (x length, y width ~2.26)
SHEET_RES = (32, 30)
SHEET_THICKNESS = 0.05
SHEET_FOLD_START = 0.66             # fraction of the length before the foot folds back
SHEET_ORIGIN = (0.40, 0.0, BED_TOP_Z + 0.22)  # drapes onto the foot half of the bed

# Camera: 3/4 view framing the whole bed (head + foot), both robots.
CAM_EYE = (3.9, -3.5, 2.7)
CAM_TARGET = (0.0, 0.0, 0.5)
CAM_RES = (854, 480)


def yaw_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) for a rotation of yaw_deg about +Z."""
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def build_scene_cfg(g1_cfg):
    """Return an InteractiveSceneCfg subclass: two floating (walk-capable) G1s +
    bed + headboard + pillows. The G1s spawn at their approach positions with
    gravity on and a free base so the locomotion policy can walk them in."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    def walker(idx: int):
        g = g1_cfg.copy()
        # Free base + gravity on so the policy can move the root (the Inspire USD
        # ships fixed + gravity-off for manipulation; demo.py also deactivates its
        # baked root_joint before reset — see locomotion.make_floating_base).
        g.spawn.rigid_props.disable_gravity = False
        g.spawn.articulation_props.fix_root_link = False
        g.init_state = g.init_state.replace(
            pos=ROBOTS[idx]["approach"],
            rot=yaw_to_quat(ROBOTS[idx]["yaw_deg"]),
            joint_pos=dict(WALK_DEFAULT_JOINTS),
        )
        return g.replace(prim_path=f"/World/envs/env_0/Robot_{idx}")

    def static_box(size, center, color, friction=1.0):
        return AssetBaseCfg(
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=friction, dynamic_friction=friction, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=center),
        )

    @configclass
    class BedMakingSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome = AssetBaseCfg(prim_path="/World/light",
                            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)))
        key = AssetBaseCfg(prim_path="/World/key",
                           spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=2.0))
        # Mattress — high friction so the sheet grips and does not slide off.
        bed = static_box(BED_SIZE, BED_CENTER, (0.42, 0.30, 0.22), friction=1.5).replace(
            prim_path="/World/Bed")
        headboard = static_box(HEADBOARD_SIZE, HEADBOARD_CENTER, (0.35, 0.24, 0.17)).replace(
            prim_path="/World/Headboard")
        pillow_l = static_box(PILLOW_SIZE, PILLOWS["left"], (0.93, 0.93, 0.96)).replace(
            prim_path="/World/PillowL")
        pillow_r = static_box(PILLOW_SIZE, PILLOWS["right"], (0.93, 0.93, 0.96)).replace(
            prim_path="/World/PillowR")
        robot_0 = walker(0)
        robot_1 = walker(1)

    return BedMakingSceneCfg
