"""Scene geometry + builders for the two-G1 Isaac Sim bed-making demo.

Layout (world frame, metres):

* Bed: a 2.0 (x) x 1.8 (y) x 0.5 (z) box centred at the origin, top at z=0.5.
* Robot 0 stands 0.5 m off the -y long side, at (0, -1.4, 0.75), facing +y.
* Robot 1 stands 0.5 m off the +y long side, at (0, +1.4, 0.75), facing -y.
  (The G1 USD spawns rotated +90 deg about Z = facing +y; robot 1 gets -90 deg.)
* Bedsheet: a particle-cloth sheet that drapes over the bed.

A planted (fixed-base) robot reaches the near half of its own side — which is
exactly the equal-peer, work-your-own-side behaviour seen in the Figure video.
Reaching the far corners needs locomotion (the learned-policy phase).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

# Bed. Raised so the top sits at the height the planted G1 palms naturally reach
# (~0.66 m) — measured: with a 0.5 m bed the palms stalled ~17 cm above the sheet
# and the grasp could never bind. A taller bed (top ~0.66) puts the sheet within a
# few cm of the palm so the PhysX attachment catches it.
BED_SIZE = (2.0, 1.8, 0.66)
BED_CENTER = (0.0, 0.0, 0.33)
BED_TOP_Z = BED_CENTER[2] + BED_SIZE[2] / 2.0  # 0.66

# Robots: (pos, yaw_deg). yaw is about Z; +90 faces +y, -90 faces -y.
# Planted close to the bed edge (~0.15 m off the y=+/-0.9 long sides) so the
# grasp is a mostly-downward reach within the arm's well-conditioned envelope.
# (The issue's "half a metre off the side" distance is restored once the learned
# locomotion lets the robots step in; planted, they need to be within reach.)
ROBOT_Z = 0.75
ROBOTS = {
    0: {"pos": (0.0, -1.05, ROBOT_Z), "yaw_deg": 90.0, "side": "-y"},
    1: {"pos": (0.0, 1.05, ROBOT_Z), "yaw_deg": -90.0, "side": "+y"},
}

# Bed corners in the world (compass labels match the swarm driver).
BED_CORNERS: Dict[str, Tuple[float, float, float]] = {
    "NE": (1.0, 0.9, BED_TOP_Z),
    "NW": (-1.0, 0.9, BED_TOP_Z),
    "SE": (1.0, -0.9, BED_TOP_Z),
    "SW": (-1.0, -0.9, BED_TOP_Z),
}

# Per-robot reachable grab point on its near sheet edge (mid-edge, in reach of a
# planted robot). Robot 0 works the -y edge, robot 1 the +y edge.
# Grab target sits ~3 cm into the sheet on the bed top so the IK presses the palm
# down onto the cloth (the IK undershoots a few cm vertically, landing it right at
# the surface).
GRAB_POINTS = {
    0: (0.10, -0.82, BED_TOP_Z - 0.03),
    1: (-0.10, 0.82, BED_TOP_Z - 0.03),
}
# Where each robot tugs its grabbed edge to square the cover: a small outward +
# downward pull that stays inside the planted arm's reach envelope.
SMOOTH_POINTS = {
    0: (0.10, -0.90, BED_TOP_Z - 0.05),
    1: (-0.10, 0.90, BED_TOP_Z - 0.05),
}

# Sheet starts draped just above the bed so it falls and covers it.
# Particles sized to touch at rest (canonical recipe) is what drapes. We use a
# moderately coarse grid so the particles are fat enough to read as a *thick*
# duvet-like sheet (SHEET_THICKNESS) — a thicker collision profile is much easier
# for the hands to catch (as in the Figure/Unitree bed-making clips). The Fabric
# fix (read tensor cloth-view + blit) means resolution no longer affects whether
# it renders, only how smoothly it drapes.
SHEET_SIZE = (2.2, 2.0)
SHEET_RES = (34, 30)
SHEET_THICKNESS = 0.06  # ~6 cm "duvet"; particle radius = thickness/2
SHEET_ORIGIN = (0.0, 0.0, BED_TOP_Z + 0.30)

# Camera: 3/4 view showing both robots and the bed.
CAM_EYE = (3.4, -3.0, 2.4)
CAM_TARGET = (0.0, 0.0, 0.55)
CAM_RES = (854, 480)


def yaw_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) for a rotation of yaw_deg about +Z."""
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def build_scene_cfg(g1_cfg):
    """Return an InteractiveSceneCfg subclass with two planted G1s + bed."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    g1_0 = g1_cfg.copy()
    g1_0.spawn.articulation_props.fix_root_link = True
    g1_0.init_state.pos = ROBOTS[0]["pos"]
    g1_0.init_state.rot = yaw_to_quat(ROBOTS[0]["yaw_deg"])
    g1_1 = g1_cfg.copy()
    g1_1.spawn.articulation_props.fix_root_link = True
    g1_1.init_state.pos = ROBOTS[1]["pos"]
    g1_1.init_state.rot = yaw_to_quat(ROBOTS[1]["yaw_deg"])

    @configclass
    class BedMakingSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome = AssetBaseCfg(prim_path="/World/light",
                            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)))
        key = AssetBaseCfg(prim_path="/World/key",
                           spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=2.0))
        bed = AssetBaseCfg(
            prim_path="/World/Bed",
            spawn=sim_utils.CuboidCfg(
                size=BED_SIZE,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # High friction so the sheet grips the bed top and does not slide
                # off when a corner is tugged (mirrors the MuJoCo bed friction).
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5, dynamic_friction=1.5, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.30, 0.22)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=BED_CENTER),
        )
        robot_0 = g1_0.replace(prim_path="/World/envs/env_0/Robot_0")
        robot_1 = g1_1.replace(prim_path="/World/envs/env_0/Robot_1")

    return BedMakingSceneCfg
