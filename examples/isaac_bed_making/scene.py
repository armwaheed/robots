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

# Bed
BED_SIZE = (2.0, 1.8, 0.5)
BED_CENTER = (0.0, 0.0, 0.25)
BED_TOP_Z = BED_CENTER[2] + BED_SIZE[2] / 2.0  # 0.5

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
GRAB_POINTS = {
    0: (0.10, -0.82, BED_TOP_Z + 0.07),
    1: (-0.10, 0.82, BED_TOP_Z + 0.07),
}
# Where each robot tugs its grabbed edge to square the cover (slightly outward
# + down to tuck), then a lift to smooth.
SMOOTH_POINTS = {
    0: (0.10, -0.95, BED_TOP_Z + 0.02),
    1: (-0.10, 0.95, BED_TOP_Z + 0.02),
}

# Sheet starts draped just above the bed so it falls and covers it.
SHEET_SIZE = (2.2, 2.0)
SHEET_RES = (44, 40)
SHEET_ORIGIN = (0.0, 0.0, BED_TOP_Z + 0.45)

# Camera: 3/4 view showing both robots and the bed.
CAM_EYE = (3.4, -3.0, 2.4)
CAM_TARGET = (0.0, 0.0, 0.55)
CAM_RES = (854, 480)


def yaw_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) for a rotation of yaw_deg about +Z."""
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def lookat_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """World orientation quaternion (w,x,y,z) for a USD camera at ``eye`` looking
    at ``target`` (camera looks down its local -Z, +Y up)."""
    import numpy as np

    eye = np.array(eye, float); target = np.array(target, float); up = np.array(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    w = max(w, 1e-6)
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.30, 0.22)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=BED_CENTER),
        )
        robot_0 = g1_0.replace(prim_path="/World/envs/env_0/Robot_0")
        robot_1 = g1_1.replace(prim_path="/World/envs/env_0/Robot_1")

    return BedMakingSceneCfg
