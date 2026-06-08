"""Scene geometry + builders for the two-G1 Isaac Sim bed-making demo.

Layout (world frame, metres). The bed's long axis is **x**: the **head** is at
-x (headboard + pillows) and the **foot** is at +x.

* Bed (mattress): 2.0 (x) x 1.8 (y) x 0.66 (z) box, top at z=0.66.
* Headboard: a board standing at the head (-x), rising above the mattress.
* Two pillows: resting at the head, in place (the robots never touch them).
* Bedsheet: a particle-cloth sheet that **starts flat, lying on the foot half of
  the bed and draping off the foot** — as if it had been pulled down toward the
  foot. The head half of the mattress is bare. The robots make the bed by grabbing
  the two **forward (head-side) corners** and pulling them toward the head, drawing
  the sheet up over the mattress. (This replaced an accordion fold-back start, which
  was too hard for the simulated G1s to unfold — issue #2.)
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

# Collision rounding for the cloth-facing props (bed + pillows). A rest offset inflates
# and rounds the collider's sharp corners and leaves a small air gap; the contact offset
# (must be > rest) widens the band where contacts are generated so the cloth catches the
# collider early instead of tunnelling. This is the "round the mattress corners" fix done
# at the collision level — see scene.static_box and .isaac/fix_tune.py.
COLLIDER_REST_OFFSET = 0.015
COLLIDER_CONTACT_OFFSET = 0.035

# ── Headboard + pillows (static props at the head) ──────────────────────────
HEADBOARD_SIZE = (0.12, 1.9, 0.95)
HEADBOARD_CENTER = (HEAD_X - 0.06, 0.0, 0.55)  # just behind the head, rises to ~1.0
PILLOW_SIZE = (0.5, 0.72, 0.16)
PILLOWS = {
    "left": (HEAD_X + 0.42, -0.43, BED_TOP_Z + PILLOW_SIZE[2] / 2.0),
    "right": (HEAD_X + 0.42, 0.43, BED_TOP_Z + PILLOW_SIZE[2] / 2.0),
}

# ── Robots: walk in from ~0.5 m off their long side, flank the bed mid-side ──
# (pos, yaw_deg). yaw about Z: +90 faces +y, -90 faces -y. They stand near the
# middle of each long side so they can grab the sheet's head-side corners (gathered
# just foot-of-centre) and pull them toward the head.
ROBOT_Z = 0.75
STAND_PELVIS_Z = 0.80  # clean upright standing pelvis height for manipulation (feet on floor)
MANIP_X = 0.05      # flank the bed near mid-side, within reach of the head-side corners
APPROACH_Y = 1.85   # spawn here (~0.5 m off the y=+/-0.9 side, plus body width)
MANIP_Y = 1.00      # walk-in target (they converge ~0.1 m short, ~1.12, just off the bed side)
ROBOTS = {
    0: {"approach": (MANIP_X, -APPROACH_Y, ROBOT_Z), "manip": (MANIP_X, -MANIP_Y),
        "yaw_deg": 90.0, "side": "-y"},
    1: {"approach": (MANIP_X, APPROACH_Y, ROBOT_Z), "manip": (MANIP_X, MANIP_Y),
        "yaw_deg": -90.0, "side": "+y"},
}
# Bent-knee standing default the rl_gym walk policy expects (its output is an
# offset from this pose); this matches RLGYM_DEFAULT_ANGLES. Unmatched joints
# (arms/waist/fingers) default to 0.
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
# Full bed width + ~9 in overhang on both long sides. It starts FLAT and gathered
# toward the foot: the head-side edge sits just foot-of-centre (~x=+0.1), the rest
# lies over the foot half and drapes off the foot of the bed — the head half of the
# mattress is bare. The robots grab the two head-side corners and pull the cover up
# toward the head. (No fold-back: that accordion start was too hard to unfold.)
#
# It is a fairly FINE, thin, drapey sheet so the side overhang actually folds down
# over the mattress edges and the cloth stays calm (a thick, coarse "duvet" grips a
# touch better but turns rigid — the overhang juts out stiff — and flutters).
#
# The foot is ACCORDION-pleated (demo passes accordion=True): the head-side strip the
# robots grip lies flat, and the rest is gathered into a ruffle of slack over the foot
# half. Pulling the flat head edge toward the head UNSPOOLS that slack rather than
# dragging a sheet stuck flat to the mattress — so the friction grip suffices and the
# robots aren't yanked over. See cloth.build_bedsheet's accordion_* args.
SHEET_SIZE = (1.3, BED_SIZE[1] + 2 * OVERHANG)  # (x length, y width ~2.26)
# MuJoCo recipe (issue #2, 2026-06-06): COARSE + FEATHERLIGHT + THICK is what made the
# MuJoCo flex sheet look good and hang still — not "triangles". MuJoCo used 143 verts /
# 0.18 kg / ~4.4 cm. Match it: a coarse grid (with cloth.py's light particle_mass this
# totals ~0.2 kg) settles instead of sloshing. NOTE: the *physics* is thick, but the
# RENDER is still a single-layer membrane (looks thin) — giving it visual thickness
# needs a render-side shell (extrude / double-layer the mesh). That is the next task.
SHEET_RES = (14, 12)                            # coarse like MuJoCo's 13×11 → stable drape
SHEET_THICKNESS = 0.05                          # thick physics/collision profile
# Visual thickness: the particle cloth is a single-layer membrane (renders thin), so
# cloth.build_shell_mesh extrudes a closed double-layer SLAB of this depth along the
# surface normal each frame — purely render-side, physics unchanged. 0.07 m reads as a
# substantial folded cover (validated in isolation 2026-06-08, .isaac/shell_tune).
SHEET_SHELL_THICKNESS = 0.07
# Centre placed so the flat head edge sits at ~x=+0.05 (right where the robots stand,
# head half of the bed bare) and the pleated ruffle gathers over the foot half. Rests
# just above the mattress top so it settles onto it (not floating in the air).
SHEET_ORIGIN = (0.70, 0.0, BED_TOP_Z + 0.04)

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
    from isaaclab.actuators import ImplicitActuatorCfg
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
        # The rl_gym walk policy was trained with specific leg PD gains; set them on
        # the Isaac actuators or the policy's targets are tracked wrong and it falls.
        # Replacing the "legs"/"feet" groups (hips+knee / ankles) leaves waist + arms
        # + hands untouched. (kp/kd from policies/g1_rlgym_walk.yaml.)
        g.actuators = dict(g.actuators)
        g.actuators["legs"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint",
                              ".*_hip_yaw_joint", ".*_knee_joint"],
            stiffness={".*_hip_pitch_joint": 100.0, ".*_hip_roll_joint": 100.0,
                       ".*_hip_yaw_joint": 100.0, ".*_knee_joint": 150.0},
            damping={".*_hip_pitch_joint": 2.0, ".*_hip_roll_joint": 2.0,
                     ".*_hip_yaw_joint": 2.0, ".*_knee_joint": 4.0},
            effort_limit_sim=200.0, velocity_limit_sim=100.0)
        g.actuators["feet"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0, damping=2.0, effort_limit_sim=100.0, velocity_limit_sim=100.0)
        g.init_state = g.init_state.replace(
            pos=ROBOTS[idx]["approach"],
            rot=yaw_to_quat(ROBOTS[idx]["yaw_deg"]),
            joint_pos=dict(WALK_DEFAULT_JOINTS),
        )
        return g.replace(prim_path=f"/World/envs/env_0/Robot_{idx}")

    def static_box(size, center, color, friction=1.0, rounded=False):
        # A rest/contact offset INFLATES + ROUNDS the collision corners and leaves a
        # small air gap, so the particle cloth drapes over a rounded edge instead of
        # catching on a sharp 90° corner (which pokes through the sheet — the "mattress
        # edge clips the bedsheet" bug) and so it reliably catches thin colliders like
        # the pillows instead of tunnelling through them. PhysX requires contact > rest.
        # (NVIDIA forums: tune contact/rest offset on both the cloth and the colliders.)
        cp = (sim_utils.CollisionPropertiesCfg(
                  contact_offset=COLLIDER_CONTACT_OFFSET, rest_offset=COLLIDER_REST_OFFSET)
              if rounded else sim_utils.CollisionPropertiesCfg())
        return AssetBaseCfg(
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=cp,
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
        # Mattress — LOW friction so the robots can actually drag the cover across it.
        # A high-friction bed anchors the cover, and the drag force then exceeds the
        # hand's friction grip and tears the cover out of it (the cover "slips"). With a
        # slick mattress the cover slides headward under even a marginal grip. (Trade-off:
        # too slick and the foot-draped cover slides off on its own — 0.4 holds it.)
        bed = static_box(BED_SIZE, BED_CENTER, (0.42, 0.30, 0.22), friction=0.4, rounded=True).replace(
            prim_path="/World/Bed")
        headboard = static_box(HEADBOARD_SIZE, HEADBOARD_CENTER, (0.35, 0.24, 0.17)).replace(
            prim_path="/World/Headboard")
        # Pillows are rounded colliders so the cover drapes OVER their tops (functional
        # pillows) instead of passing through them.
        pillow_l = static_box(PILLOW_SIZE, PILLOWS["left"], (0.93, 0.93, 0.96), rounded=True).replace(
            prim_path="/World/PillowL")
        pillow_r = static_box(PILLOW_SIZE, PILLOWS["right"], (0.93, 0.93, 0.96), rounded=True).replace(
            prim_path="/World/PillowR")
        robot_0 = walker(0)
        robot_1 = walker(1)

    return BedMakingSceneCfg
