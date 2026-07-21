"""Two-G1 bed-making scene for the pure-mjlab / MuJoCo(-Warp) demo.

Builds ONE MuJoCo model containing TWO Unitree G1s flanking a correctly-proportioned wide low bed
(mattress + headboard + two propped pillows + a REAL deformable bedsheet). Both robots are the
mjlab-configured G1 (identical actuators / armature / collision / IMU sensors as the trained env),
attached into a shared world with ``MjSpec.attach`` so each keeps its own floating base and is
addressable by prefix.

The sheet is a native MuJoCo ``flexcomp`` cloth (see cloth_sheet.py) — it drapes, folds and wrinkles.
Each robot's drawing hand holds it through a ``connect`` equality that is switched on at the grip and
off at the release, wired at runtime to whichever cloth vertex is under the palm. That mechanism is
forced by the robot, not chosen for convenience: the mjlab G1 has no fingers (its hand is one rigid
capsule), and a measured friction sweep showed a fingerless palm drags flex cloth by <=3% of its own
travel even at mu=10 — see README "Does the cloth hold a grasp?".

Nothing here is Isaac / Newton / NVIDIA-proprietary: it is ``mujoco`` + the mjlab G1 asset. The bed
geometry is copied verbatim from ``examples/isaac_bed_making/scene.py`` (the authoritative target).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from cloth_sheet import (CLOTH_BIT, ROBOT_BIT, WORLD_BIT, flexcomp_xml, grid_elements,
                         sheet_points)

# ── Bed geometry (world metres) — copied from examples/isaac_bed_making/scene.py ──────────────────
# Long axis is x: head at -x (headboard + pillows), foot at +x.
BED_SIZE = (2.0, 1.8, 0.61)
BED_CENTER = (0.0, 0.0, 0.305)
BED_TOP_Z = BED_CENTER[2] + BED_SIZE[2] / 2.0          # 0.61
HEAD_X = -BED_SIZE[0] / 2.0                            # -1.0
FOOT_X = +BED_SIZE[0] / 2.0                            # +1.0
SIDE_Y = BED_SIZE[1] / 2.0                             # 0.9
OVERHANG = 0.2286                                      # 9 inches side/foot overhang

HEADBOARD_SIZE = (0.12, 1.9, 0.95)
HEADBOARD_CENTER = (HEAD_X, 0.0, 0.55)

PILLOW_SIZE = (0.5, 0.72, 0.16)
PILLOW_PROP_DEG = 70.0
PILLOWS = {
    "left": (HEAD_X + 0.22, -0.43, BED_TOP_Z + 0.22),   # (-0.78, -0.43, 0.83)
    "right": (HEAD_X + 0.22, 0.43, BED_TOP_Z + 0.22),   # (-0.78, +0.43, 0.83)
}

# ── The sheet: a deformable flex cloth, lying unmade — shoved footward, hanging off the foot ──────
# ~1.35 m of material along x: ~1.05 m on the mattress plus a 0.30 m fall off the foot end. The robots
# draw it headward toward the pillows. Two things this layout is NOT, both because they were tried and
# measured: the slack is not stored as deep folds (a sheet rumpled hard enough to hold 0.5 m of slack
# in 0.6 m of bed is a rolled tube, and a rolled tube on a slick mattress rolls off the foot), and the
# foot fall is not the isaac demo's full 0.5 m (that much free-hanging cloth anchors the sheet — the
# robots then draw it 0 cm — and destabilises the solver during the drape).
SHEET_HALF_W = BED_SIZE[1] / 2.0 + OVERHANG            # 1.129 — full width + 9-in overhang per side
SHEET_HEAD_X = -0.05                                   # world x of the head (grab) edge at start
SHEET_FLAT_TO_X = 0.35                                 # flat run the robots grab, then a light rumple
SHEET_BUNCH_TO_X = 1.00                                # rumpled run out to the foot edge
SHEET_FOOT_DROP = 0.30                                 # the rest of the sheet hangs off the foot
# Hang the foot lip this far outboard of the foot edge. 0 = against the bed's collision skirt, which
# is what a real sheet does and what draws best; clearing it (0.025) halves the draw, because the free
# lip has to be lifted over the foot corner on every stroke. Measured, not assumed.
SHEET_LIP_CLEAR = 0.0
SHEET_RIPPLE_AMP = 0.025                               # a shallow rumple: unmade, not rolled up
SHEET_RIPPLE_LEN = 0.25
SHEET_SPACING = 0.085                                  # cloth resolution (grid pitch)
SHEET_Z = BED_TOP_Z + 0.008                            # rests on the mattress top
SHEET_MASS = 0.5                                       # a queen top sheet
SHEET_RADIUS = 0.006
SHEET_YOUNG = 5.0e4                                    # capped by the 5 ms step: 2e5 blows up
SHEET_POISSON = 0.3
SHEET_THICKNESS = 8e-4
SHEET_FRICTION = 0.20                                  # sheet on a mattress: it has to be draggable
SHEET_EDGE_DAMPING = 0.004                             # 0.06 is enough to damp the draw away entirely

# Physics rate: the 5 ms step / 50 Hz control the policy trained at. Do not "improve" this — a 2 ms
# step lets the cloth be stiffer but the policy comes apart under it (measured: both robots off their
# marks within one rollout), and matching the training step matters more than a stiffer sheet.
TIMESTEP = 0.005
DECIMATION = 4

# Each hand closes on a HANDFUL of cloth, not a point: the nearest vertex plus its neighbours, each
# held at its own offset so the patch keeps its shape. One point stretches the sheet locally and the
# bulk never moves (measured: 6.6 cm of vertex travel, 2 cm of sheet).
GRIP_POINTS = 8
GRIP_PATCH = 0.30                                      # radius of the handful (m)
SHEET_RGBA = (0.46, 0.60, 0.82, 1.0)

# ── Robots — flank opposite long sides, face the bed, draw the head edge headward ─────────────────
MANIP_X = 0.40
MANIP_Y = 1.10                                         # bedside stance, clear of the y=+/-0.9 edge
PELVIS_Z = 0.76                                        # mjlab G1 KNEES_BENT keyframe height
ROBOTS = {
    0: {"pos": (MANIP_X, -MANIP_Y, PELVIS_Z), "yaw_deg": 90.0, "hand": "left", "mirror": False},
    1: {"pos": (MANIP_X, +MANIP_Y, PELVIS_Z), "yaw_deg": -90.0, "hand": "right", "mirror": True},
}

# ── Hero camera: a 3/4 view from the head/+y side that frames the whole bed, both propped pillows,
# the headboard, the drawn sheet, and both robots (the "made bed" read). ──────────────────────────
CAM_AZIMUTH = 150.0
CAM_ELEVATION = -19.0
CAM_DISTANCE = 5.1
CAM_LOOKAT = (-0.25, 0.0, 0.60)

# ── The 29 actuated joints, in the exact order the trained policy's obs/action use ────────────────
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
NUM_JOINTS = len(JOINT_NAMES)

# Palm capsule radius (left_hand_collision / right_hand_collision in the mjlab G1 asset): how far
# below the palm site the hand's underside — where the grip holds the cloth — actually is.
HAND_CAPSULE_RADIUS = 0.035


def _yaw_quat(deg: float) -> list[float]:
    h = math.radians(deg) / 2.0
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _mirror_partner(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def _mirror_sign(name: str) -> float:
    # Sagittal (xz-plane) reflection negates roll/yaw joints; pitch/knee/elbow are symmetric.
    return -1.0 if ("_roll" in name or "_yaw" in name) else 1.0


def build_mirror_tables() -> tuple[np.ndarray, np.ndarray]:
    """Return (perm, sign) with mirrored[i] = sign[i] * source[perm[i]] over the 29 joints."""
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    perm = np.array([idx[_mirror_partner(n)] for n in JOINT_NAMES], dtype=np.int64)
    sign = np.array([_mirror_sign(n) for n in JOINT_NAMES], dtype=np.float64)
    return perm, sign


@dataclass
class RobotHandles:
    prefix: str
    mirror: bool
    hand: str                            # "left" or "right" — the drawing hand
    hand_site: int                       # left_palm or right_palm site id (the drawing hand)
    left_palm_site: int
    right_palm_site: int
    hand_body: int                       # wrist_yaw_link of the drawing hand (grip constraint target)
    pelvis_body: int
    qpos_adr: np.ndarray                  # (29,) qpos address per JOINT_NAMES joint
    dof_adr: np.ndarray                   # (29,) qvel/dof address per joint
    act_id: np.ndarray                    # (29,) actuator id per joint
    free_qadr: int                        # freejoint qpos base address (7 vals: xyz + wxyz)
    free_dofadr: int                      # freejoint dof base address (6)
    imu_linvel_adr: int
    imu_angvel_adr: int
    imu_up_adr: int
    eq_ids: np.ndarray = None             # the connect equalities that hold the cloth
    default_qpos: np.ndarray = field(default=None)  # (29,) default joint pose (obs baseline)
    pos: tuple = (0, 0, 0)
    yaw_deg: float = 0.0


@dataclass
class SceneHandles:
    model: mujoco.MjModel
    robots: list[RobotHandles]
    flex_id: int
    vert_adr: int
    vert_num: int
    vert_bodies: np.ndarray               # (nvert,) body id per cloth vertex
    default_qpos_by_joint: np.ndarray
    cam_azimuth: float
    cam_elevation: float
    cam_distance: float
    cam_lookat: tuple

    def cloth(self, data) -> np.ndarray:
        """(nvert, 3) world positions of the cloth vertices."""
        return data.flexvert_xpos[self.vert_adr:self.vert_adr + self.vert_num]

    def coverage(self, data, reach=0.09) -> float:
        """Fraction of the SHEETABLE mattress top that has sheet lying on it — "how made is the bed".

        Samples the mattress top on a 4 cm grid and asks whether any cloth vertex resting at bed
        height is within one grid pitch of it. Vertex counts do not answer this: the sheet can pile
        up in one place and leave the foot bare. The pillow zone (the head 0.5 m, where the two
        propped pillows sit) is excluded — a made bed does not have the top sheet over the pillows.
        """
        cloth = self.cloth(data)
        on = cloth[cloth[:, 2] > BED_TOP_Z - 0.06][:, :2]
        if len(on) == 0:
            return 0.0
        gx, gy = np.meshgrid(np.arange(HEAD_X + 0.5, 1.0, 0.04), np.arange(-0.88, 0.9, 0.04),
                             indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        near = np.linalg.norm(pts[:, None, :] - on[None, :, :], axis=2).min(axis=1)
        return float((near < reach).mean())


def _mattress_pads() -> str:
    """Tile the mattress top with thin cloth-only pads.

    MuJoCo generates at most ``mjMAXCONPAIR`` = 50 contacts per colliding BODY pair, and a flex is
    one side of that pair — so a 500-vertex sheet lying on a mattress body is supported at 50 points
    no matter how the mattress is tiled: it sags between them and sinks straight through (measured:
    364 of 504 vertices above the mattress top at t=0, 29 half a second later). Each pad therefore
    gets its OWN static body, which buys 50 contacts apiece. The pads are flush with the mattress
    top, share its colour, and are invisible to the robots (CLOTH_BIT only).
    """
    nx, ny, ht = 6, 6, 0.01
    hx, hy = BED_SIZE[0] / (2 * nx), BED_SIZE[1] / (2 * ny)
    skirt_h = 0.30                                       # deep enough for the side/foot drape
    out = []

    def pad(name, pos, size):
        out.append(
            f'    <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">\n'
            f'      <geom name="{name}_geom" type="box" '
            f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" rgba="0.74 0.66 0.55 1" '
            f'friction="{SHEET_FRICTION} 0.02 0.001" '
            f'contype="{CLOTH_BIT}" conaffinity="{CLOTH_BIT}"/>\n'
            f'    </body>')

    for i in range(nx):
        px = BED_CENTER[0] - BED_SIZE[0] / 2 + hx * (2 * i + 1)
        for j in range(ny):                              # mattress top
            py = BED_CENTER[1] - BED_SIZE[1] / 2 + hy * (2 * j + 1)
            pad(f"pad_{i}_{j}", (px, py, BED_TOP_Z - ht), (hx, hy, ht))
        for sgn in (-1, 1):                              # long sides, so the overhang drapes on them
            pad(f"skirt_y{'m' if sgn < 0 else 'p'}_{i}",
                (px, BED_CENTER[1] + sgn * (BED_SIZE[1] / 2 + ht), BED_TOP_Z - skirt_h),
                (hx, ht, skirt_h))
    for j in range(ny):                                  # foot face, for the sheet hanging off it
        py = BED_CENTER[1] - BED_SIZE[1] / 2 + hy * (2 * j + 1)
        pad(f"skirt_foot_{j}", (FOOT_X + ht, py, BED_TOP_Z - skirt_h), (ht, hy, skirt_h))
    return "\n".join(out)


def _pillow_quat() -> list[float]:
    h = math.radians(PILLOW_PROP_DEG) / 2.0
    return [math.cos(h), 0.0, math.sin(h), 0.0]          # tip about +y (prop against headboard)


def _world_xml() -> str:
    """The static world: floor, lights, bed, headboard, pillows, and the deformable sheet.

    Emitted as XML because ``flexcomp`` is an XML-level model macro (it expands into the cloth's
    vertex bodies + flex at parse time); the two G1s are attached onto this spec afterwards.
    """
    pts, nu, nv = sheet_points(
        head_x=SHEET_HEAD_X, flat_to_x=SHEET_FLAT_TO_X, bunch_to_x=SHEET_BUNCH_TO_X,
        top_z=SHEET_Z, half_width=SHEET_HALF_W, spacing=SHEET_SPACING,
        ripple_amp=SHEET_RIPPLE_AMP, ripple_len=SHEET_RIPPLE_LEN, foot_drop=SHEET_FOOT_DROP,
        lip_clear=SHEET_LIP_CLEAR,
    )
    cloth = flexcomp_xml(
        "sheet", pts, grid_elements(nu, nv), mass=SHEET_MASS, radius=SHEET_RADIUS,
        young=SHEET_YOUNG, poisson=SHEET_POISSON, thickness=SHEET_THICKNESS,
        friction=SHEET_FRICTION, edge_damping=SHEET_EDGE_DAMPING, rgba=SHEET_RGBA,
    )
    pw = " ".join(f"{v}" for v in _pillow_quat())
    scenery = f'contype="{WORLD_BIT}" conaffinity="{WORLD_BIT}"'
    return f"""
<mujoco model="mjlab_bed_making">
  <compiler autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse="0.28 0.28 0.30" ambient="0.22 0.22 0.24" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.30 0.34 0.42"
             rgb2="0.09 0.10 0.13" width="256" height="256"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.26 0.28 0.32"
             rgb2="0.20 0.22 0.26" width="512" height="512"/>
    <material name="floormat" texture="floortex" texrepeat="6 6" reflectance="0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="floormat" {scenery}/>
    <light pos="4.5 -4.5 6.5" dir="-4.5 4.5 -6.5" diffuse="0.36 0.36 0.36" specular="0 0 0"/>
    <light pos="-3.5 4.0 6.5" dir="3.5 -4.0 -6.5" diffuse="0.24 0.24 0.24" specular="0 0 0"/>

    <body name="bed" pos="{BED_CENTER[0]} {BED_CENTER[1]} {BED_CENTER[2]}">
      <geom name="mattress" type="box"
            size="{BED_SIZE[0]/2} {BED_SIZE[1]/2} {BED_SIZE[2]/2}"
            rgba="0.74 0.66 0.55 1" friction="{SHEET_FRICTION} 0.02 0.001"
            contype="{ROBOT_BIT}" conaffinity="{ROBOT_BIT}"/>
    </body>
{_mattress_pads()}
    <body name="headboard" pos="{HEADBOARD_CENTER[0]} {HEADBOARD_CENTER[1]} {HEADBOARD_CENTER[2]}">
      <geom name="headboard_geom" type="box"
            size="{HEADBOARD_SIZE[0]/2} {HEADBOARD_SIZE[1]/2} {HEADBOARD_SIZE[2]/2}"
            rgba="0.38 0.25 0.14 1" {scenery}/>
    </body>
    <body name="pillow_left" pos="{PILLOWS['left'][0]} {PILLOWS['left'][1]} {PILLOWS['left'][2]}"
          quat="{pw}">
      <geom name="pillow_left_geom" type="box"
            size="{PILLOW_SIZE[0]/2} {PILLOW_SIZE[1]/2} {PILLOW_SIZE[2]/2}"
            rgba="0.97 0.95 0.90 1" contype="{CLOTH_BIT}" conaffinity="{CLOTH_BIT}"/>
    </body>
    <body name="pillow_right" pos="{PILLOWS['right'][0]} {PILLOWS['right'][1]} {PILLOWS['right'][2]}"
          quat="{pw}">
      <geom name="pillow_right_geom" type="box"
            size="{PILLOW_SIZE[0]/2} {PILLOW_SIZE[1]/2} {PILLOW_SIZE[2]/2}"
            rgba="0.97 0.95 0.90 1" contype="{CLOTH_BIT}" conaffinity="{CLOTH_BIT}"/>
    </body>
{cloth}
  </worldbody>
</mujoco>
"""


def build_scene() -> SceneHandles:
    """Compile the two-robot bed scene and return handles for the rollout."""
    from mjlab.entity.entity import Entity
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg

    world = mujoco.MjSpec.from_string(_world_xml())

    # Attach the two G1s at identity frames so each freejoint qpos == world pose.
    for i in (0, 1):
        ent = Entity(get_g1_robot_cfg())
        world.attach(ent.spec, prefix=f"r{i}/", frame=world.worldbody.add_frame())

    # GRIP_POINTS connect equalities per robot: the handful of cloth under the palm <-> the drawing
    # hand. Wired to actual vertices and anchors at grip time (see two_robot_bed.grip), inactive here.
    for i in (0, 1):
        for k in range(GRIP_POINTS):
            eq = world.add_equality()
            eq.name = f"grip{i}_{k}"
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = "sheet_0"
            eq.name2 = f"r{i}/{ROBOTS[i]['hand']}_wrist_yaw_link"
            eq.data[:6] = [0.0] * 6
            eq.active = False
            eq.solref = [0.03, 1.0]                      # soft enough that the grab is not a snap

    model = world.compile()

    # The hand capsules are the only robot geoms allowed to touch the cloth (see cloth_sheet.py).
    for i in (0, 1):
        gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"r{i}/{ROBOTS[i]['hand']}_hand_collision")
        model.geom_contype[gid] = WORLD_BIT
        model.geom_conaffinity[gid] = WORLD_BIT

    default_by_joint = np.array(_default_joint_pose(), dtype=np.float64)
    robots: list[RobotHandles] = []
    for i in (0, 1):
        pfx = f"r{i}/"
        cfg = ROBOTS[i]
        qadr = np.array([model.jnt_qposadr[_jid(model, pfx + n)] for n in JOINT_NAMES])
        dadr = np.array([model.jnt_dofadr[_jid(model, pfx + n)] for n in JOINT_NAMES])
        aid = np.array([_aid(model, pfx + n) for n in JOINT_NAMES])
        fj = _jid(model, pfx + "floating_base_joint")
        robots.append(RobotHandles(
            prefix=pfx, mirror=cfg["mirror"], hand=cfg["hand"],
            hand_site=_sid(model, pfx + f"{cfg['hand']}_palm"),
            left_palm_site=_sid(model, pfx + "left_palm"),
            right_palm_site=_sid(model, pfx + "right_palm"),
            hand_body=_bid(model, pfx + f"{cfg['hand']}_wrist_yaw_link"),
            pelvis_body=_bid(model, pfx + "pelvis"),
            qpos_adr=qadr, dof_adr=dadr, act_id=aid,
            free_qadr=model.jnt_qposadr[fj], free_dofadr=model.jnt_dofadr[fj],
            imu_linvel_adr=_sensor_adr(model, pfx + "imu_lin_vel"),
            imu_angvel_adr=_sensor_adr(model, pfx + "imu_ang_vel"),
            imu_up_adr=_sensor_adr(model, pfx + "imu_upvector"),
            eq_ids=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip{i}_{k}")
                             for k in range(GRIP_POINTS)]),
            default_qpos=default_by_joint.copy(), pos=cfg["pos"], yaw_deg=cfg["yaw_deg"],
        ))

    vadr = int(model.flex_vertadr[0])
    vnum = int(model.flex_vertnum[0])
    return SceneHandles(
        model=model, robots=robots,
        flex_id=0, vert_adr=vadr, vert_num=vnum,
        vert_bodies=np.array(model.flex_vertbodyid[vadr:vadr + vnum]),
        default_qpos_by_joint=default_by_joint,
        cam_azimuth=CAM_AZIMUTH, cam_elevation=CAM_ELEVATION,
        cam_distance=CAM_DISTANCE, cam_lookat=CAM_LOOKAT,
    )


def set_init_pose(model, data, scene: SceneHandles) -> None:
    """Place each robot at its bedside stance (freejoint xyz+yaw) with the default joint pose.

    qpos is zeroed first, which also returns every cloth vertex to its authored (rumpled) rest
    position — the flex vertices are 3 slide joints each, measured from where cloth_sheet.py put them.
    """
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for r in scene.robots:
        q = r.free_qadr
        data.qpos[q:q + 3] = r.pos
        data.qpos[q + 3:q + 7] = _yaw_quat(r.yaw_deg)
        for j in range(NUM_JOINTS):
            data.qpos[r.qpos_adr[j]] = r.default_qpos[j]
    data.eq_active[:] = 0
    mujoco.mj_forward(model, data)


def _default_joint_pose() -> list[float]:
    """The mjlab G1 KNEES_BENT default pose, in JOINT_NAMES order (obs baseline)."""
    d = {
        "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669, "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
        "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
        "left_elbow_joint": 0.6, "right_elbow_joint": 0.6,
    }
    return [d.get(n, 0.0) for n in JOINT_NAMES]


def _jid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
def _aid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
def _sid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
def _bid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def _sensor_adr(m, n):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
    return int(m.sensor_adr[sid])


if __name__ == "__main__":
    sc = build_scene()
    m = sc.model
    print(f"compiled: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} nsensor={m.nsensor} "
          f"neq={m.neq} nflexvert={m.nflexvert} nflexelem={m.nflexelem}")
    d = mujoco.MjData(m)
    set_init_pose(m, d, sc)
    for r in sc.robots:
        pelvis_z = d.xpos[r.pelvis_body][2]
        lf = _sid(m, r.prefix + "left_foot")
        foot_z = d.site_xpos[lf][2]
        palm = d.site_xpos[r.hand_site]
        print(f"  {r.prefix} pelvis_z={pelvis_z:.3f} left_foot_z={foot_z:.3f} "
              f"{r.hand}_palm={palm.round(3).tolist()} eqs={r.eq_ids.tolist()}")
    v = sc.cloth(d)
    print(f"cloth: {sc.vert_num} verts, x {v[:,0].min():.2f}..{v[:,0].max():.2f} "
          f"y {v[:,1].min():.2f}..{v[:,1].max():.2f} z {v[:,2].min():.2f}..{v[:,2].max():.2f}")
    print(f"cam az={sc.cam_azimuth:.1f} el={sc.cam_elevation:.1f} dist={sc.cam_distance:.2f} "
          f"look={sc.cam_lookat}")
    perm, sign = build_mirror_tables()
    print("mirror perm", perm.tolist())
    print("mirror sign", sign.tolist())
