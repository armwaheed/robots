"""Two-G1 bed-making scene for the pure-mjlab / MuJoCo(-Warp) demo.

Builds ONE MuJoCo model containing TWO Unitree G1s flanking a correctly-proportioned wide low bed
(mattress + headboard + two propped pillows + a draping sheet). Both robots are the mjlab-configured
G1 (identical actuators / armature / collision / IMU sensors as the trained env), attached into a
shared world with ``MjSpec.attach`` so each keeps its own floating base and is addressable by prefix.

Nothing here is Isaac / Newton / NVIDIA-proprietary: it is ``mujoco`` + the mjlab G1 asset. The bed
geometry is copied verbatim from ``examples/isaac_bed_making/scene.py`` (the authoritative target).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

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

SHEET_LEN = 1.55                                       # x: head edge -> off the foot
SHEET_W = BED_SIZE[1] + 2 * OVERHANG                   # ~2.257, incl. 9-in side overhang
SHEET_THICKNESS = 0.05
SHEET_HEAD_X = 0.10                                    # world x of the sheet's head (grab) edge at start
SHEET_Z = BED_TOP_Z + SHEET_THICKNESS / 2.0 + 0.005    # rests just on the mattress top

# ── Robots — flank opposite long sides, face the bed, draw the head edge headward ─────────────────
MANIP_X = 0.40
MANIP_Y = 1.10                                         # bedside stance, clear of the y=+/-0.9 edge
PELVIS_Z = 0.76                                        # mjlab G1 KNEES_BENT keyframe height
ROBOTS = {
    0: {"pos": (MANIP_X, -MANIP_Y, PELVIS_Z), "yaw_deg": 90.0, "hand": "left", "mirror": False},
    1: {"pos": (MANIP_X, +MANIP_Y, PELVIS_Z), "yaw_deg": -90.0, "hand": "right", "mirror": True},
}

# ── Hero camera: a 3/4 view from the head/+y side that frames the whole bed, both propped pillows,
# the headboard, the drawn cover, and both robots (the "made bed" read). ──────────────────────────
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
    pelvis_body: int
    qpos_adr: np.ndarray                  # (29,) qpos address per JOINT_NAMES joint
    dof_adr: np.ndarray                   # (29,) qvel/dof address per joint
    act_id: np.ndarray                    # (29,) actuator id per joint
    free_qadr: int                        # freejoint qpos base address (7 vals: xyz + wxyz)
    free_dofadr: int                      # freejoint dof base address (6)
    imu_linvel_adr: int
    imu_angvel_adr: int
    imu_up_adr: int
    default_qpos: np.ndarray = field(default=None)  # (29,) default joint pose (obs baseline)
    pos: tuple = (0, 0, 0)
    yaw_deg: float = 0.0


@dataclass
class SceneHandles:
    model: mujoco.MjModel
    robots: list[RobotHandles]
    sheet_mocap: int
    sheet_body: int
    default_qpos_by_joint: np.ndarray
    cam_azimuth: float
    cam_elevation: float
    cam_distance: float
    cam_lookat: tuple


def _add_box(body, name, half, rgba, collide, group=0):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(half)
    g.rgba = list(rgba)
    g.group = group
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


def _add_bed(world: mujoco.MjSpec) -> None:
    wb = world.worldbody
    bed = wb.add_body(name="bed")
    bed.pos = list(BED_CENTER)
    _add_box(bed, "mattress", (BED_SIZE[0] / 2, BED_SIZE[1] / 2, BED_SIZE[2] / 2),
             (0.74, 0.66, 0.55, 1.0), collide=True)          # warm mattress/box-spring tan

    hb = wb.add_body(name="headboard")
    hb.pos = list(HEADBOARD_CENTER)
    _add_box(hb, "headboard_geom",
             (HEADBOARD_SIZE[0] / 2, HEADBOARD_SIZE[1] / 2, HEADBOARD_SIZE[2] / 2),
             (0.38, 0.25, 0.14, 1.0), collide=True)          # wood

    for side in ("left", "right"):
        pw = wb.add_body(name=f"pillow_{side}")
        pw.pos = list(PILLOWS[side])
        h = math.radians(PILLOW_PROP_DEG) / 2.0
        pw.quat = [math.cos(h), 0.0, math.sin(h), 0.0]     # tip about +y (prop against headboard)
        _add_box(pw, f"pillow_{side}_geom",
                 (PILLOW_SIZE[0] / 2, PILLOW_SIZE[1] / 2, PILLOW_SIZE[2] / 2),
                 (0.97, 0.95, 0.90, 1.0), collide=False)      # cream pillow


def _add_sheet(world: mujoco.MjSpec) -> None:
    """A draping fitted cover as a single kinematic (mocap) body: a top panel over the bed plus
    side + foot flaps that hang down over the edges. Rigid, so it never destabilizes the balance-
    critical policy; the record loop translates it headward as the two hands draw (grip-lock)."""
    wb = world.worldbody
    sheet = wb.add_body(name="sheet")
    sheet.mocap = True
    cx = SHEET_HEAD_X + SHEET_LEN / 2.0
    sheet.pos = [cx, 0.0, SHEET_Z]
    rgba = (0.46, 0.60, 0.82, 1.0)                            # distinct light-blue top sheet
    lx, ly = SHEET_LEN / 2.0, BED_SIZE[1] / 2.0
    # Top panel over the mattress width.
    g = sheet.add_geom()
    g.name = "sheet_top"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [lx, ly, SHEET_THICKNESS / 2.0]
    g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    # Side flaps: hang down the +/-y edges (9-in overhang folded down).
    flap = OVERHANG
    for side, sy in (("l", -1), ("r", +1)):
        fg = sheet.add_geom()
        fg.name = f"sheet_flap_{side}"
        fg.type = mujoco.mjtGeom.mjGEOM_BOX
        fg.size = [lx, SHEET_THICKNESS / 2.0, flap / 2.0]
        fg.pos = [0.0, sy * (ly + SHEET_THICKNESS / 2.0), -flap / 2.0]
        fg.rgba = list(rgba)
        fg.contype = 0
        fg.conaffinity = 0
    # Foot flap: hangs off the +x foot edge.
    ff = sheet.add_geom()
    ff.name = "sheet_flap_foot"
    ff.type = mujoco.mjtGeom.mjGEOM_BOX
    ff.size = [SHEET_THICKNESS / 2.0, ly, flap / 2.0]
    ff.pos = [lx + SHEET_THICKNESS / 2.0, 0.0, -flap / 2.0]
    ff.rgba = list(rgba)
    ff.contype = 0
    ff.conaffinity = 0


def build_scene() -> SceneHandles:
    """Compile the two-robot bed scene and return handles for the rollout."""
    from mjlab.entity.entity import Entity
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg

    world = mujoco.MjSpec()
    world.compiler.autolimits = True
    world.option.timestep = 0.005
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.visual.global_.offwidth = 1920
    world.visual.global_.offheight = 1080
    world.visual.headlight.diffuse = [0.28, 0.28, 0.30]
    world.visual.headlight.ambient = [0.22, 0.22, 0.24]
    world.visual.headlight.specular = [0.1, 0.1, 0.1]

    # Skybox + checker floor textures for depth.
    sky = world.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.30, 0.34, 0.42]
    sky.rgb2 = [0.09, 0.10, 0.13]
    sky.width = sky.height = 256
    ftex = world.add_texture()
    ftex.name = "floortex"
    ftex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ftex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    ftex.rgb1 = [0.26, 0.28, 0.32]
    ftex.rgb2 = [0.20, 0.22, 0.26]
    ftex.width = ftex.height = 512
    fmat = world.add_material()
    fmat.name = "floormat"
    fmat.textures = ["", "floortex"] + [""] * 8        # index 1 = mjTEXROLE_RGB (diffuse)
    fmat.texrepeat = [6, 6]
    fmat.reflectance = 0.0
    floor = world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], name="floor")
    floor.material = "floormat"

    # Soft key + fill lights, high and wide so the bed is evenly lit with no floor hotspots.
    for (lx, ly), diff in (((4.5, -4.5), 0.36), ((-3.5, 4.0), 0.24)):
        lt = world.worldbody.add_light()
        lt.pos = [lx, ly, 6.5]
        lt.dir = [-lx, -ly, -6.5]
        lt.diffuse = [diff, diff, diff]
        lt.specular = [0.0, 0.0, 0.0]

    _add_bed(world)
    _add_sheet(world)

    # Attach the two G1s at identity frames so each freejoint qpos == world pose.
    g1_specs = []
    for i in (0, 1):
        ent = Entity(get_g1_robot_cfg())
        g1_specs.append(ent.spec)
        world.attach(ent.spec, prefix=f"r{i}/", frame=world.worldbody.add_frame())

    model = world.compile()

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
            pelvis_body=_bid(model, pfx + "pelvis"),
            qpos_adr=qadr, dof_adr=dadr, act_id=aid,
            free_qadr=model.jnt_qposadr[fj], free_dofadr=model.jnt_dofadr[fj],
            imu_linvel_adr=_sensor_adr(model, pfx + "imu_lin_vel"),
            imu_angvel_adr=_sensor_adr(model, pfx + "imu_ang_vel"),
            imu_up_adr=_sensor_adr(model, pfx + "imu_upvector"),
            default_qpos=default_by_joint.copy(), pos=cfg["pos"], yaw_deg=cfg["yaw_deg"],
        ))

    return SceneHandles(
        model=model, robots=robots,
        sheet_mocap=int(model.body_mocapid[_bid(model, "sheet")]),
        sheet_body=_bid(model, "sheet"),
        default_qpos_by_joint=default_by_joint,
        cam_azimuth=CAM_AZIMUTH, cam_elevation=CAM_ELEVATION,
        cam_distance=CAM_DISTANCE, cam_lookat=CAM_LOOKAT,
    )


def set_init_pose(model, data, scene: SceneHandles) -> None:
    """Place each robot at its bedside stance (freejoint xyz+yaw) with the default joint pose."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for r in scene.robots:
        q = r.free_qadr
        data.qpos[q:q + 3] = r.pos
        data.qpos[q + 3:q + 7] = _yaw_quat(r.yaw_deg)
        for j in range(NUM_JOINTS):
            data.qpos[r.qpos_adr[j]] = r.default_qpos[j]
    # Place the sheet mocap at its authored rest pose.
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
          f"nmocap={m.nmocap}")
    d = mujoco.MjData(m)
    set_init_pose(m, d, sc)
    for r in sc.robots:
        pelvis_z = d.xpos[r.pelvis_body][2]
        lf = _sid(m, r.prefix + "left_foot")
        foot_z = d.site_xpos[lf][2]
        palm = d.site_xpos[r.hand_site]
        print(f"  {r.prefix} pelvis_z={pelvis_z:.3f} left_foot_z={foot_z:.3f} "
              f"{r.hand}_palm={palm.round(3).tolist()}")
    print(f"cam az={sc.cam_azimuth:.1f} el={sc.cam_elevation:.1f} dist={sc.cam_distance:.2f} "
          f"look={sc.cam_lookat}")
    perm, sign = build_mirror_tables()
    print("mirror perm", perm.tolist())
    print("mirror sign", sign.tolist())
