#!/usr/bin/env python3
"""Unitree G1 bed-making robotic system in MuJoCo.

The demo builds a scene with:
- a control G1 task planner,
- a worker G1 assisting as an effector,
- a 2.0 m x 1.8 m x 0.5 m bed object,
- a triangulated bed mesh,
- a MuJoCo flex-grid bedsheet that starts at the foot of the bed and is
  randomly dropped before the scripted task begins.

Usage:
    .venv/bin/python examples/unitree_g1_bed_making_demo.py
    .venv/bin/python examples/unitree_g1_bed_making_demo.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("STRANDS_ASSETS_DIR", str(REPO_ROOT / ".strands_robots" / "assets"))


def _maybe_reexec_with_mjpython() -> None:
    """MuJoCo passive viewer on macOS should run under mjpython."""

    if sys.platform != "darwin":
        return
    if not sys.stdin.isatty():
        return
    if "--dry-run" in sys.argv:
        return
    if "--no-viewer" in sys.argv and "--render-frames" not in sys.argv:
        return
    if os.path.basename(sys.executable) == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    if os.environ.get("STRANDS_MJPYTHON_REEXEC") == "1":
        return

    venv_mjpython = Path(sys.executable).with_name("mjpython")
    mjpython = str(venv_mjpython) if venv_mjpython.exists() else shutil.which("mjpython")
    if not mjpython:
        print(
            "Warning: MuJoCo viewer on macOS usually requires mjpython, but no mjpython executable was found. "
            "The simulation will continue without the automatic viewer handoff.",
            file=sys.stderr,
        )
        return

    env = dict(os.environ)
    env["STRANDS_MJPYTHON_REEXEC"] = "1"
    os.execve(mjpython, [mjpython, *sys.argv], env)


_maybe_reexec_with_mjpython()

from strands_robots.simulation import _configure_gl_backend  # noqa: E402
from strands_robots.simulation._model_registry import resolve_model  # noqa: E402
from strands_robots.tools.download_assets import download_robots  # noqa: E402


_configure_gl_backend()


BED_LENGTH_M = 2.0
BED_WIDTH_M = 1.8
BED_HEIGHT_M = 0.5
SHEET_LENGTH_M = 2.2
SHEET_WIDTH_M = 2.0
SHEET_GRID_X = 17
SHEET_GRID_Y = 15
SHEET_SPACING_X_M = SHEET_LENGTH_M / (SHEET_GRID_X - 1)
SHEET_SPACING_Y_M = SHEET_WIDTH_M / (SHEET_GRID_Y - 1)

CONTROL_PREFIX = "control_"
WORKER_PREFIX = "worker_"
SheetLayout = Dict[int, Tuple[float, float, float]]


@dataclass(frozen=True)
class MeshAssetPaths:
    bed_obj: Path
    bedsheet_obj: Path


@dataclass(frozen=True)
class RobotSceneSpec:
    role: str
    prefix: str
    position: Tuple[float, float, float]
    yaw_radians: float


@dataclass(frozen=True)
class SceneBuildResult:
    xml: str
    scene_path: Path
    mesh_assets: MeshAssetPaths
    robot_joint_names: Dict[str, List[str]]
    robot_actuator_names: Dict[str, List[str]]
    sheet_corner_nodes: Dict[str, int]
    bed_corner_targets: Dict[str, Tuple[float, float, float]]


@dataclass(frozen=True)
class PlacementStep:
    sheet_corner: str
    bed_corner: str
    label: str
    ask_worker_after: bool = False


@dataclass(frozen=True)
class BasePose:
    x: float
    y: float
    z: float
    yaw: float


def default_robot_specs() -> Tuple[RobotSceneSpec, RobotSceneSpec]:
    side_offset = BED_WIDTH_M / 2.0 + 0.5
    return (
        RobotSceneSpec("control", CONTROL_PREFIX, (0.0, side_offset, 0.0), -math.pi / 2.0),
        RobotSceneSpec("worker", WORKER_PREFIX, (0.0, -side_offset, 0.0), math.pi / 2.0),
    )


def placement_plan() -> Tuple[PlacementStep, ...]:
    return (
        PlacementStep("sheet_foot_left", "bed_head_right", "control robot locates a loose sheet corner"),
        PlacementStep("sheet_foot_right", "bed_head_left", "control robot traces the sheet edge to the next corner"),
        PlacementStep(
            "sheet_head_right",
            "bed_foot_left",
            "worker robot holds the partly placed sheet while the controller continues",
            ask_worker_after=True,
        ),
        PlacementStep(
            "sheet_head_left",
            "bed_foot_right",
            "worker robot assists while the controller places the final corner",
            ask_worker_after=True,
        ),
    )


def sheet_corner_nodes() -> Dict[str, int]:
    return {
        "sheet_foot_left": _sheet_node_index(0, 0),
        "sheet_foot_right": _sheet_node_index(0, SHEET_GRID_Y - 1),
        "sheet_head_left": _sheet_node_index(SHEET_GRID_X - 1, 0),
        "sheet_head_right": _sheet_node_index(SHEET_GRID_X - 1, SHEET_GRID_Y - 1),
    }


def bed_corner_targets(z_lift: float = 0.055) -> Dict[str, Tuple[float, float, float]]:
    x = BED_LENGTH_M / 2.0
    y = BED_WIDTH_M / 2.0
    z = BED_HEIGHT_M + z_lift
    return {
        "bed_foot_left": (-x, -y, z),
        "bed_foot_right": (-x, y, z),
        "bed_head_left": (x, -y, z),
        "bed_head_right": (x, y, z),
    }


def write_mesh_assets(mesh_dir: Path) -> MeshAssetPaths:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    bed_obj = mesh_dir / "bed_2m_x_1p8m_x_0p5m.obj"
    bedsheet_obj = mesh_dir / "bedsheet_2p2m_x_2m_triangles.obj"
    _write_obj(bed_obj, *_bed_mesh())
    _write_obj(bedsheet_obj, *_bedsheet_mesh())
    return MeshAssetPaths(bed_obj=bed_obj, bedsheet_obj=bedsheet_obj)


def build_scene(
    *,
    g1_xml_path: Path,
    output_dir: Path,
    robot_specs: Sequence[RobotSceneSpec] | None = None,
) -> SceneBuildResult:
    g1_xml_path = g1_xml_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_assets = write_mesh_assets(output_dir / "meshes")
    robot_specs = tuple(robot_specs or default_robot_specs())

    g1_root = ET.parse(g1_xml_path).getroot()
    g1_body = _required(g1_root.find("worldbody"), "worldbody").find("body")
    if g1_body is None:
        raise ValueError(f"No root body found in {g1_xml_path}")

    base_joint_names = [node.attrib["name"] for node in g1_body.iter("joint") if "name" in node.attrib]
    base_actuator_names = [
        node.attrib["name"] for node in _required(g1_root.find("actuator"), "actuator") if "name" in node.attrib
    ]

    scene = ET.Element("mujoco", {"model": "unitree_g1_two_humanoids_bed_making"})
    ET.SubElement(
        scene,
        "compiler",
        {
            "angle": "radian",
            "autolimits": "true",
            "meshdir": str((g1_xml_path.parent / "assets").resolve()),
        },
    )
    ET.SubElement(scene, "option", {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"})

    default = g1_root.find("default")
    if default is not None:
        scene.append(copy.deepcopy(default))

    asset = copy.deepcopy(_required(g1_root.find("asset"), "asset"))
    ET.SubElement(asset, "material", {"name": "bed_fabric", "rgba": "0.48 0.42 0.36 1"})
    ET.SubElement(asset, "material", {"name": "bedsheet_blue", "rgba": "0.16 0.38 0.88 0.78"})
    ET.SubElement(asset, "material", {"name": "floor_matte", "rgba": "0.78 0.78 0.74 1"})
    ET.SubElement(asset, "mesh", {"name": "bed_mesh", "file": str(mesh_assets.bed_obj.resolve())})
    scene.append(asset)

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"name": "key_light", "pos": "0 -2.5 3.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "light",
        {"name": "fill_light", "pos": "-2 2 2.5", "dir": "1 -1 -1", "diffuse": "0.4 0.4 0.4"},
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "-2.7 -3.0 2.0",
            "xyaxes": "0.74 -0.67 0 0.36 0.40 0.84",
            "fovy": "45",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "4 4 0.04",
            "material": "floor_matte",
            "condim": "3",
            "friction": "1 0.3 0.001",
        },
    )
    _add_bed(worldbody)
    _add_bedsheet_flex(worldbody)

    robot_joint_names: Dict[str, List[str]] = {}
    robot_actuator_names: Dict[str, List[str]] = {}
    for spec in robot_specs:
        worldbody.append(_clone_robot_body(g1_body, spec))
        robot_joint_names[spec.role] = [f"{spec.prefix}{name}" for name in base_joint_names]
        robot_actuator_names[spec.role] = [f"{spec.prefix}{name}" for name in base_actuator_names]

    actuator = ET.SubElement(scene, "actuator")
    source_actuator = _required(g1_root.find("actuator"), "actuator")
    for spec in robot_specs:
        for actuator_node in source_actuator:
            copied = copy.deepcopy(actuator_node)
            _prefix_reference_attrs(copied, spec.prefix, {"name", "joint"})
            actuator.append(copied)

    ET.indent(scene, space="  ")
    xml = ET.tostring(scene, encoding="unicode")
    scene_path = output_dir / "unitree_g1_bed_making_scene.xml"
    scene_path.write_text(xml)

    return SceneBuildResult(
        xml=xml,
        scene_path=scene_path,
        mesh_assets=mesh_assets,
        robot_joint_names=robot_joint_names,
        robot_actuator_names=robot_actuator_names,
        sheet_corner_nodes=sheet_corner_nodes(),
        bed_corner_targets=bed_corner_targets(),
    )


def pose_for_phase(prefix: str, side: str, phase: str) -> Dict[str, float]:
    active_sign = 1 if side == "left" else -1
    inactive_side = "right" if side == "left" else "left"
    inactive_sign = -active_sign
    waist_pitch = {
        "home": 0.02,
        "walk": 0.05,
        "reach": 0.36,
        "grasp": 0.43,
        "carry": 0.24,
        "trace": 0.30,
        "place": 0.38,
        "hold": 0.30,
    }.get(phase, 0.02)
    pose: Dict[str, float] = {
        f"{prefix}waist_yaw_joint": 0.12 * active_sign if phase in {"reach", "grasp", "place", "trace"} else 0.0,
        f"{prefix}waist_roll_joint": 0.04 * active_sign if phase in {"reach", "grasp", "place"} else 0.0,
        f"{prefix}waist_pitch_joint": waist_pitch,
    }
    _apply_leg_pose(pose, prefix, phase)
    _apply_arm_pose(pose, prefix, side, active_sign, phase)
    _apply_arm_pose(pose, prefix, inactive_side, inactive_sign, "balance" if phase == "walk" else "home")
    return pose


def worker_hold_pose(prefix: str = WORKER_PREFIX) -> Dict[str, float]:
    pose = {
        f"{prefix}waist_yaw_joint": 0.0,
        f"{prefix}waist_roll_joint": 0.0,
        f"{prefix}waist_pitch_joint": 0.28,
    }
    _apply_leg_pose(pose, prefix, "hold")
    _apply_arm_pose(pose, prefix, "left", 1, "hold")
    _apply_arm_pose(pose, prefix, "right", -1, "hold")
    return pose


def home_pose(prefix: str) -> Dict[str, float]:
    pose = {
        f"{prefix}waist_yaw_joint": 0.0,
        f"{prefix}waist_roll_joint": 0.0,
        f"{prefix}waist_pitch_joint": 0.02,
    }
    _apply_leg_pose(pose, prefix, "home")
    _apply_arm_pose(pose, prefix, "left", 1, "home")
    _apply_arm_pose(pose, prefix, "right", -1, "home")
    return pose


def blend_pose(start: Dict[str, float], target: Dict[str, float], alpha: float) -> Dict[str, float]:
    keys = set(start) | set(target)
    return {key: start.get(key, 0.0) + (target.get(key, 0.0) - start.get(key, 0.0)) * alpha for key in keys}


def default_base_pose(role: str) -> BasePose:
    for spec in default_robot_specs():
        if spec.role == role:
            return BasePose(spec.position[0], spec.position[1], 0.793, spec.yaw_radians)
    raise ValueError(f"Unknown robot role: {role}")


def base_pose_for_corner(role: str, bed_corner: str) -> BasePose:
    target = bed_corner_targets()[bed_corner]
    side_y = math.copysign(BED_WIDTH_M / 2.0 + 0.52, target[1])
    stand_x = max(-0.92, min(0.92, target[0] * 0.78))
    yaw = -math.pi / 2.0 if side_y > 0 else math.pi / 2.0
    if role == "worker":
        side_y = -side_y
        stand_x *= 0.85
        yaw = -math.pi / 2.0 if side_y > 0 else math.pi / 2.0
    return BasePose(stand_x, side_y, 0.793, yaw)


def blend_base(start: BasePose, target: BasePose, alpha: float) -> BasePose:
    yaw_delta = math.atan2(math.sin(target.yaw - start.yaw), math.cos(target.yaw - start.yaw))
    return BasePose(
        start.x + (target.x - start.x) * alpha,
        start.y + (target.y - start.y) * alpha,
        start.z + (target.z - start.z) * alpha,
        start.yaw + yaw_delta * alpha,
    )


def crumpled_sheet_layout(seed: int) -> SheetLayout:
    rng = random.Random(seed)
    layout: SheetLayout = {}
    for ix, iy, node in _sheet_nodes():
        u = ix / (SHEET_GRID_X - 1)
        v = iy / (SHEET_GRID_Y - 1)
        fold_a = math.sin(5.0 * math.pi * u + 1.7 * math.sin(3.0 * math.pi * v))
        fold_b = math.sin(4.0 * math.pi * v + 0.8 * math.cos(2.0 * math.pi * u))
        x = -1.45 + 0.68 * u + 0.10 * fold_b + rng.uniform(-0.018, 0.018)
        y = -0.32 + 0.64 * v + 0.13 * fold_a + rng.uniform(-0.018, 0.018)
        z = 0.56 + 0.22 * abs(fold_a) + 0.10 * abs(fold_b) + rng.uniform(-0.02, 0.035)
        layout[node] = (x, y, z)
    return layout


def final_sheet_layout(seed: int) -> SheetLayout:
    rng = random.Random(seed + 31)
    origin = (1.02, 0.87, BED_HEIGHT_M + 0.045)
    u_axis = (-SHEET_LENGTH_M * 0.985, -0.08, 0.0)
    v_axis = (-0.05, -SHEET_WIDTH_M * 0.965, 0.0)
    layout: SheetLayout = {}
    for ix, iy, node in _sheet_nodes():
        u = ix / (SHEET_GRID_X - 1)
        v = iy / (SHEET_GRID_Y - 1)
        wrinkle = 0.024 * math.sin(4.0 * math.pi * u + 0.7) * math.sin(3.0 * math.pi * v + 0.2)
        edge_sag = 0.0
        if ix in {0, SHEET_GRID_X - 1} or iy in {0, SHEET_GRID_Y - 1}:
            edge_sag = -0.035 * (0.45 + rng.random() * 0.55)
        x = origin[0] + u_axis[0] * u + v_axis[0] * v + 0.015 * math.sin(5.0 * math.pi * v)
        y = origin[1] + u_axis[1] * u + v_axis[1] * v + 0.018 * math.sin(4.0 * math.pi * u)
        z = origin[2] + wrinkle + edge_sag
        layout[node] = (x, y, max(BED_HEIGHT_M + 0.012, z))
    return layout


def progressive_sheet_layout(
    crumpled: SheetLayout,
    final: SheetLayout,
    placed_count: int,
    seed: int,
) -> SheetLayout:
    rng = random.Random(seed + placed_count * 17)
    coverage = min(1.0, placed_count / 4.0)
    layout: SheetLayout = {}
    for ix, iy, node in _sheet_nodes():
        u = ix / (SHEET_GRID_X - 1)
        v = iy / (SHEET_GRID_Y - 1)
        if placed_count == 1:
            local = (1.0 - u) * (1.0 - v)
            weight = min(0.58, 0.25 + 0.48 * local)
        elif placed_count == 2:
            local = 1.0 - u
            weight = min(0.76, 0.48 + 0.36 * local)
        elif placed_count == 3:
            local = max(u * v, 1.0 - abs(v - 0.5) * 1.2)
            weight = min(0.92, 0.70 + 0.24 * local)
        else:
            local = 1.0
            weight = 0.96
        weight = max(coverage * 0.72, weight)
        start = crumpled[node]
        end = final[node]
        slack_x = 0.025 * (1.0 - weight) * math.sin(3.0 * math.pi * v + placed_count)
        slack_y = 0.030 * (1.0 - weight) * math.sin(2.0 * math.pi * u + placed_count)
        slack_z = 0.045 * (1.0 - weight) * abs(math.sin(4.0 * math.pi * (u + v)))
        layout[node] = (
            start[0] + (end[0] - start[0]) * weight + slack_x + rng.uniform(-0.006, 0.006),
            start[1] + (end[1] - start[1]) * weight + slack_y + rng.uniform(-0.006, 0.006),
            start[2] + (end[2] - start[2]) * weight + slack_z,
        )
    return layout


def lift_corner_region(layout: SheetLayout, corner: str, hand_target: Tuple[float, float, float]) -> SheetLayout:
    corner_node = sheet_corner_nodes()[corner]
    cx, cy = _sheet_node_grid(corner_node)
    lifted = dict(layout)
    source_corner = layout[corner_node]
    for ix, iy, node in _sheet_nodes():
        grid_dist = math.hypot((ix - cx) / 3.0, (iy - cy) / 3.0)
        weight = max(0.0, 1.0 - grid_dist)
        if weight <= 0.0:
            continue
        node_pos = layout[node]
        relative = (
            node_pos[0] - source_corner[0],
            node_pos[1] - source_corner[1],
            node_pos[2] - source_corner[2],
        )
        target = (
            hand_target[0] + relative[0] * 0.72,
            hand_target[1] + relative[1] * 0.72,
            hand_target[2] + relative[2] * 0.45 + 0.08 * weight,
        )
        lifted[node] = _blend_vec(node_pos, target, weight * 0.88)
    return lifted


def blend_sheet_layout(start: SheetLayout, target: SheetLayout, alpha: float) -> SheetLayout:
    return {node: _blend_vec(start[node], target[node], alpha) for node in start}


def _apply_leg_pose(pose: Dict[str, float], prefix: str, phase: str) -> None:
    hip, knee, ankle = {
        "home": (0.02, 0.10, -0.05),
        "walk": (0.06, 0.18, -0.09),
        "reach": (0.28, 0.70, -0.34),
        "grasp": (0.36, 0.92, -0.44),
        "carry": (0.16, 0.42, -0.20),
        "trace": (0.22, 0.58, -0.28),
        "place": (0.30, 0.78, -0.37),
        "hold": (0.24, 0.64, -0.30),
    }.get(phase, (0.02, 0.10, -0.05))
    for side, sign in (("left", 1), ("right", -1)):
        pose[f"{prefix}{side}_hip_pitch_joint"] = hip
        pose[f"{prefix}{side}_hip_roll_joint"] = 0.035 * sign if phase in {"reach", "grasp", "place"} else 0.0
        pose[f"{prefix}{side}_hip_yaw_joint"] = 0.0
        pose[f"{prefix}{side}_knee_joint"] = knee
        pose[f"{prefix}{side}_ankle_pitch_joint"] = ankle
        pose[f"{prefix}{side}_ankle_roll_joint"] = -0.02 * sign if phase in {"reach", "grasp", "place"} else 0.0


def _apply_gait_pose(pose: Dict[str, float], prefix: str, gait_phase: float, intensity: float) -> None:
    if intensity <= 0.0:
        return
    swing = math.sin(gait_phase)
    for side, sign in (("left", 1), ("right", -1)):
        leg_swing = swing * sign
        pose[f"{prefix}{side}_hip_pitch_joint"] += 0.18 * intensity * leg_swing
        pose[f"{prefix}{side}_knee_joint"] += 0.16 * intensity * max(0.0, leg_swing)
        pose[f"{prefix}{side}_ankle_pitch_joint"] -= 0.08 * intensity * leg_swing
        pose[f"{prefix}{side}_hip_roll_joint"] += 0.025 * intensity * sign * math.cos(gait_phase)


def _apply_arm_pose(pose: Dict[str, float], prefix: str, side: str, sign: int, phase: str) -> None:
    side_prefix = f"{prefix}{side}_"
    values = {
        "home": (-0.05, 0.05, 0.0, 0.18, 0.0, 0.0, 0.0, False),
        "balance": (-0.25, 0.18, 0.05, 0.55, 0.0, -0.10, 0.04, False),
        "reach": (-0.98, 0.38, 0.34, 1.22, 0.12, -0.32, 0.20, False),
        "grasp": (-1.08, 0.46, 0.38, 1.36, 0.10, -0.48, 0.24, True),
        "carry": (-0.50, 0.28, 0.22, 0.92, 0.04, -0.12, 0.12, True),
        "trace": (-0.70, 0.34, 0.26, 1.08, 0.06, -0.18, 0.16, True),
        "place": (-0.92, 0.46, 0.36, 1.22, 0.08, -0.36, 0.22, True),
        "hold": (-0.58, 0.56, 0.26, 1.08, 0.04, -0.16, 0.14, True),
    }.get(phase, (-0.05, 0.05, 0.0, 0.18, 0.0, 0.0, 0.0, False))
    shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw, closed = values
    pose[f"{side_prefix}shoulder_pitch_joint"] = shoulder_pitch
    pose[f"{side_prefix}shoulder_roll_joint"] = shoulder_roll * sign
    pose[f"{side_prefix}shoulder_yaw_joint"] = shoulder_yaw * sign
    pose[f"{side_prefix}elbow_joint"] = elbow
    pose[f"{side_prefix}wrist_roll_joint"] = wrist_roll * sign
    pose[f"{side_prefix}wrist_pitch_joint"] = wrist_pitch
    pose[f"{side_prefix}wrist_yaw_joint"] = wrist_yaw * sign
    _apply_hand_pose(pose, side_prefix, side, closed)


def _apply_hand_pose(pose: Dict[str, float], side_prefix: str, side: str, closed: bool) -> None:
    if not closed:
        curl = 0.0
        thumb0 = 0.20 if side == "left" else -0.20
        thumb1 = 0.0
        thumb2 = 0.0
    elif side == "left":
        curl = -1.05
        thumb0 = 0.50
        thumb1 = 0.42
        thumb2 = 0.90
    else:
        curl = 1.05
        thumb0 = -0.50
        thumb1 = -0.42
        thumb2 = -0.90
    pose[f"{side_prefix}hand_thumb_0_joint"] = thumb0
    pose[f"{side_prefix}hand_thumb_1_joint"] = thumb1
    pose[f"{side_prefix}hand_thumb_2_joint"] = thumb2
    pose[f"{side_prefix}hand_index_0_joint"] = curl
    pose[f"{side_prefix}hand_index_1_joint"] = curl * 0.85
    pose[f"{side_prefix}hand_middle_0_joint"] = curl
    pose[f"{side_prefix}hand_middle_1_joint"] = curl * 0.85


def _bed_mesh() -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    x = BED_LENGTH_M / 2.0
    y = BED_WIDTH_M / 2.0
    z = BED_HEIGHT_M / 2.0
    vertices = [
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    ]
    triangles = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 8, 7),
        (5, 7, 6),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]
    return vertices, triangles


def _bedsheet_mesh() -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    vertices: List[Tuple[float, float, float]] = []
    for ix in range(SHEET_GRID_X):
        x = -SHEET_LENGTH_M / 2.0 + ix * SHEET_SPACING_X_M
        for iy in range(SHEET_GRID_Y):
            y = -SHEET_WIDTH_M / 2.0 + iy * SHEET_SPACING_Y_M
            vertices.append((x, y, 0.0))

    triangles: List[Tuple[int, int, int]] = []
    for ix in range(SHEET_GRID_X - 1):
        for iy in range(SHEET_GRID_Y - 1):
            a = ix * SHEET_GRID_Y + iy + 1
            b = (ix + 1) * SHEET_GRID_Y + iy + 1
            c = (ix + 1) * SHEET_GRID_Y + iy + 2
            d = ix * SHEET_GRID_Y + iy + 2
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return vertices, triangles


def _write_obj(
    path: Path,
    vertices: Iterable[Tuple[float, float, float]],
    triangles: Iterable[Tuple[int, int, int]],
) -> None:
    lines = [f"# Generated for Strands Robots Unitree G1 bed-making demo: {path.name}"]
    for vertex in vertices:
        lines.append("v " + _fmt(vertex))
    for tri in triangles:
        lines.append(f"f {tri[0]} {tri[1]} {tri[2]}")
    path.write_text("\n".join(lines) + "\n")


def _add_bed(worldbody: ET.Element) -> None:
    body = ET.SubElement(worldbody, "body", {"name": "bed", "pos": "0 0 0.25"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "bed_collision",
            "type": "box",
            "size": f"{BED_LENGTH_M / 2.0} {BED_WIDTH_M / 2.0} {BED_HEIGHT_M / 2.0}",
            "rgba": "0.48 0.42 0.36 1",
            "condim": "3",
            "friction": "1.0 0.4 0.001",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "bed_mesh_visual",
            "type": "mesh",
            "mesh": "bed_mesh",
            "material": "bed_fabric",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def _add_bedsheet_flex(worldbody: ET.Element) -> None:
    flex = ET.SubElement(
        worldbody,
        "flexcomp",
        {
            "name": "bedsheet",
            "type": "grid",
            "count": f"{SHEET_GRID_X} {SHEET_GRID_Y} 1",
            "spacing": f"{SHEET_SPACING_X_M:.6f} {SHEET_SPACING_Y_M:.6f} 0.02",
            "pos": f"{-(BED_LENGTH_M / 2.0 + 0.65):.3f} 0 0.72",
            "dim": "2",
            "mass": "0.55",
            "radius": "0.006",
            "rgba": "0.16 0.38 0.88 0.78",
        },
    )
    ET.SubElement(flex, "contact", {"contype": "1", "conaffinity": "1", "condim": "3", "friction": "1.4 0.03 0.001"})
    ET.SubElement(flex, "edge", {"equality": "true", "damping": "0.08"})


def _clone_robot_body(source_body: ET.Element, spec: RobotSceneSpec) -> ET.Element:
    body = copy.deepcopy(source_body)
    source_pos = _parse_floats(body.attrib.get("pos", "0 0 0"), expected=3)
    placed_pos = (
        spec.position[0] + source_pos[0],
        spec.position[1] + source_pos[1],
        spec.position[2] + source_pos[2],
    )
    body.set("pos", _fmt(placed_pos))
    body.set("quat", _fmt(_yaw_quat(spec.yaw_radians)))
    _prefix_reference_attrs(body, spec.prefix, {"name"})
    return body


def _prefix_reference_attrs(node: ET.Element, prefix: str, attrs: set[str]) -> None:
    for elem in node.iter():
        for attr in attrs:
            if attr in elem.attrib:
                elem.set(attr, f"{prefix}{elem.attrib[attr]}")


def _required(value: ET.Element | None, name: str) -> ET.Element:
    if value is None:
        raise ValueError(f"Required MJCF element missing: {name}")
    return value


def _sheet_node_index(ix: int, iy: int) -> int:
    return ix * SHEET_GRID_Y + iy


def _sheet_node_grid(node: int) -> Tuple[int, int]:
    return divmod(node, SHEET_GRID_Y)


def _sheet_nodes():
    for ix in range(SHEET_GRID_X):
        for iy in range(SHEET_GRID_Y):
            yield ix, iy, _sheet_node_index(ix, iy)


def _yaw_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _parse_floats(value: str, expected: int) -> Tuple[float, ...]:
    parsed = tuple(float(item) for item in value.split())
    if len(parsed) != expected:
        raise ValueError(f"Expected {expected} floats, got {value!r}")
    return parsed


def _blend_vec(
    start: Tuple[float, float, float],
    target: Tuple[float, float, float],
    alpha: float,
) -> Tuple[float, float, float]:
    return tuple(float(start[axis] + (target[axis] - start[axis]) * alpha) for axis in range(3))


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="make the bed", help="High-level task label to record in the summary.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "unitree_g1_bed_making"))
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=640, help="Render height.")
    parser.add_argument(
        "--render-every",
        type=int,
        default=24,
        help="Save every Nth control frame with --render-frames.",
    )
    parser.add_argument("--substeps", type=int, default=5, help="Physics substeps per control frame.")
    parser.add_argument("--settle-steps", type=int, default=260, help="Physics steps for the initial sheet drop.")
    parser.add_argument("--phase-steps", type=int, default=72, help="Control frames per lower-level task phase.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the initial bedsheet drop.")
    parser.add_argument("--no-viewer", action="store_true", help="Skip opening the MuJoCo passive viewer.")
    parser.add_argument("--render-frames", action="store_true", help="Export PNG frames during the scripted run.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Do not export PNG frames. Viewer still opens by default.",
    )
    parser.add_argument("--strict-render", action="store_true", help="Fail if rendering is unavailable.")
    parser.add_argument("--dry-run", action="store_true", help="Build and compile the scene, but do not simulate.")
    return parser.parse_args()


def _render_troubleshooting_hint() -> str:
    if sys.platform == "darwin":
        return (
            "On macOS, render from a normal Terminal session with `.venv/bin/mjpython "
            "examples/unitree_g1_bed_making_demo.py`, or run with `--no-render` in non-interactive shells."
        )
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return (
                "On headless Linux, use `--no-render` or install an offscreen MuJoCo GL backend. "
                "For CPU rendering, install OSMesa, for example `sudo apt-get install libosmesa6-dev`, "
                "then run with `MUJOCO_GL=osmesa`. For GPU rendering, install EGL/NVIDIA libraries and "
                "run with `MUJOCO_GL=egl`."
            )
        return (
            "On Linux desktop sessions, check that DISPLAY or WAYLAND_DISPLAY points to a working display "
            "and that OpenGL/GLFW system libraries are installed. Use `--no-render` for simulation-only runs."
        )
    return "Use `--no-render` for simulation-only runs, or configure a MuJoCo-compatible OpenGL backend."


def _viewer_troubleshooting_hint() -> str:
    if sys.platform == "darwin":
        return (
            "On macOS, launch the viewer from a normal Terminal session. The script will re-exec through "
            "`.venv/bin/mjpython` when needed. Use `--no-viewer` for non-interactive runs."
        )
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return (
                "On headless Linux there is no desktop display for the MuJoCo viewer. Use `--no-viewer`, "
                "or run under a desktop/X11/Wayland session."
            )
        return (
            "On Linux, check that DISPLAY or WAYLAND_DISPLAY points to a working display and that GLFW/OpenGL "
            "system libraries are installed. Use `--no-viewer` for simulation-only runs."
        )
    return "Use `--no-viewer` for simulation-only runs."


def _preflight_render_environment() -> None:
    if _is_darwin_noninteractive():
        raise RuntimeError(
            "MuJoCo rendering on macOS requires a foreground Terminal TTY or mjpython. "
            + _render_troubleshooting_hint()
        )
    if sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        has_gl_backend = bool(os.environ.get("MUJOCO_GL"))
        if not has_display and not has_gl_backend:
            raise RuntimeError(
                "MuJoCo rendering is not configured for this headless Linux session. "
                + _render_troubleshooting_hint()
            )


def _is_darwin_noninteractive() -> bool:
    return (
        sys.platform == "darwin"
        and not sys.stdin.isatty()
        and os.environ.get("STRANDS_MJPYTHON_REEXEC") != "1"
        and not os.environ.get("MJPYTHON_BIN")
    )


def _should_export_frames(args: argparse.Namespace) -> bool:
    return args.render_frames and not args.no_render


def _should_open_viewer(args: argparse.Namespace) -> bool:
    if args.no_viewer:
        return False
    if not sys.stdin.isatty():
        print(f"Skipping MuJoCo viewer in this non-interactive shell. {_viewer_troubleshooting_hint()}")
        return False
    return True


def _open_viewer(mujoco, model, data, args: argparse.Namespace):
    if not _should_open_viewer(args):
        return None
    if _is_darwin_noninteractive():
        raise RuntimeError("MuJoCo viewer on macOS requires mjpython. " + _viewer_troubleshooting_hint())

    try:
        import mujoco.viewer as viewer

        handle = viewer.launch_passive(model, data)
        print("Interactive MuJoCo viewer opened.")
        return handle
    except Exception as exc:
        raise RuntimeError(f"Could not open MuJoCo viewer: {exc}. {_viewer_troubleshooting_hint()}") from exc


def _sync_viewer(viewer_handle) -> None:
    if viewer_handle is None:
        return
    try:
        if hasattr(viewer_handle, "is_running") and not viewer_handle.is_running():
            return
        viewer_handle.sync()
    except Exception:
        pass


def _close_viewer(viewer_handle) -> None:
    if viewer_handle is None:
        return
    try:
        viewer_handle.close()
    except Exception:
        pass


def _ensure_g1_xml() -> Path:
    result = download_robots(names=["unitree_g1"])
    failures = result.get("failed_details", {})
    if failures:
        raise RuntimeError(f"Failed to prepare Unitree G1 assets: {failures}")

    resolved = resolve_model("unitree_g1")
    if not resolved:
        raise RuntimeError("Could not resolve the Unitree G1 MuJoCo scene.")

    resolved_path = Path(resolved)
    sibling_g1_with_hands = resolved_path.parent / "g1_with_hands.xml"
    if sibling_g1_with_hands.exists():
        return sibling_g1_with_hands
    sibling_g1 = resolved_path.parent / "g1.xml"
    return sibling_g1 if sibling_g1.exists() else resolved_path


def _load_mujoco_scene(scene_path: Path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _set_controls(mujoco, model, data, actuator_cache: Dict[str, int], pose: Dict[str, float]) -> None:
    for name, value in pose.items():
        act_id = actuator_cache.get(name)
        if act_id is None:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            actuator_cache[name] = act_id
        if act_id >= 0:
            data.ctrl[act_id] = float(value)


def _set_robot_base(mujoco, model, data, prefix: str, base: BasePose) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}floating_base_joint")
    if joint_id < 0:
        return
    qpos_addr = model.jnt_qposadr[joint_id]
    qvel_addr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_addr : qpos_addr + 3] = (base.x, base.y, base.z)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = _yaw_quat(base.yaw)
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def _sheet_body_id(mujoco, model, node_index: int) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bedsheet_{node_index}")
    if body_id < 0:
        raise RuntimeError(f"Could not find bedsheet node body bedsheet_{node_index}")
    return body_id


def _sheet_node_position(mujoco, model, data, node_index: int):
    body_id = _sheet_body_id(mujoco, model, node_index)
    return data.xpos[body_id].copy()


def _set_sheet_node_position(mujoco, model, data, node_index: int, target: Tuple[float, float, float]) -> None:
    body_id = _sheet_body_id(mujoco, model, node_index)
    joint_start = model.body_jntadr[body_id]
    joint_count = model.body_jntnum[body_id]
    if joint_count < 3:
        raise RuntimeError(f"Bedsheet node {node_index} does not have xyz slide joints")
    for axis in range(3):
        joint_id = joint_start + axis
        qpos_addr = model.jnt_qposadr[joint_id]
        qvel_addr = model.jnt_dofadr[joint_id]
        data.qpos[qpos_addr] = target[axis] - model.body_pos[body_id][axis]
        data.qvel[qvel_addr] = 0.0
    mujoco.mj_forward(model, data)


def _set_sheet_layout(mujoco, model, data, layout: SheetLayout) -> None:
    for node_index, target in layout.items():
        body_id = _sheet_body_id(mujoco, model, node_index)
        joint_start = model.body_jntadr[body_id]
        joint_count = model.body_jntnum[body_id]
        if joint_count < 3:
            raise RuntimeError(f"Bedsheet node {node_index} does not have xyz slide joints")
        for axis in range(3):
            joint_id = joint_start + axis
            data.qpos[model.jnt_qposadr[joint_id]] = target[axis] - model.body_pos[body_id][axis]
            data.qvel[model.jnt_dofadr[joint_id]] = 0.0
    mujoco.mj_forward(model, data)


def _read_sheet_layout(mujoco, model, data) -> SheetLayout:
    return {
        node_index: tuple(float(value) for value in _sheet_node_position(mujoco, model, data, node_index))
        for node_index in range(SHEET_GRID_X * SHEET_GRID_Y)
    }


def _crumple_and_drop_sheet(mujoco, model, data, seed: int, settle_steps: int) -> SheetLayout:
    layout = crumpled_sheet_layout(seed)
    _set_sheet_layout(mujoco, model, data, layout)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    return _read_sheet_layout(mujoco, model, data)


def _count_corner_drift(
    mujoco,
    model,
    data,
    scene: SceneBuildResult,
    placed: Dict[str, Tuple[float, float, float]],
) -> int:
    worker_assists = 0
    for corner, target in placed.items():
        node = scene.sheet_corner_nodes[corner]
        current = _sheet_node_position(mujoco, model, data, node)
        if math.dist(current, target) > 0.09:
            worker_assists += 1
    return worker_assists


def _hand_target_for_base(base: BasePose) -> Tuple[float, float, float]:
    side_sign = 1.0 if base.y > 0 else -1.0
    return (base.x, base.y - side_sign * 0.46, BED_HEIGHT_M + 0.22)


def _sheet_motion_weight(node: int) -> float:
    ix, iy = _sheet_node_grid(node)
    edge_x = min(ix, SHEET_GRID_X - 1 - ix) / max(1, SHEET_GRID_X // 2)
    edge_y = min(iy, SHEET_GRID_Y - 1 - iy) / max(1, SHEET_GRID_Y // 2)
    return 0.35 + 0.65 * (1.0 - min(edge_x, edge_y))


def _render_frame(mujoco, model, data, output_path: Path, width: int, height: int) -> None:
    _preflight_render_environment()

    from PIL import Image

    renderer = None
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        renderer.update_scene(data, camera=camera_id if camera_id >= 0 else None)
        img = renderer.render()
    except Exception as exc:
        raise RuntimeError(f"{exc}. {_render_troubleshooting_hint()}") from exc
    finally:
        if renderer is not None:
            renderer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)


def _run_phase(
    *,
    mujoco,
    model,
    data,
    scene: SceneBuildResult,
    actuator_cache: Dict[str, int],
    current_control_pose: Dict[str, float],
    target_control_pose: Dict[str, float],
    current_worker_pose: Dict[str, float],
    target_worker_pose: Dict[str, float],
    current_control_base: BasePose,
    target_control_base: BasePose,
    current_worker_base: BasePose,
    target_worker_base: BasePose,
    current_sheet_layout: SheetLayout,
    target_sheet_layout: SheetLayout | None,
    placed: Dict[str, Tuple[float, float, float]],
    label: str,
    args: argparse.Namespace,
    frame_state: Dict[str, int],
    viewer_handle,
    control_gait: bool = False,
    worker_gait: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], BasePose, BasePose, SheetLayout, int, str | None]:
    render_error = None
    worker_assists = 0
    next_sheet_layout = current_sheet_layout

    for step in range(args.phase_steps):
        alpha = (step + 1) / args.phase_steps
        pose = {}
        pose.update(blend_pose(current_control_pose, target_control_pose, alpha))
        pose.update(blend_pose(current_worker_pose, target_worker_pose, alpha))
        gait_phase = frame_state["control_frame"] * 0.65
        _apply_gait_pose(pose, CONTROL_PREFIX, gait_phase, 1.0 if control_gait else 0.0)
        _apply_gait_pose(pose, WORKER_PREFIX, gait_phase + math.pi, 0.75 if worker_gait else 0.0)
        _set_controls(mujoco, model, data, actuator_cache, pose)

        control_base = blend_base(current_control_base, target_control_base, alpha)
        worker_base = blend_base(current_worker_base, target_worker_base, alpha)
        _set_robot_base(mujoco, model, data, CONTROL_PREFIX, control_base)
        _set_robot_base(mujoco, model, data, WORKER_PREFIX, worker_base)

        if target_sheet_layout is not None:
            eased = 0.5 - 0.5 * math.cos(math.pi * alpha)
            lift = 0.06 * math.sin(math.pi * alpha) if "grasp" in label or "place" in label else 0.0
            next_sheet_layout = blend_sheet_layout(current_sheet_layout, target_sheet_layout, eased)
            if lift:
                next_sheet_layout = {
                    node: (pos[0], pos[1], pos[2] + lift * _sheet_motion_weight(node))
                    for node, pos in next_sheet_layout.items()
                }
            _set_sheet_layout(mujoco, model, data, next_sheet_layout)

        worker_assists += _count_corner_drift(mujoco, model, data, scene, placed)

        for _ in range(args.substeps):
            mujoco.mj_step(model, data)

        _sync_viewer(viewer_handle)
        frame_state["control_frame"] += 1
        if (
            _should_export_frames(args)
            and not frame_state.get("render_failed", 0)
            and frame_state["control_frame"] % args.render_every == 0
            and render_error is None
        ):
            frame_state["saved_frame"] += 1
            frame_path = Path(args.output_dir) / f"frame_{frame_state['saved_frame']:04d}_{label}.png"
            try:
                _render_frame(mujoco, model, data, frame_path, args.width, args.height)
            except Exception as exc:
                render_error = str(exc)
                frame_state["render_failed"] = 1
                if args.strict_render:
                    raise

    if target_sheet_layout is not None:
        next_sheet_layout = target_sheet_layout

    return (
        target_control_pose,
        target_worker_pose,
        target_control_base,
        target_worker_base,
        next_sheet_layout,
        worker_assists,
        render_error,
    )


def _run_make_bed(
    args: argparse.Namespace,
    scene: SceneBuildResult,
    mujoco,
    model,
    data,
    viewer_handle=None,
) -> Tuple[int, str | None, int]:
    actuator_cache: Dict[str, int] = {}
    placed: Dict[str, Tuple[float, float, float]] = {}
    frame_state = {"control_frame": 0, "saved_frame": 0, "render_failed": 0}
    render_error = None
    worker_assists = 0

    current_control = home_pose(CONTROL_PREFIX)
    current_worker = home_pose(WORKER_PREFIX)
    current_control_base = default_base_pose("control")
    current_worker_base = default_base_pose("worker")
    _set_controls(mujoco, model, data, actuator_cache, {**current_control, **current_worker})
    _set_robot_base(mujoco, model, data, CONTROL_PREFIX, current_control_base)
    _set_robot_base(mujoco, model, data, WORKER_PREFIX, current_worker_base)
    crumpled_layout = _crumple_and_drop_sheet(mujoco, model, data, args.seed, args.settle_steps)
    current_sheet_layout = crumpled_layout
    final_layout = final_sheet_layout(args.seed)

    for index, step in enumerate(placement_plan(), start=1):
        control_stand = base_pose_for_corner("control", step.bed_corner)
        worker_stand = (
            base_pose_for_corner("worker", step.bed_corner) if step.ask_worker_after else current_worker_base
        )

        (
            current_control,
            current_worker,
            current_control_base,
            current_worker_base,
            current_sheet_layout,
            assists,
            error,
        ) = _run_phase(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            actuator_cache=actuator_cache,
            current_control_pose=current_control,
            target_control_pose=pose_for_phase(CONTROL_PREFIX, "right", "walk"),
            current_worker_pose=current_worker,
            target_worker_pose=(
                pose_for_phase(WORKER_PREFIX, "left", "walk") if step.ask_worker_after else current_worker
            ),
            current_control_base=current_control_base,
            target_control_base=control_stand,
            current_worker_base=current_worker_base,
            target_worker_base=worker_stand,
            current_sheet_layout=current_sheet_layout,
            target_sheet_layout=None,
            placed=placed,
            label=f"{index:02d}_walk",
            args=args,
            frame_state=frame_state,
            viewer_handle=viewer_handle,
            control_gait=True,
            worker_gait=step.ask_worker_after,
        )
        worker_assists += assists
        render_error = render_error or error

        lifted_layout = lift_corner_region(
            current_sheet_layout,
            step.sheet_corner,
            _hand_target_for_base(current_control_base),
        )
        (
            current_control,
            current_worker,
            current_control_base,
            current_worker_base,
            current_sheet_layout,
            assists,
            error,
        ) = _run_phase(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            actuator_cache=actuator_cache,
            current_control_pose=current_control,
            target_control_pose=pose_for_phase(CONTROL_PREFIX, "right", "grasp"),
            current_worker_pose=current_worker,
            target_worker_pose=worker_hold_pose() if step.ask_worker_after else current_worker,
            current_control_base=current_control_base,
            target_control_base=current_control_base,
            current_worker_base=current_worker_base,
            target_worker_base=current_worker_base,
            current_sheet_layout=current_sheet_layout,
            target_sheet_layout=lifted_layout,
            placed=placed,
            label=f"{index:02d}_grasp",
            args=args,
            frame_state=frame_state,
            viewer_handle=viewer_handle,
        )
        worker_assists += assists
        render_error = render_error or error

        partial_layout = progressive_sheet_layout(crumpled_layout, final_layout, index, args.seed)
        (
            current_control,
            current_worker,
            current_control_base,
            current_worker_base,
            current_sheet_layout,
            assists,
            error,
        ) = _run_phase(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            actuator_cache=actuator_cache,
            current_control_pose=current_control,
            target_control_pose=pose_for_phase(CONTROL_PREFIX, "right", "place"),
            current_worker_pose=current_worker,
            target_worker_pose=worker_hold_pose() if step.ask_worker_after else current_worker,
            current_control_base=current_control_base,
            target_control_base=current_control_base,
            current_worker_base=current_worker_base,
            target_worker_base=current_worker_base,
            current_sheet_layout=current_sheet_layout,
            target_sheet_layout=partial_layout,
            placed=placed,
            label=f"{index:02d}_place",
            args=args,
            frame_state=frame_state,
            viewer_handle=viewer_handle,
        )
        placed[step.sheet_corner] = partial_layout[scene.sheet_corner_nodes[step.sheet_corner]]
        worker_assists += assists
        render_error = render_error or error

        (
            current_control,
            current_worker,
            current_control_base,
            current_worker_base,
            current_sheet_layout,
            assists,
            error,
        ) = _run_phase(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            actuator_cache=actuator_cache,
            current_control_pose=current_control,
            target_control_pose=pose_for_phase(CONTROL_PREFIX, "right", "trace"),
            current_worker_pose=current_worker,
            target_worker_pose=worker_hold_pose() if step.ask_worker_after else current_worker,
            current_control_base=current_control_base,
            target_control_base=current_control_base,
            current_worker_base=current_worker_base,
            target_worker_base=current_worker_base,
            current_sheet_layout=current_sheet_layout,
            target_sheet_layout=None,
            placed=placed,
            label=f"{index:02d}_trace",
            args=args,
            frame_state=frame_state,
            viewer_handle=viewer_handle,
        )
        worker_assists += assists
        render_error = render_error or error

    return frame_state["saved_frame"], render_error, worker_assists


def _write_summary(
    args: argparse.Namespace,
    scene: SceneBuildResult,
    model,
    saved_frames: int,
    render_error: str | None,
    worker_assists: int,
) -> None:
    lines = [
        "Unitree G1 two-humanoid bed-making demo complete.",
        f"Task: {args.task}",
        f"Scene: {scene.scene_path}",
        f"Bed mesh: {scene.mesh_assets.bed_obj}",
        f"Bedsheet mesh: {scene.mesh_assets.bedsheet_obj}",
        f"Robots: {', '.join(scene.robot_joint_names)}",
        f"Sheet flex corners: {scene.sheet_corner_nodes}",
        f"Bed targets: {bed_corner_targets()}",
        f"MuJoCo bodies: {model.nbody}, joints: {model.njnt}, actuators: {model.nu}, flexes: {model.nflex}",
        f"Frame export: {'enabled' if _should_export_frames(args) else 'disabled'}",
        f"Frames exported: {saved_frames}",
        f"Worker hold corrections: {worker_assists}",
        f"Render status: {render_error or 'ok'}",
    ]
    lines.extend(f"Plan {index}: {step.label}" for index, step in enumerate(placement_plan(), start=1))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g1_xml = _ensure_g1_xml()
    scene = build_scene(g1_xml_path=g1_xml, output_dir=output_dir)
    mujoco, model, data = _load_mujoco_scene(scene.scene_path)

    if args.dry_run:
        print(f"Built scene: {scene.scene_path}")
        print(f"Bodies={model.nbody} joints={model.njnt} actuators={model.nu} flexes={model.nflex}")
        print(f"Bed mesh: {scene.mesh_assets.bed_obj}")
        print(f"Bedsheet mesh: {scene.mesh_assets.bedsheet_obj}")
        return 0

    viewer_handle = None
    try:
        viewer_handle = _open_viewer(mujoco, model, data, args)
        saved_frames, render_error, worker_assists = _run_make_bed(
            args,
            scene,
            mujoco,
            model,
            data,
            viewer_handle=viewer_handle,
        )
    finally:
        _close_viewer(viewer_handle)

    _write_summary(args, scene, model, saved_frames, render_error, worker_assists)

    if render_error:
        print(f"Simulation completed, but rendering stopped after: {render_error}")
        print(f"Wrote summary to {output_dir / 'summary.txt'}")
    else:
        frame_text = f"Saved {saved_frames} frames and " if _should_export_frames(args) else ""
        print(f"{frame_text}Wrote summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
