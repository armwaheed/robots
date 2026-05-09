"""Tests for the Unitree G1 two-humanoid bed-making demo scene."""

import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from examples import unitree_g1_bed_making_demo
from examples.unitree_g1_bed_making_demo import (
    SHEET_GRID_X,
    SHEET_GRID_Y,
    build_scene,
    final_sheet_layout,
    placement_plan,
    pose_for_phase,
    sheet_corner_nodes,
    write_mesh_assets,
)


def test_mesh_assets_have_requested_dimensions_and_triangles(tmp_path):
    assets = write_mesh_assets(tmp_path)

    bed_lines = assets.bed_obj.read_text().splitlines()
    sheet_lines = assets.bedsheet_obj.read_text().splitlines()

    assert sum(line.startswith("v ") for line in bed_lines) == 8
    assert sum(line.startswith("f ") for line in bed_lines) == 12
    assert sum(line.startswith("v ") for line in sheet_lines) == SHEET_GRID_X * SHEET_GRID_Y
    assert sum(line.startswith("f ") for line in sheet_lines) == (SHEET_GRID_X - 1) * (SHEET_GRID_Y - 1) * 2


def test_scene_builder_creates_two_prefixed_g1_instances(tmp_path):
    g1_xml = _write_minimal_g1_xml(tmp_path)

    scene = build_scene(g1_xml_path=g1_xml, output_dir=tmp_path / "out")
    root = ET.fromstring(scene.xml)

    body_names = {node.attrib["name"] for node in root.findall(".//body") if "name" in node.attrib}
    actuator_names = {node.attrib["name"] for node in root.findall(".//actuator/*") if "name" in node.attrib}

    assert "control_pelvis" in body_names
    assert "worker_pelvis" in body_names
    assert "control_right_shoulder_pitch_joint" in actuator_names
    assert "worker_right_shoulder_pitch_joint" in actuator_names
    assert root.find(".//flexcomp[@name='bedsheet']") is not None
    assert root.find(".//body[@name='bed']/geom[@name='bed_mesh_visual']") is not None
    assert scene.sheet_corner_nodes["sheet_head_right"] == SHEET_GRID_X * SHEET_GRID_Y - 1


def test_placement_plan_and_pose_targets_are_prefixed():
    plan = placement_plan()
    pose = pose_for_phase("control_", "right", "grasp")

    assert len(plan) == 4
    assert plan[0].sheet_corner == "sheet_foot_left"
    assert plan[-1].ask_worker_after is True
    assert "control_right_shoulder_pitch_joint" in pose
    assert "control_right_knee_joint" in pose
    assert "control_right_hand_index_0_joint" in pose
    assert "right_shoulder_pitch_joint" not in pose


def test_final_sheet_layout_keeps_adjacent_grid_spacing_realistic():
    layout = final_sheet_layout(seed=7)
    x_edges = []
    y_edges = []
    for ix in range(SHEET_GRID_X):
        for iy in range(SHEET_GRID_Y):
            node = ix * SHEET_GRID_Y + iy
            if ix + 1 < SHEET_GRID_X:
                x_edges.append(math.dist(layout[node], layout[(ix + 1) * SHEET_GRID_Y + iy]))
            if iy + 1 < SHEET_GRID_Y:
                y_edges.append(math.dist(layout[node], layout[ix * SHEET_GRID_Y + iy + 1]))

    assert abs(sum(x_edges) / len(x_edges) - unitree_g1_bed_making_demo.SHEET_SPACING_X_M) < 0.03
    assert abs(sum(y_edges) / len(y_edges) - unitree_g1_bed_making_demo.SHEET_SPACING_Y_M) < 0.03

    corners = sheet_corner_nodes()
    assert layout[corners["sheet_foot_left"]][0] > 0.95
    assert layout[corners["sheet_head_right"]][0] < -1.0


def test_linux_render_hint_does_not_reference_mjpython():
    original_platform = sys.platform
    original_display = os.environ.pop("DISPLAY", None)
    original_wayland = os.environ.pop("WAYLAND_DISPLAY", None)
    try:
        sys.platform = "linux"
        hint = unitree_g1_bed_making_demo._render_troubleshooting_hint()
    finally:
        sys.platform = original_platform
        if original_display is not None:
            os.environ["DISPLAY"] = original_display
        if original_wayland is not None:
            os.environ["WAYLAND_DISPLAY"] = original_wayland

    assert "mjpython" not in hint
    assert "MUJOCO_GL=osmesa" in hint


def _write_minimal_g1_xml(tmp_path: Path) -> Path:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    g1_xml = tmp_path / "g1.xml"
    g1_xml.write_text(
        """<mujoco model="minimal_g1">
  <compiler angle="radian" meshdir="assets"/>
  <default>
    <default class="g1">
      <joint armature="0.01"/>
      <position kp="100"/>
    </default>
  </default>
  <asset>
    <material name="metal" rgba="0.7 0.7 0.7 1"/>
  </asset>
  <worldbody>
    <body name="pelvis" pos="0 0 0.793" childclass="g1">
      <freejoint name="floating_base_joint"/>
      <body name="torso_link">
        <joint name="waist_yaw_joint" axis="0 0 1"/>
        <body name="right_shoulder_pitch_link">
          <joint name="right_shoulder_pitch_joint" axis="0 1 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position class="g1" name="waist_yaw_joint" joint="waist_yaw_joint"/>
    <position class="g1" name="right_shoulder_pitch_joint" joint="right_shoulder_pitch_joint"/>
  </actuator>
</mujoco>
"""
    )
    return g1_xml
