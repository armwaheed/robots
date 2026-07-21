"""Drive one G1 with the trained mjlab bed-reach policy from raw MuJoCo state.

The trained policy is an ONNX ``obs[103] -> action[29]``. This rebuilds the exact 103-dim observation
mjlab feeds at train/deploy time, out of the shared two-robot ``mjData``:

    [ base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
      joint_pos_rel(29), joint_vel_rel(29), last_action(29),
      twist_cmd(3), reach_cmd(4) ]

``base_lin_vel`` / ``base_ang_vel`` come straight off the robot's IMU velocimeter / gyro sensors and
``projected_gravity`` off its up-vector sensor, so no frame math is re-derived here — the same sensors
the trained env read. ``joint_pos_rel`` subtracts the default pose; ``last_action`` is the previous
(raw) policy output; ``twist_cmd`` is zero (standing); ``reach_cmd`` is the scripted pelvis-frame goal.

The ``+y``-side robot runs the SAME single-hand policy through a **sagittal mirror**: its state is
reflected into the canonical (left-hand) frame, the policy runs, and the action is reflected back — so
one left-hand reach policy drives a right-hand reach on the mirrored robot. This is a standard way to
run a single-handed policy bimanually; it is not a second trained policy.
"""
from __future__ import annotations

import re

import numpy as np
import onnxruntime as ort

from bed_scene import JOINT_NAMES, NUM_JOINTS, RobotHandles, build_mirror_tables


def _build_scale() -> np.ndarray:
    """Per-joint action scale (0.25 * effort / stiffness), matched from mjlab's regex-keyed dict."""
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE
    scale = np.zeros(NUM_JOINTS)
    for i, name in enumerate(JOINT_NAMES):
        for pat, val in G1_ACTION_SCALE.items():
            if re.fullmatch(pat, name):
                scale[i] = val
                break
        else:
            raise KeyError(f"no action scale for {name}")
    return scale


# Mirror maps for the sagittal reflection (shared by all mirrored robots).
_MIRROR_PERM, _MIRROR_SIGN = build_mirror_tables()
# Base-term sign flips under the y->-y reflection: lin_vel & gravity are true vectors (flip y);
# ang_vel is a pseudovector (flip x and z).
_LINVEL_SIGN = np.array([1.0, -1.0, 1.0])
_ANGVEL_SIGN = np.array([-1.0, 1.0, -1.0])
_GRAV_SIGN = np.array([1.0, -1.0, 1.0])


class G1ReachController:
    """One policy-driven G1. ``compute_ctrl`` reads state, runs the policy, returns the 29 joint
    position targets to write into ``data.ctrl`` for this robot's actuators."""

    def __init__(self, session: ort.InferenceSession, robot: RobotHandles) -> None:
        self.s = session
        self.in_name = session.get_inputs()[0].name
        self.r = robot
        self.scale = _build_scale()
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)  # canonical raw action
        self.reach_b = np.zeros(3, dtype=np.float32)
        self.is_reach = 0.0

    def set_reach(self, target_b, is_reach: bool) -> None:
        self.reach_b = np.asarray(target_b, dtype=np.float32)
        self.is_reach = 1.0 if is_reach else 0.0

    def _obs(self, d) -> np.ndarray:
        r = self.r
        lin = d.sensordata[r.imu_linvel_adr:r.imu_linvel_adr + 3].copy()
        ang = d.sensordata[r.imu_angvel_adr:r.imu_angvel_adr + 3].copy()
        grav = -d.sensordata[r.imu_up_adr:r.imu_up_adr + 3].copy()
        qpos = d.qpos[r.qpos_adr] - r.default_qpos
        qvel = d.qvel[r.dof_adr].copy()
        if r.mirror:
            lin = lin * _LINVEL_SIGN
            ang = ang * _ANGVEL_SIGN
            grav = grav * _GRAV_SIGN
            qpos = _MIRROR_SIGN * qpos[_MIRROR_PERM]
            qvel = _MIRROR_SIGN * qvel[_MIRROR_PERM]
        obs = np.concatenate([
            lin, ang, grav, qpos, qvel, self.last_action,
            np.zeros(3, dtype=np.float32),                # twist command (standing)
            self.reach_b, [self.is_reach],
        ]).astype(np.float32)
        return obs[None, :]

    def compute_ctrl(self, d) -> np.ndarray:
        obs = self._obs(d)
        action = self.s.run(None, {self.in_name: obs})[0].reshape(-1).astype(np.float32)
        self.last_action = action                                     # store canonical raw action
        # Reflect the canonical (left-hand) action onto this robot's real joints if mirrored.
        real_action = _MIRROR_SIGN * action[_MIRROR_PERM] if self.r.mirror else action
        target = self.r.default_qpos + self.scale * real_action
        return target.astype(np.float64)

    def palm_world(self, d) -> np.ndarray:
        return d.site_xpos[self.r.hand_site].copy()
