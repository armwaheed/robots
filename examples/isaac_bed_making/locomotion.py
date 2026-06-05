"""Learned approach-walk for the G1 using Isaac Lab's pretrained **agile**
locomotion policy.

Instead of teleporting the robots to the bedside, each G1 walks there under a
learned RL policy. We reuse the pretrained leg policy shipped with Isaac Lab
(``{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt``) — the same one the
``Isaac-PickPlace-Locomanipulation-G1`` env uses. It is a *leg* policy: it
**observes the whole body** (29 joints) and **outputs 12 leg-joint targets** to
track a base velocity command while balancing, so the upper body is free for
manipulation.

The policy is run by hand (outside the RL env), reproducing the exact observation
the env builds (see ``configs/agile_locomotion_observation_cfg.py`` and
``mdp/actions.py`` in Isaac Lab)::

    policy_input = [ vx, vy, wz, hip_height,           # command (4)
                     base_lin_vel(3), base_ang_vel(3), # body frame
                     projected_gravity(3),
                     joint_pos_rel(29),                # all body joints
                     joint_vel_rel(29) * 0.1,
                     last_leg_action(12) ]             # previous policy output
    leg_target = policy(policy_input) * 0.25 + default_leg_pos

This is the policy's native robot (``G1_29DOF_CFG``); the same 29-joint ordering
is reproduced here by resolving the joints with the same name patterns.
"""

from __future__ import annotations

import math
from typing import Optional

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Leg joints the policy *controls* (output), and the body joints it *observes*.
LEG_RE = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
OBS_RE = [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint",
          ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]
POLICY_NUCLEUS_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
STAND_HIP_HEIGHT = 0.72       # nominal command hip height (from the standing retargeter)
OUTPUT_SCALE = 0.25           # policy_output_scale from the locomanip action cfg


def make_floating_base(stage, robot_prim_path: str) -> bool:
    """Turn a fixed-base-authored G1 USD into a **floating-base** articulation so a
    locomotion policy can move its root.

    The Inspire-hand G1 USD bakes a ``root_joint`` (a ``PhysicsFixedJoint`` that
    pins the pelvis to the world) and puts the ``ArticulationRootAPI`` on *that
    joint*. With ``fix_root_link=False`` Isaac still finds the root on the joint and
    fails to build a floating articulation. We **deactivate** the ``root_joint``
    (it lives in a referenced layer, so it can't be deleted — deactivating prunes
    it, and its world-pin + root API, from composition) and move the articulation
    root onto the ``pelvis`` body. Must run BEFORE ``sim.reset()``.
    """
    from pxr import PhysxSchema, UsdPhysics

    rj = stage.GetPrimAtPath(robot_prim_path + "/root_joint")
    pelvis = stage.GetPrimAtPath(robot_prim_path + "/pelvis")
    if not pelvis:
        return False
    if rj and rj.IsValid():
        rj.SetActive(False)
    UsdPhysics.ArticulationRootAPI.Apply(pelvis)
    PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
    return True


def load_agile_policy(device: str, path: Optional[str] = None):
    """Fetch + load the pretrained agile locomotion TorchScript policy (shareable
    across robots)."""
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.io.torchscript import load_torchscript_model

    return load_torchscript_model(retrieve_file_path(path or POLICY_NUCLEUS_PATH), device=device)


class LocomotionPolicy:
    """Drives one G1's legs from the shared agile policy to track a base command."""

    def __init__(self, robot, device: str, policy):
        import torch

        self._torch = torch
        self.robot = robot
        self.device = device
        self.policy = policy
        self.leg_ids, self.leg_names = robot.find_joints(LEG_RE)
        self.obs_ids, self.obs_names = robot.find_joints(OBS_RE)
        self.default_leg = robot.data.default_joint_pos[:, self.leg_ids].clone()
        self.last_action = torch.zeros(1, len(self.leg_ids), device=device)

    def reset(self) -> None:
        self.last_action = self._torch.zeros(1, len(self.leg_ids), device=self.device)

    def _obs(self, cmd):
        torch = self._torch
        d = self.robot.data
        base_lin = d.root_lin_vel_b[:, :3]
        base_ang = d.root_ang_vel_b[:, :3]
        grav = d.projected_gravity_b[:, :3]
        jpr = (d.joint_pos - d.default_joint_pos)[:, self.obs_ids]
        jvr = (d.joint_vel - d.default_joint_vel)[:, self.obs_ids] * 0.1
        cmd_t = torch.tensor([cmd], device=self.device, dtype=torch.float32)
        return torch.cat([cmd_t, base_lin, base_ang, grav, jpr, jvr, self.last_action], dim=-1)

    def act(self, cmd) -> None:
        """Run the policy for command ``[vx, vy, wz, hip_height]`` and set leg targets.
        Call once per control step (every ``decimation`` sim steps, ~50 Hz)."""
        out = self.policy.forward(self._obs(cmd))
        self.last_action = out.detach()
        tgt = out * OUTPUT_SCALE + self.default_leg
        self.robot.set_joint_position_target(tgt, joint_ids=self.leg_ids)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _base_yaw(self) -> float:
        q = self.robot.data.root_quat_w[0]  # (w, x, y, z)
        w, x, y, z = (float(v) for v in q.tolist())
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def base_xy(self):
        p = self.robot.data.root_pos_w[0, :2]
        return float(p[0]), float(p[1])

    def command_to(self, target_xy, *, stop_radius: float = 0.12,
                   max_vx: float = 0.45, turn_gain: float = 1.5):
        """Velocity command that walks the base toward ``target_xy`` (world). Returns
        ``(cmd, arrived)``. Walks forward in the heading direction and steers by yaw
        error; stops (zero command) within ``stop_radius``."""
        px, py = self.base_xy()
        dx, dy = target_xy[0] - px, target_xy[1] - py
        dist = math.hypot(dx, dy)
        if dist < stop_radius:
            return [0.0, 0.0, 0.0, STAND_HIP_HEIGHT], True
        yaw = self._base_yaw()
        heading = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        # Walk forward only when roughly facing the target; otherwise turn in place.
        vx = max_vx * max(0.0, math.cos(yaw_err)) * min(1.0, dist / 0.5)
        wz = max(-1.0, min(1.0, turn_gain * yaw_err))
        return [vx, 0.0, wz, STAND_HIP_HEIGHT], False

    def stand(self):
        """A pure standing command (balance in place)."""
        return [0.0, 0.0, 0.0, STAND_HIP_HEIGHT]
