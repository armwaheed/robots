"""Deploy driver for the trained G1 whole-body bed-reach policy (Isaac Lab 3.0 / Newton).

The policy was trained in the manager-based ``Isaac-BedReach-G1`` env; here we drive it in a bespoke
two-robot scene, reproducing its observation contract by hand off each robot's ``Articulation`` data
buffers. Contract (verified against the training config, no per-term scaling, noise off at play)::

    obs[127] = root_lin_vel_b(3) | root_ang_vel_b(3) | projected_gravity_b(3) | vel_cmd(3)=0
             | (joint_pos - default)(37) | joint_vel(37) | prev_action(37)
             | reach_cmd(4) = [target_pelvis_xyz(3), is_reach(1)]
    action[37] -> joint_pos_target = action * 0.5 + default_joint_pos     (JointPositionAction)

The reach command lives in the PELVIS frame; the policy reaches its ``left_palm_link`` toward
``pelvis + R_pelvis @ target_b``. It is a LEFT-hand policy, so the +y-side robot is driven through a
bilateral MIRROR: its sensed state is reflected into the policy's left-hand ("canonical") frame, the
policy runs there, and the resulting action is reflected back — turning it into a right-hand reacher
so both robots draw headward. The mirror is an involution built from the runtime joint names, so it
cannot silently drift from the USD's DOF order.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

# Joints whose value negates under a left<->right (sagittal-plane) mirror: the roll / yaw / lateral
# DOFs. Pitch, knee and finger-flex DOFs keep their sign. Matched against the default pose, which is
# itself bilaterally symmetric (e.g. right_one = -left_one), so mirror(default) == default.
_NEGATED_UNDER_MIRROR = (
    "hip_roll", "hip_yaw", "ankle_roll", "torso",
    "shoulder_roll", "shoulder_yaw", "elbow_roll",
    "one_joint", "two_joint",  # thumb rotation (right_one = -left_one in the default pose)
)


def _build_mirror(joint_names: List[str]) -> Tuple[List[int], List[float]]:
    """Return (perm, sign) for the left<->right joint mirror over ``joint_names``.

    ``mirrored[i] = sign[i] * value[perm[i]]``. ``perm`` swaps each ``left_*`` DOF with its ``right_*``
    twin (self-mapping for centreline DOFs like ``torso_joint``); ``sign`` negates the lateral DOFs.
    """
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    perm: List[int] = []
    sign: List[float] = []
    for name in joint_names:
        if name.startswith("left_"):
            twin = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            twin = "left_" + name[len("right_"):]
        else:
            twin = name  # centreline (e.g. torso_joint)
        perm.append(name_to_idx[twin])
        sign.append(-1.0 if any(tok in name for tok in _NEGATED_UNDER_MIRROR) else 1.0)
    return perm, sign


class BedReachPolicy:
    """One robot driven by the shared bed-reach policy; holds that robot's ``prev_action``.

    All robots share one loaded policy module. Construct one :class:`BedReachPolicy` per robot with
    its ``mirror`` flag; call :meth:`compute_targets` each control tick with the pelvis-frame reach
    command to get the 37 joint position targets to write to the sim.
    """

    def __init__(self, policy: torch.jit.ScriptModule, robot, device: str, mirror: bool = False):
        self.policy = policy
        self.robot = robot
        self.device = device
        self.mirror = mirror
        self.n_dof = robot.num_joints
        self.prev_action = torch.zeros(1, self.n_dof, device=device)  # canonical (policy) frame

        perm, sign = _build_mirror(list(robot.joint_names))
        self._perm = torch.tensor(perm, dtype=torch.long, device=device)
        self._sign = torch.tensor(sign, dtype=torch.float32, device=device)
        # Base-frame reflections for a sagittal-plane mirror: lin vel & gravity are polar vectors
        # (y flips); angular velocity is a pseudovector (x and z flip, y stays).
        self._lin_flip = torch.tensor([[1.0, -1.0, 1.0]], device=device)
        self._ang_flip = torch.tensor([[-1.0, 1.0, -1.0]], device=device)

    def _mirror_dofs(self, x: torch.Tensor) -> torch.Tensor:
        """Reflect a (1, n_dof) joint vector across the sagittal plane."""
        return x[:, self._perm] * self._sign

    def _build_obs(self, target_b: torch.Tensor, is_reach: float) -> torch.Tensor:
        d = self.robot.data
        lin = d.root_lin_vel_b.torch
        ang = d.root_ang_vel_b.torch
        grav = d.projected_gravity_b.torch
        q_rel = d.joint_pos.torch - d.default_joint_pos.torch
        q_vel = d.joint_vel.torch
        if self.mirror:
            lin = lin * self._lin_flip
            ang = ang * self._ang_flip
            grav = grav * self._lin_flip
            q_rel = self._mirror_dofs(q_rel)
            q_vel = self._mirror_dofs(q_vel)
        vel_cmd = torch.zeros(1, 3, device=self.device)
        reach_cmd = torch.cat(
            [target_b, torch.full((1, 1), float(is_reach), device=self.device)], dim=1
        )
        return torch.cat([lin, ang, grav, vel_cmd, q_rel, q_vel, self.prev_action, reach_cmd], dim=1)

    def compute_targets(self, target_b: torch.Tensor, is_reach: float = 1.0) -> torch.Tensor:
        """Run one policy step for the given pelvis-frame reach target; return (1, n_dof) targets.

        ``target_b`` is a (1, 3) pelvis-frame point in the LEFT-hand (canonical) frame — the SAME
        command for both robots; the mirror makes the +y robot reach with its right hand.
        """
        obs = self._build_obs(target_b, is_reach)
        with torch.inference_mode():
            action = self.policy(obs)  # canonical-frame raw action, (1, n_dof)
        self.prev_action = action
        action_real = self._mirror_dofs(action) if self.mirror else action
        return action_real * 0.5 + self.robot.data.default_joint_pos.torch
