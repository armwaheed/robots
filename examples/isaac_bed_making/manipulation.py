"""G1 arm manipulation: world-frame differential IK + cloth grasp.

The G1 pelvis spawns rotated 90 deg about Z and 0.75 m up, so its root frame is
neither the world frame nor the identity. Isaac Lab's ``DifferentialIKController``
must therefore be driven **entirely in the world frame** — world EE pose, the
world-frame PhysX Jacobian, and a world-frame target. (Running it in the root
frame, as the stock ``run_diff_ik.py`` tutorial does, diverges for the G1
because that tutorial's robots have their root at the world origin.) Verified on
the DGX Spark: in-workspace targets converge to ~1 cm.

Grasping is a PhysX auto-attachment between the cloth and the hand's palm link;
releasing deletes it.
"""

from __future__ import annotations

from typing import List, Optional

ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
             "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
              "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
}
EE_LINK = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
PALM_LINK = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}


class ArmIK:
    """World-frame differential-IK driver for one G1 arm."""

    def __init__(self, robot, scene, side: str, device: str):
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.managers import SceneEntityCfg

        self._torch = torch
        self.robot = robot
        self.side = side
        self.device = device
        cfg = SceneEntityCfg("robot", joint_names=ARM_JOINTS[side], body_names=[EE_LINK[side]])
        cfg.resolve(scene)
        self.cfg = cfg
        self.ee_body_id = int(cfg.body_ids[0])
        self.ej = self.ee_body_id - 1 if robot.is_fixed_base else self.ee_body_id
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            num_envs=robot.num_instances if hasattr(robot, "num_instances") else 1,
            device=device,
        )
        self.target = None

    def _ee(self):
        p = self.robot.data.body_pose_w[:, self.ee_body_id]
        return p[:, 0:3], p[:, 3:7]

    def set_target(self, pos_w) -> None:
        """Command a world-frame EE position (orientation held free)."""
        t = self._torch.tensor([list(pos_w)], device=self.device, dtype=self._torch.float32)
        self.target = t
        _, qw = self._ee()
        self.ik.reset()
        self.ik.set_command(t, ee_quat=qw)

    def clear(self) -> None:
        self.target = None

    def tick(self) -> Optional[float]:
        """Advance one IK step toward the target. Returns distance-to-target (m)."""
        if self.target is None:
            return None
        jac = self.robot.root_physx_view.get_jacobians()[:, self.ej, :, self.cfg.joint_ids]
        pw, qw = self._ee()
        jpd = self.ik.compute(pw, qw, jac, self.robot.data.joint_pos[:, self.cfg.joint_ids])
        self.robot.set_joint_position_target(jpd, joint_ids=self.cfg.joint_ids)
        return float(self._torch.norm(pw[0] - self.target[0]).item())

    def ee_pos(self) -> List[float]:
        return [float(x) for x in self._ee()[0][0].tolist()]

    def palm_path(self, robot_prim_path: str) -> str:
        return f"{robot_prim_path}/{PALM_LINK[self.side]}"
