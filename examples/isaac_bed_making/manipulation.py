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

        self._torch = torch
        self.robot = robot
        self.side = side
        self.device = device
        # Resolve joint/body indices directly off THIS robot's articulation rather
        # than via SceneEntityCfg("robot"): the bed-making scene holds two robots
        # ("robot_0"/"robot_1"), so there is no entity literally named "robot".
        # find_joints(preserve_order=True) keeps ARM_JOINTS order, which the IK
        # uses consistently for the Jacobian columns, joint_pos, and the target.
        self.joint_ids, _ = robot.find_joints(ARM_JOINTS[side], preserve_order=True)
        body_ids, _ = robot.find_bodies([EE_LINK[side]], preserve_order=True)
        self.ee_body_id = int(body_ids[0])
        self.ej = self.ee_body_id - 1 if robot.is_fixed_base else self.ee_body_id
        # Palm link name varies by hand (Dex3 vs Inspire); fall back to the wrist
        # EE link if the named palm link isn't present.
        try:
            palm_ids, _ = robot.find_bodies([PALM_LINK[side]], preserve_order=True)
        except Exception:
            palm_ids = []
        self.palm_body_id = int(palm_ids[0]) if palm_ids else self.ee_body_id
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
        jac = self.robot.root_physx_view.get_jacobians()[:, self.ej, :, self.joint_ids]
        pw, qw = self._ee()
        jpd = self.ik.compute(pw, qw, jac, self.robot.data.joint_pos[:, self.joint_ids])
        self.robot.set_joint_position_target(jpd, joint_ids=self.joint_ids)
        return float(self._torch.norm(pw[0] - self.target[0]).item())

    def ee_pos(self) -> List[float]:
        return [float(x) for x in self._ee()[0][0].tolist()]

    def palm_pos(self) -> List[float]:
        """World position of the hand palm link (where the cloth grasp binds)."""
        p = self.robot.data.body_pose_w[:, self.palm_body_id]
        return [float(x) for x in p[0, 0:3].tolist()]

    def palm_path(self, robot_prim_path: str) -> str:
        return f"{robot_prim_path}/{PALM_LINK[self.side]}"


def apply_hand_friction(stage, robot_prim_path: str, side: str,
                        static_friction: float = 2.5, dynamic_friction: float = 2.5) -> int:
    """Give the hand's colliders a high-friction ("rubberized palm/fingertips")
    material so the hand grips the cloth by *friction* — no kinematic attachment.
    Binds the material to every collider under the hand/wrist subtree. Returns the
    number of colliders bound."""
    import isaaclab.sim as sim_utils
    from omni.physx.scripts import physicsUtils
    from pxr import UsdPhysics

    mat_path = f"{robot_prim_path}/grip_material_{side}"
    if not stage.GetPrimAtPath(mat_path):
        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=0.0)
        cfg.func(mat_path, cfg)

    bound = []
    # Inspire fingers use an R_/L_ prefix (e.g. "R_index_proximal", "R_thumb_distal");
    # Dex3 uses "{side}_hand_*". Match the wrist + both naming styles so the whole
    # gripping surface (palm + fingertips) gets the rubberized material.
    sp = "R_" if side == "right" else "L_"
    want = (f"{side}_hand", f"{side}_wrist",
            f"{sp}index", f"{sp}middle", f"{sp}thumb", f"{sp}ring", f"{sp}pinky")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(robot_prim_path):
            continue
        if not any(w in path for w in want):
            continue
        # Bind to the actual collider geometry (collision meshes) and to any prim
        # carrying a collision API — covers both how the G1 hand colliders appear.
        if prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in path.lower() or prim.GetTypeName() == "Mesh":
            try:
                physicsUtils.add_physics_material_to_prim(stage, prim, mat_path)
                bound.append(path)
            except Exception:
                pass
    return bound
