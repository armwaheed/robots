"""Drive a Unitree G1 in Newton with NVIDIA's shipped MuJoCo-Warp locomotion policy.

This is the load-bearing proof for the Isaac -> Newton port: the SAME robust walking policy that
ships in newton-assets (``rl_policies/mjw_g1_29DOF.pt``, trained in MuJoCo-Warp) runs on our G1 under
Newton's SolverMuJoCo, with NO retraining. It replaces the fragile custom velocity-walk path the
Isaac demo used.

Why this can't be the stock ``newton.examples robot_policy``: that example loads the G1 from USD, and
there is no ``usd-core`` wheel for aarch64 (the DGX Spark). We load the identical robot from the
bundled MJCF instead. The only wrinkle that creates is joint ORDER — the MJCF orders the finger
joints (thumb/index/middle) differently from the policy's ``mjw_joint_names``; the legs, waist and
arms already match. We resolve it by NAME, authoritatively, so the observation the policy sees and
the action it emits are always in its own trained order regardless of how the MJCF is laid out.

The observation contract is Isaac-Lab-locomotion-standard, read straight off the shipped example:
``[lin_vel_b(3), ang_vel_b(3), gravity_b(3), command(3), (q - q_default)(43), qd(43), prev_action(43)]``
at 50 Hz control (200 Hz sim, decimation 4), action = policy_out * 0.5 + q_default as position target.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
import yaml

import newton
import newton.utils
from newton import JointTargetMode

CONTROL_HZ = 50
SIM_HZ = 200
DECIMATION = SIM_HZ // CONTROL_HZ


def _quat_rotate_inverse(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse of a wxyz quaternion (Newton's quaternion convention)."""
    w = q_wxyz[:, 0:1]
    xyz = q_wxyz[:, 1:4]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(xyz, v, dim=-1) * w * 2.0
    c = xyz * torch.sum(xyz * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


class G1Locomotion:
    """One G1, driven by the shipped MuJoCo-Warp walking policy under Newton's SolverMuJoCo.

    Call :meth:`command` to steer (vx, vy, wz in the base frame) and :meth:`step` once per control
    tick. The policy holds balance at zero command and walks at a velocity command — the same single
    policy for stand and walk, which is exactly the robustness the Isaac path lacked.
    """

    def __init__(self, spawn=(0.0, 0.0, 0.80), yaw_deg=0.0, device=None):
        self.device = device or wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        asset = Path(newton.utils.download_asset("unitree_g1"))
        mjcf_path = asset / "mjcf" / "g1_29dof_with_hand_rev_1_0.xml"
        cfg = yaml.safe_load((asset / "rl_policies" / "g1_29dof.yaml").read_text())
        self.action_scale = float(cfg["action_scale"])
        mjw_names = cfg["mjw_joint_names"]
        self.n_dof = len(mjw_names)

        # Authoritative model joint order, read from the same MJCF MuJoCo will parse.
        mj = mujoco.MjModel.from_xml_path(str(mjcf_path))
        model_names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(mj.njnt) if mj.jnt_type[i] != 0]
        assert set(model_names) == set(mjw_names), "MJCF/policy joint sets disagree"

        # obs_reorder: model-order -> policy-order (for the observation the policy reads).
        # act_reorder: policy-order -> model-order (for the target the model applies).
        self.obs_reorder = torch.tensor([model_names.index(n) for n in mjw_names],
                                        device=self.torch_device, dtype=torch.long)
        self.act_reorder = torch.tensor([mjw_names.index(n) for n in model_names],
                                        device=self.torch_device, dtype=torch.long)

        self._build_model(mjcf_path, cfg, model_names, mjw_names, spawn, yaw_deg)
        self._load_policy(asset / cfg.get("policy_file", "rl_policies/mjw_g1_29DOF.pt"))

    def _build_model(self, mjcf_path, cfg, model_names, mjw_names, spawn, yaw_deg):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0)
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        yaw = np.radians(yaw_deg)
        rot = wp.quat(0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))  # xyzw about +Z
        builder.add_mjcf(str(mjcf_path), xform=wp.transform(wp.vec3(*spawn), rot),
                         floating=True, enable_self_collisions=False)
        builder.approximate_meshes("convex_hull")
        builder.add_ground_plane()

        # Default pose + per-joint gains arrive in POLICY order; scatter each into its model slot.
        default_pose_model = np.zeros(self.n_dof)
        for k, name in enumerate(mjw_names):
            mi = model_names.index(name)
            default_pose_model[mi] = cfg["mjw_joint_pos"][k]
            builder.joint_target_ke[6 + mi] = cfg["mjw_joint_stiffness"][k]
            builder.joint_target_kd[6 + mi] = cfg["mjw_joint_damping"][k]
            builder.joint_armature[6 + mi] = cfg["mjw_joint_armature"][k]
            builder.joint_target_mode[6 + mi] = int(JointTargetMode.POSITION)

        # Free-joint state is [x, y, z, qx, qy, qz, qw]; seed the actuated dofs at the default pose.
        builder.joint_q[:3] = list(spawn)
        builder.joint_q[3:7] = [rot[0], rot[1], rot[2], rot[3]]
        builder.joint_q[7:] = default_pose_model.tolist()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_cpu=False, solver="newton", nconmax=60, njmax=120)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.default_pose_model = torch.tensor(default_pose_model, device=self.torch_device,
                                               dtype=torch.float32).unsqueeze(0)
        self.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.torch_device)
        self.command_t = torch.zeros((1, 3), device=self.torch_device)
        self.prev_action = torch.zeros((1, self.n_dof), device=self.torch_device)
        self.sim_dt = 1.0 / SIM_HZ

    def _load_policy(self, policy_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.policy = torch.jit.load(str(policy_path), map_location=self.torch_device)
        self.policy.eval()

    def command(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        """Set the base-frame velocity command (m/s, m/s, rad/s). Zero = balance in place."""
        self.command_t[0, 0] = vx
        self.command_t[0, 1] = vy
        self.command_t[0, 2] = wz

    def _obs(self) -> torch.Tensor:
        jq = torch.tensor(self.state_0.joint_q.numpy(), device=self.torch_device).unsqueeze(0)
        jqd = torch.tensor(self.state_0.joint_qd.numpy(), device=self.torch_device).unsqueeze(0)
        root_quat = jq[:, 3:7]
        # Newton free-joint quat is stored xyzw; the policy's obs kernel expects wxyz.
        root_quat_wxyz = torch.cat([root_quat[:, 3:4], root_quat[:, 0:3]], dim=1)
        lin_vel_w = jqd[:, 0:3]
        ang_vel_w = jqd[:, 3:6]
        q_rel = jq[:, 7:] - self.default_pose_model
        qd = jqd[:, 6:]
        return torch.cat([
            _quat_rotate_inverse(root_quat_wxyz, lin_vel_w),
            _quat_rotate_inverse(root_quat_wxyz, ang_vel_w),
            _quat_rotate_inverse(root_quat_wxyz, self.gravity_vec),
            self.command_t,
            torch.index_select(q_rel, 1, self.obs_reorder),
            torch.index_select(qd, 1, self.obs_reorder),
            self.prev_action,
        ], dim=1).float()

    def step(self) -> None:
        """One 50 Hz control tick: run the policy, then advance DECIMATION physics substeps."""
        with torch.no_grad():
            action = self.policy(self._obs())
        self.prev_action = action
        target_model = (torch.index_select(action, 1, self.act_reorder) * self.action_scale
                        + self.default_pose_model)
        # Newton 1.2's Control exposes position targets as joint_target_pos (one entry per dof;
        # the 6 free-base dofs lead and take no target).
        joint_target = self.control.joint_target_pos.numpy()
        joint_target[6:] = target_model.squeeze(0).cpu().numpy()
        self.control.joint_target_pos = wp.array(joint_target, dtype=wp.float32, device=self.device)
        for _ in range(DECIMATION):
            self.contacts = self.model.collide(self.state_0)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def root_pose(self):
        jq = self.state_0.joint_q.numpy()
        return float(jq[0]), float(jq[1]), float(jq[2])


def main():
    print("[walk] building G1 + loading shipped MuJoCo-Warp policy")
    g1 = G1Locomotion(spawn=(0.0, 0.0, 0.80))

    print("[walk] phase 1: STAND (zero command) — the policy should hold balance")
    g1.command(0.0, 0.0, 0.0)
    for t in range(100):
        g1.step()
        if t % 25 == 0:
            x, y, z = g1.root_pose()
            print(f"  stand t={t:3d}  root=({x:+.2f},{y:+.2f},{z:.2f})")

    print("[walk] phase 2: WALK forward (vx=0.4 m/s)")
    g1.command(0.4, 0.0, 0.0)
    x0, _, _ = g1.root_pose()
    for t in range(200):
        g1.step()
        if t % 25 == 0:
            x, y, z = g1.root_pose()
            print(f"  walk  t={t:3d}  root=({x:+.2f},{y:+.2f},{z:.2f})")
    x1, _, z1 = g1.root_pose()

    upright = z1 > 0.55
    moved = (x1 - x0) > 0.15
    print(f"[walk] moved {x1 - x0:+.2f} m forward, final height {z1:.2f} m")
    print(f"[walk] {'PASS' if upright and moved else 'FAIL'} — "
          f"upright={upright} walked_forward={moved}")
    return 0 if (upright and moved) else 1


if __name__ == "__main__":
    raise SystemExit(main())
