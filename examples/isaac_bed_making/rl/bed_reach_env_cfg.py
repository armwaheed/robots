"""Whole-body bed-reach RL env — a free-base Inspire-hand G1 learns to balance while
reaching a commanded hand target in a forward+down workspace (the loco-manipulation skill
a pure walking policy can't hold: the deep bed-making lean throws the CoM past the feet).

Manager-based RL env on Isaac Lab's native rails (trains with the bundled rsl-rl-lib 5.x).
Physically valid for sim-to-real: NO base pinning / teleporting / joint freezing. The legs
balance the free base through the normal actuator path; the waist + arms reach. A deep reach
that would topple the robot must be handled by stepping / counter-leaning — like real hardware.

The reach target is sampled in the robot's BASE frame (UniformPoseCommand), so the policy is
yaw-invariant; at deploy the behavior layer commands the bed-corner positions as base-frame
targets. The hand tracked is the right `wrist_yaw_link` EE (left EE reserved for bimanual).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide as feet_slide_fn
from isaaclab_tasks.manager_based.manipulation.reach.mdp import position_command_error, position_command_error_tanh

from .robot_cfg import (
    BODY_JOINTS,
    FOOT_BODIES,
    PELVIS_BODY,
    RIGHT_EE_BODY,
    WAIST_JOINTS,
    make_bed_g1_cfg,
)

REACH_BODY = RIGHT_EE_BODY


##
# Scene
##
@configclass
class BedReachSceneCfg(InteractiveSceneCfg):
    """Flat ground + the free-base Inspire-hand G1. No bed: the reach target lives in a
    forward+down workspace that mimics the bed-corner positions; the bed is added at deploy."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = make_bed_g1_cfg(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # Eval-only overview camera (populated in BedReachEnvCfg_PLAY; None during training so it
    # adds no render cost). play.py reads its rgb each step and ffmpeg-encodes the mp4.
    eval_cam: CameraCfg | None = None

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP
##
@configclass
class CommandsCfg:
    """A hand-target pose command sampled in the robot's base frame (position-dominant)."""

    hand_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=REACH_BODY,
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # base frame: +x forward, +y left, +z up (pelvis-local). Forward + below pelvis
            # spans easy (near, ~pelvis height) to hard (far forward, low = deep bed lean).
            pos_x=(0.15, 0.50),
            pos_y=(-0.35, 0.10),
            pos_z=(-0.45, 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Position targets on the 29 body joints (legs + waist + arms). Inspire fingers are not
    part of the policy — they stay at default, held by their own actuator group."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=BODY_JOINTS, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_target"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.1),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )

    # Light perturbations for balance robustness (sim-to-real). Gentle so early learning isn't
    # swamped; a real robot nudged mid-reach must recover rather than topple.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    # -- task: reach the hand to the commanded target (coarse shaping + sharp bonus near it)
    reach_coarse = RewTerm(
        func=position_command_error_tanh,
        weight=2.0,
        params={
            "std": 0.20,
            "command_name": "hand_target",
            "asset_cfg": SceneEntityCfg("robot", body_names=REACH_BODY),
        },
    )
    reach_fine = RewTerm(
        func=position_command_error_tanh,
        weight=1.5,
        params={
            "std": 0.06,
            "command_name": "hand_target",
            "asset_cfg": SceneEntityCfg("robot", body_names=REACH_BODY),
        },
    )
    reach_l2 = RewTerm(
        func=position_command_error,
        weight=-0.3,
        params={"command_name": "hand_target", "asset_cfg": SceneEntityCfg("robot", body_names=REACH_BODY)},
    )

    # -- balance / staying alive (the hard part: hold balance through the lean)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": 0.70, "asset_cfg": SceneEntityCfg("robot", body_names=PELVIS_BODY)},
    )
    feet_slide = RewTerm(
        func=feet_slide_fn,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )

    # -- regularizers (smooth, sim-to-real-able motion)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*_joint", ".*_knee_joint"])},
    )
    # keep the lower body from splaying; let the WAIST stay near neutral but free to lean
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})  # ~57 deg pelvis tilt
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]), "threshold": 1.0},
    )


@configclass
class BedReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: BedReachSceneCfg = BedReachSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005  # 200 Hz physics, 50 Hz control
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BedReachEnvCfg_PLAY(BedReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # hold a fixed target band so the played behavior is easy to read by eye
        self.commands.hand_target.resampling_time_range = (4.0, 4.0)
        # per-env overview camera framing each robot (play.py sets the exact look-at pose)
        self.scene.eval_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/eval_cam",
            update_period=0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(2.4, 2.4, 1.7), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
