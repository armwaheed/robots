"""Unitree G1 whole-body BED-REACH environment configuration (pure mjlab / MuJoCo-Warp).

Architecture (per the design): reuse mjlab's proven G1 *velocity* task — which already gives the G1
balance + locomotion + all the per-robot wiring (asset, foot sites, contact sensors, DR) — and
EXTEND it with the FALCON bed-reach machinery:

  1. a pelvis-frame ReachCommand (active palm -> target),
  2. a reach reward,
  3. FALCON station-keeping (stance_root_xy, com_over_support, drift penalty + drift termination),
  4. an EE force curriculum on the reaching wrist,
  5. a mostly-STANDING velocity command (the reach is done standing).

We build on the FLAT G1 velocity cfg (simpler + faster than rough for a standing reach; no terrain
scan obs).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .mdp.bed_reach_terms import (
    ReachCommandCfg,
    apply_reach_force,
    base_drift,
    com_over_support,
    drifted,
    force_curriculum,
    reach_position,
    stance_root_xy,
)

# Which hand reaches, and its matching wrist body (headward-side hand, committed up front).
_HAND_SITE = "left_palm"
_WRIST_BODY = "left_wrist_yaw_link"
_FEET_SITES = ("left_foot", "right_foot")


def make_bed_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the G1 bed-reach config by extending the flat G1 velocity task."""
    cfg = unitree_g1_flat_env_cfg(play=play)

    # ── 1. Make the velocity command mostly STANDING ────────────────────────────────────────────
    # The reach is performed standing, so we shrink the twist ranges toward zero and make most envs
    # pure-stance. We also drop the velocity-range curriculum (it would ramp the ranges back up) and
    # the random pushes (the EE force curriculum IS the disturbance).
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.rel_standing_envs = 0.85
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.ranges.lin_vel_x = (-0.3, 0.3)
    twist.ranges.lin_vel_y = (-0.2, 0.2)
    twist.ranges.ang_vel_z = (-0.3, 0.3)
    cfg.curriculum.pop("command_vel", None)
    cfg.events.pop("push_robot", None)

    # Loosen the standing posture reward on the arms/waist so the reach is not crushed by it (the
    # base cfg pins every joint to default with std 0.05). Legs stay tight for balance.
    cfg.rewards["pose"].params["std_standing"] = {
        r".*hip.*": 0.05,
        r".*knee.*": 0.05,
        r".*ankle.*": 0.05,
        r".*waist.*": 0.25,
        r".*shoulder.*": 1.0,
        r".*elbow.*": 1.0,
        r".*wrist.*": 1.0,
    }

    # ── 2. Pelvis-frame reach command ───────────────────────────────────────────────────────────
    cfg.commands["reach"] = ReachCommandCfg(
        resampling_time_range=(5.0, 5.0),
        asset_name="robot",
        hand_site=_HAND_SITE,
        pelvis_body="pelvis",
        # Reach box, pelvis frame. Centered on a region the arm can reach WHILE STANDING (natural
        # standing left_palm measured at pelvis-frame ~(0.15, 0.23, -0.08)); it still asks for a
        # forward+down reach onto the bed, but does not force a fall-inducing lean. The wider design
        # box (forward 0.20-0.55) is the deploy target and needs the full station-keeping tune.
        ranges_lo=(0.15, 0.10, -0.30),
        ranges_hi=(0.40, 0.35, 0.05),
        stand_prob=0.4,
    )

    # Expose the pelvis-frame reach command to the policy (actor) and critic.
    reach_obs = ObservationTermCfg(
        func=vel_mdp.generated_commands, params={"command_name": "reach"}
    )
    cfg.observations["actor"].terms["reach_command"] = reach_obs
    cfg.observations["critic"].terms["reach_command"] = reach_obs

    # ── 3. Reach reward + FALCON station keeping ────────────────────────────────────────────────
    cfg.rewards["reach_position"] = RewardTermCfg(
        func=reach_position,
        weight=2.0,
        params={"command_name": "reach", "std": 0.3},
    )
    cfg.rewards["stance_root_xy"] = RewardTermCfg(
        func=stance_root_xy,
        weight=-2.0,
        params={
            "command_name": "reach",
            "asset_cfg": SceneEntityCfg("robot", site_names=_FEET_SITES),
        },
    )
    cfg.rewards["com_over_support"] = RewardTermCfg(
        func=com_over_support,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=_FEET_SITES),
            "std": 0.15,
        },
    )
    cfg.rewards["base_drift"] = RewardTermCfg(
        func=base_drift,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "reach",
        },
    )
    # Termination penalty (design wants a large anti-drift/fall penalty; scaled down from -250 so it
    # does not swamp the shaped reward during the smoke run).
    cfg.rewards["terminated_penalty"] = RewardTermCfg(
        func=vel_mdp.is_terminated,
        weight=-20.0,
    )

    # ── 4. Drift termination ────────────────────────────────────────────────────────────────────
    cfg.terminations["drifted"] = TerminationTermCfg(
        func=drifted,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_drift": 0.50,
            "command_name": "reach",
        },
    )

    # ── 5. EE force curriculum on the reaching wrist ────────────────────────────────────────────
    if not play:
        cfg.events["apply_reach_force"] = EventTermCfg(
            func=apply_reach_force,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(_WRIST_BODY,)),
                "force_scale": 0.1,
                "max_force": (25.0, 25.0, 30.0),
                "command_name": "reach",
            },
        )
        cfg.curriculum["force_curriculum"] = CurriculumTermCfg(
            func=force_curriculum,
            params={
                "term_name": "apply_reach_force",
                "up_steps": 150,
                "down_steps": 140,
                "step": 0.05,
                "max_scale": 1.0,
            },
        )

    return cfg
