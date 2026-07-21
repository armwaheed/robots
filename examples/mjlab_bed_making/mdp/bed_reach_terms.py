"""Custom MDP terms for the G1 whole-body bed-reach task (pure mjlab / MuJoCo-Warp).

These are the pieces mjlab's velocity task does not ship, and they are the reason the Isaac reach
policy was fragile. Every design choice traces to FALCON (arXiv 2505.06776, validated on a real
Unitree G1) or HuB (arXiv 2505.07294):

* a PELVIS-FRAME hand-reach command — a world-frame target makes stepping back reduce the error,
  which is the drift feedback loop we measured;
* a station-keeping penalty and a CoM-over-support reward that make "stay planted" beat "lean and
  step";
* an end-effector FORCE CURRICULUM so the policy trains under the very drag it must resist, instead
  of meeting a load for the first time at deploy;
* a drift termination — the single cheapest anti-drift signal.

Pure MuJoCo-Warp: no Isaac, no Newton, no NVIDIA-proprietary anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

# NOTE (API correction vs draft): quat_apply lives in mjlab.utils.lab_api.math, not
# mjlab.utils.math (which does not exist in mjlab 1.5.2).
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ── Pelvis-frame hand-reach command ──────────────────────────────────────────────────────────────
class ReachCommand(CommandTerm):
    """Sample a hand target in the ROBOT'S PELVIS FRAME and hold it for the episode segment.

    The command is stored in the pelvis frame precisely so that a robot which steps backward does not
    reduce its own reach error — the failure mode that let the Isaac policy walk itself off the
    bedside. Observations expose the same pelvis-frame vector, and the reach reward measures the
    active hand (a palm site) against the target transformed back to world.

    ``hand``: which palm site the command is for. For the bed draw each robot uses the hand on the
    headward side, committed up front, so the command is single-hand (no mid-task flip)."""

    cfg: ReachCommandCfg

    def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]
        self._site_id = self.robot.find_sites([cfg.hand_site])[0][0]
        self._pelvis_id = self.robot.find_bodies([cfg.pelvis_body])[0][0]
        # target in the pelvis frame (B, 3); a stance flag (B,) marks reach-vs-stand envs.
        self._target_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._is_reach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Episode-start base xy (the drift anchor), captured on the first compute() AFTER reset.
        self._start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # API correction vs draft: the base xy CANNOT be read inside reset() — at command-manager
        # reset time the reset events have written qpos but sim.forward() has not run yet, so
        # root_link_pos_w is stale (last step's terminal pose). We therefore defer the anchor read
        # to the first compute(), which always runs right after sim.forward() (see
        # ManagerBasedRlEnv.reset / .step). This flag marks envs whose anchor is still pending.
        self._anchor_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Pre-register the metric so it appears in the logs from the very first reset.
        self.metrics["reach_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        # [target_pelvis(3), is_reach(1)] — the policy sees both the goal and whether it is a reach.
        return torch.cat([self._target_b, self._is_reach.float().unsqueeze(1)], dim=1)

    def reset(self, env_ids=None) -> dict:
        # Mark these envs for a fresh anchor read on the next compute() (post-forward), then run the
        # normal command reset (metrics + resample). We deliberately do NOT read poses here.
        if env_ids is not None:
            self._anchor_pending[env_ids] = True
        return super().reset(env_ids)

    def compute(self, dt: float) -> None:
        super().compute(dt)
        # Capture the drift anchor for any env that just reset, now that sim.forward() has refreshed
        # the kinematics.
        if bool(self._anchor_pending.any()):
            ids = self._anchor_pending.nonzero().flatten()
            self._start_xy[ids] = self.robot.data.root_link_pos_w[ids, :2]
            self._anchor_pending[ids] = False

    @property
    def start_xy(self) -> torch.Tensor:
        return self._start_xy

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        lo = torch.tensor(self.cfg.ranges_lo, device=self.device)
        hi = torch.tensor(self.cfg.ranges_hi, device=self.device)
        self._target_b[env_ids] = lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        # A fraction of envs are pure STANCE (no reach) so the policy keeps a solid stand — FALCON's
        # stand_prob. Their target is ignored by the reach reward but the stance rewards still apply.
        self._is_reach[env_ids] = torch.rand(n, device=self.device) > self.cfg.stand_prob

    def _update_command(self) -> None:
        pass  # held constant in the pelvis frame for the whole segment (in-distribution, drift-proof)

    def _update_metrics(self) -> None:
        self.metrics["reach_error"] = torch.norm(
            self._active_hand_pos_w() - self.target_pos_w(), dim=-1
        )

    # helpers shared with the reward terms
    def target_pos_w(self) -> torch.Tensor:
        """The pelvis-frame target mapped to world for the current robot pose."""
        pelvis_pos = self.robot.data.body_link_pos_w[:, self._pelvis_id]
        pelvis_quat = self.robot.data.body_link_quat_w[:, self._pelvis_id]
        return pelvis_pos + quat_apply(pelvis_quat, self._target_b)

    def _active_hand_pos_w(self) -> torch.Tensor:
        return self.robot.data.site_pos_w[:, self._site_id]

    @property
    def is_reach(self) -> torch.Tensor:
        return self._is_reach


@dataclass(kw_only=True)
class ReachCommandCfg(CommandTermCfg):
    # NOTE (API correction vs draft): the base CommandTermCfg is a kw_only dataclass whose
    # ``resampling_time_range`` field has NO default, so this subclass must also be kw_only and the
    # env cfg must pass ``resampling_time_range=...`` explicitly. The original plain ``@dataclass``
    # draft could not be instantiated (missing required base field / positional-after-kwonly).
    asset_name: str = "robot"
    hand_site: str = "left_palm"
    pelvis_body: str = "pelvis"
    # Pelvis-frame reach box (m). Forward+lateral+down onto the bed, sampled from a reachable set.
    ranges_lo: tuple[float, float, float] = (0.20, 0.05, -0.35)
    ranges_hi: tuple[float, float, float] = (0.55, 0.45, 0.10)
    stand_prob: float = 0.4

    def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
        return ReachCommand(self, env)


# ── Rewards ──────────────────────────────────────────────────────────────────────────────────────
def reach_position(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    """Exponential reach reward: the active palm to the pelvis-frame target, world-transformed.
    Only active on reach envs (stance envs get no reach pressure)."""
    cmd: ReachCommand = env.command_manager.get_term(command_name)
    err = torch.norm(cmd._active_hand_pos_w() - cmd.target_pos_w(), dim=-1)
    return torch.exp(-(err**2) / std**2) * cmd.is_reach.float()


def stance_root_xy(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """FALCON penalty_stance_root, extended to BOTH x and y. Penalize the pelvis drifting off the
    midpoint of the feet in the ground plane — the direct anti-lean-and-step term. The shipped FALCON
    code penalizes y only; a headward drag needs x too, so both are included here."""
    robot: Entity = env.scene[asset_cfg.name]
    pelvis = robot.data.root_link_pos_w[:, :2]
    feet = robot.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (B, 2 feet, 2)
    feet_mid = feet.mean(dim=1)
    return torch.norm(pelvis - feet_mid, dim=-1)


def com_over_support(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """HuB centre-of-mass-over-support reward: keep the CoM (approximated by the pelvis xy) over the
    support-foot midpoint. This is the term that most directly kills the lean."""
    robot: Entity = env.scene[asset_cfg.name]
    com = robot.data.root_link_pos_w[:, :2]
    feet = robot.data.site_pos_w[:, asset_cfg.site_ids, :2]
    support = feet.mean(dim=1)
    return torch.exp(-torch.sum((com - support) ** 2, dim=-1) / std**2)


def base_drift(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
               command_name: str = "reach") -> torch.Tensor:
    """Distance of the base from where this episode started — for both a penalty and a termination.
    The anchor is captured on the first post-reset compute() by the reach command."""
    robot: Entity = env.scene[asset_cfg.name]
    start = env.command_manager.get_term(command_name).start_xy
    return torch.norm(robot.data.root_link_pos_w[:, :2] - start, dim=-1)


# ── End-effector force curriculum ────────────────────────────────────────────────────────────────
def apply_reach_force(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    force_scale: float,
    max_force: tuple[float, float, float],
    command_name: str = "reach",
) -> None:
    """Write a steady external wrench on the reaching wrist, opposing the draw — FALCON's force
    curriculum via MuJoCo ``xfrc_applied``. ``force_scale`` (0→1, ramped by the curriculum term)
    scales a per-episode random force sampled up to ``max_force``. Applied on reach envs only.

    This is how the policy learns to brace against the sheet's drag WITHOUT needing a two-way cloth
    coupling in training — exactly FALCON's method. The cloth itself is a deploy-time object.

    Event-manager contract (verified): ``func(env, env_ids, **params)``; ``env_ids`` is a subset
    tensor in ``interval`` mode, ``None`` in ``step`` mode.
    """
    robot: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cmd: ReachCommand = env.command_manager.get_term(command_name)
    hi = torch.tensor(max_force, device=env.device) * force_scale
    # Sample a wrench per env, opposing the reach (headward, world -x-ish); zero on stance envs.
    f = (2.0 * torch.rand(len(env_ids), 3, device=env.device) - 1.0) * hi
    f = f * cmd.is_reach[env_ids].float().unsqueeze(1)
    forces = f.unsqueeze(1)  # (N, 1, 3), world frame — one reaching wrist.
    torques = torch.zeros_like(forces)
    robot.write_external_wrench_to_sim(
        forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )


def force_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, term_name: str,
                     up_steps: int, down_steps: int, step: float, max_scale: float) -> float:
    """Survival-gated force ramp (FALCON): if the episode lasted long enough, push the force scale up;
    if it died early, pull it down. Returns the current scale for logging (Curriculum/force_curriculum).
    The scale is read by ``apply_reach_force`` via the event's params, updated in place here.

    Curriculum-manager contract (verified): ``func(env, env_ids, **params)`` is called from
    ``_reset_idx`` with the resetting env ids, BEFORE ``episode_length_buf`` is zeroed — so the
    length read here is the just-completed episode length (correct survival gate).
    """
    event_cfg = env.event_manager.get_term_cfg(term_name)
    scale = event_cfg.params.get("force_scale", 0.1)
    lengths = env.episode_length_buf[env_ids].float().mean().item()
    if lengths > up_steps:
        scale = min(scale + step, max_scale)
    elif lengths < down_steps:
        scale = max(scale - step, 0.0)
    event_cfg.params["force_scale"] = scale
    return scale


# ── Termination ──────────────────────────────────────────────────────────────────────────────────
def drifted(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, max_drift: float,
            command_name: str = "reach") -> torch.Tensor:
    """Terminate when the base has walked more than ``max_drift`` from its episode-start anchor.
    The single cheapest, strongest anti-drift signal (paired with the termination penalty)."""
    return base_drift(env, asset_cfg, command_name) > max_drift
