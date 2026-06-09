"""Custom MDP terms for the bed-reach / bed-pull RL env.

The v1 bed-reach policy balanced + reached beautifully in FREE SPACE, but at deploy — standing
beside a bed, feet outside it, reaching over it — it stepped BACKWARD off its spot: reaching
forward shifts the CoM forward, the free base step-recovers backward, and (chasing a fixed sheet
target) it leans harder and walks itself over. Root cause: nothing in v1 penalized translating
the base, so the policy was free to reach by stepping rather than by leaning.

`base_xy_anchor_l2` adds that missing incentive: it penalizes the base's horizontal distance from
its per-env spawn anchor (the env origin), so the policy must reach by LEANING / SQUATTING with its
feet planted — exactly "lean over and pull the sheet without losing balance." The penalty is
quadratic, so a small lean shift (~0.1 m) is nearly free while a backward step (~0.5-1 m) is heavily
penalized. (Pair it with zero reset xy-noise so env_origin is the true spawn anchor.)
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def base_xy_anchor_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared horizontal (xy) distance of the robot base from its per-env spawn anchor.

    The anchor is the env origin (with the reset xy-noise zeroed, the robot spawns there), so this
    is the base's drift from where it started. Returns (num_envs,)."""
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_pos_w[:, :2]
    anchor = env.scene.env_origins[:, :2]
    return torch.sum(torch.square(xy - anchor), dim=1)


def randomize_ee_load(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    force_range: tuple[float, float] = (0.0, 35.0),
    slip_prob: float = 0.4,
) -> None:
    """Apply a random horizontal external force to the reaching hand — the bedsheet's tension/drag
    load — that RANDOMLY DROPS TO ZERO (grip slip / let-go). Run on an interval, so the load steps
    up and down every 1-2.5 s; the policy learns to absorb a SUDDEN load change without toppling
    ("don't fall when the sheet slips or you release it"). This is the FALCON-style force
    disturbance the loco-manipulation literature uses for force-adaptive whole-body control.

    The force is a horizontal vector of random magnitude in ``force_range`` and random direction;
    with probability ``slip_prob`` it is zero (no load / just slipped). Set on the EE body via the
    articulation's external-wrench buffer, so PhysX re-applies it every step until the next interval
    resamples it — the on↔off transitions at interval boundaries are the sudden changes to absorb."""
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    n = len(env_ids)
    nb = len(body_ids) if hasattr(body_ids, "__len__") else 1
    mag = torch.rand(n, device=env.device) * (force_range[1] - force_range[0]) + force_range[0]
    slipped = torch.rand(n, device=env.device) < slip_prob
    mag = torch.where(slipped, torch.zeros_like(mag), mag)
    theta = torch.rand(n, device=env.device) * 6.2831853
    forces = torch.zeros(n, nb, 3, device=env.device)
    forces[:, 0, 0] = mag * torch.cos(theta)
    forces[:, 0, 1] = mag * torch.sin(theta)
    torques = torch.zeros(n, nb, 3, device=env.device)
    asset.set_external_force_and_torque(forces, torques, body_ids=body_ids, env_ids=env_ids)
