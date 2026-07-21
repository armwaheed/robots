"""Unitree G1 whole-body BED-REACH task (pure mjlab / MuJoCo-Warp, ZERO Isaac/Newton/NVIDIA).

Registers ``Mjlab-BedReach-Unitree-G1``. This package lives under ``mjlab.tasks`` (via a symlink to
its canonical location in the repo) so mjlab's task importer auto-imports it and the registration
side-effect fires for ``train`` / ``list-envs``.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .bed_reach_env_cfg import make_bed_reach_env_cfg
from .rl_cfg import unitree_g1_bed_reach_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-BedReach-Unitree-G1",
    env_cfg=make_bed_reach_env_cfg(),
    play_env_cfg=make_bed_reach_env_cfg(play=True),
    rl_cfg=unitree_g1_bed_reach_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
