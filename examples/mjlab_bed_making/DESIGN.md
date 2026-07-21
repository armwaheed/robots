# G1 bed-reach/draw policy — training design (pure mjlab / MuJoCo-Warp)

Target: a whole-body G1 policy that balances while reaching onto the sheet AND holds the drag as it
draws the cover headward — the thing the earlier Isaac reach policy could not do. Trained on **mjlab /
MuJoCo-Warp** (no Isaac / Newton / NVIDIA-proprietary), exported to ONNX, and deployed in the two-robot
demo (`demo/two_robot_bed.py`). Reward/curriculum design is FALCON (arXiv 2505.06776, validated on a
real G1) adapted to our task; citations in the LP references section. (The reward suite below is
engine-agnostic; the sibling Newton example applies the same FALCON design on its own backend.)

## Root causes we are fixing (all measured on the Isaac policy)

1. **World-frame reach target → drift feedback loop.** Command the target in the PELVIS frame so
   stepping back does not change the error. (FALCON: "if your reach reward is world-frame, stepping
   backward changes the error — that alone can produce a drift feedback loop.")
2. **No proprioceptive history → actor cannot sense load → arm retracts.** Add a 5-step history
   stack to the actor obs; give the critic privileged EE force + root lin-vel.
3. **No station-keeping term → leaning-and-stepping is optimal.** Add the FALCON stance penalties +
   HuB CoM-over-support reward + a drift termination.
4. **Never trained under load.** EE force curriculum opposing the draw direction.

## Observation / action

- Actor obs (per step, then stack last 5): `[root_ang_vel_b(3), gravity_b(3), reach_cmd_pelvis(3),
  q_rel(29 body dofs, no fingers), qd(29), prev_action(29)]`. NOTE: drop root LINEAR velocity from the
  actor (a real G1 IMU can't measure it — matches Isaac Lab's own G1 distillation choice); keep it in
  the critic only.
- Critic obs = actor obs (unstacked) + privileged `[F_ee(3), root_lin_vel_b(3), cloth_load_est(1)]`.
- Action: 29 body-joint position deltas (legs+waist+arms; fingers held at grasp pose by the grip
  layer, not the policy), scale 0.5, added to default pose. 50 Hz control.
- Reach command sampled in the PELVIS frame from a reachability-filtered box (pre-sweep arm joints,
  keep collision-free EE poses); ~40% of envs get a STANCE command (no reach) per FALCON stand_prob.

## Rewards (weights are FALCON/HuB starting points — re-tune to our reward magnitudes)

| term | weight | form |
|---|---|---|
| reach_pos (pelvis frame) | +2.0 coarse / +1.5 fine | `exp(-‖ee-cmd‖²/σ²)`, σ=0.2 then 0.06 |
| **penalty_stance_root** | −5.0 | `‖proj_pelvis(pelvis-feet_mid)_{x,y}‖` — BOTH x and y (FALCON code ships y-only; a headward drag needs x) |
| **CoM over support** (HuB) | +tuned | `exp(-‖com_xy - support_foot_xy‖²/0.1²)` |
| penalty_shift_in_zero_command | −1.0 | base xy speed while commanded stance |
| close_feet / feet_split | −10 / −5 | keep the stance from splaying |
| upright | −1.0 | `flat_orientation_l2` |
| base_height | −0.5 | target 0.70 m |
| idle_arm_deviation | −0.2 | non-reaching arm stays at side |
| action_rate / dof_acc / torques | small neg | smoothness (sim-to-real) |
| termination | −250 | fall OR **drift > 0.25–0.30 m from episode-start anchor** (the single cheapest anti-drift signal) |

## Force curriculum (the payload fix)

- Apply an external force at the active wrist each physics step, opposing the draw (world −x-ish),
  magnitude ramped survival-gated: start α=0.1, ±40 N in xy, [−50,+5] N in z, contact point
  randomized wrist→fingertip, sustained 3–5 s, per-axis zero-force prob 0.25.
- **Torque-limit-aware cap** `f_max = min_j (τ_j^lim − τ_j^grav)/(|J_ee^{ji}|+ε)` with
  `dof_effort_limit_scale=0.9` — skip this and the curriculum stalls at α≈0.6 (FALCON measured).
- Ramp: episode survives >210 steps → α += 0.02; < 200 → α −= 0.02; clamp [0,1].

## Domain randomization (FALCON `domain_rand_rl_gym.yaml`, verified from repo)

link mass ×[0.9,1.2], base mass +[−1,3] kg, PD ×[0.9,1.1], friction [0.25,1.25], ctrl delay [0,1]
step, wrist mass +[0,2] kg (ULC — directly a payload proxy). Random pushes OFF (the EE force IS the
disturbance). Add temporally-correlated (OU) IMU noise if the lean proves brittle.

## Architecture escalation

Start single-policy (all 29 dofs, one actor). If it still fights itself after the anti-drift +
history changes, split upper/lower into FALCON's two-agent setup — the paper's thesis is that
upper/lower reward interference is what weight-tuning cannot fix. Do NOT start there; single-policy is
enough to test the reward suite.

## Deploy contract

Export TorchScript + ONNX. The Newton demo drives it exactly like the walking policy
(newton_g1_locomotion pattern): build obs in the SAME order, reorder by joint NAME, action*scale +
default → joint_target_pos. Validate the obs width against the exported policy before the first step.

## Deploy-side: the cloth, and what it says about the checkpoint

The demo deploys against a real deformable sheet (MuJoCo `flexcomp`, 504 verts) rather than a rigid
cover, which turns the "hold under load" claim into a measurement instead of an assertion. What the
cloth exposed about this checkpoint:

- **~0.10 m of headward palm travel per stroke** is the hard ceiling (unchanged from the rigid-cover
  measurement — it is the policy, not the payload). Six hand-over-hand strokes draw the sheet's head
  edge ~0.3 m up the bed; each stroke transfers less than the last as more of the sheet comes taut.
- **A single grip point is not enough** to move a 2.26 m-wide sheet: it stretches the cloth locally
  (6.6 cm of vertex travel → 2 cm of sheet). The demo grips a handful (8 vertices, 0.3 m radius),
  which is also what a hand does. A policy trained with the cloth in the loop could learn *where* to
  grab; this one cannot, since it never saw cloth.
- **Sheet stiffness is capped by the training timestep.** A sheet stiff enough to transmit a pull the
  way real cotton does (`young` 2e5) is unstable at the 5 ms step the policy trained at, and dropping
  the sim to 2 ms to fix that puts the policy off its marks within one rollout. So the shipped sheet is
  softer than real cloth (`young` 5e4) and absorbs part of every stroke in stretch. **Retraining at a
  smaller timestep is the clean fix** and should be bundled with the wider-`ReachCommand` retrain below.
- The next training iteration should therefore: widen the `ReachCommand` box for a full-length draw,
  train at the smaller step, and — if cloth-in-the-loop training is affordable — replace the scripted
  `apply_reach_force` wrench with the actual sheet, so the force the critic sees is the force the sheet
  applies.

## Milestones

1. Cartpole `zero_agent --physics newton_mjwarp` runs on the Spark (install viability). ← gating
2. G1 velocity-flat trains a few hundred iters; measure steps/sec on GB10 (throughput viability).
3. Port bed_reach_env_cfg to the manager-based Newton env; add the reward suite above; smoke-train.
4. Full train w/ force curriculum; export; deploy in the Newton bed demo; measure corner metric.
