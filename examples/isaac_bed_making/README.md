# Two Unitree G1s make a bed in NVIDIA Isaac Sim — coordinated over Arm Device Connect

Two **Unitree G1 humanoids** (with **Inspire 5‑finger hands**) **walk up to a bed
under a learned RL locomotion policy** and make it together in **NVIDIA Isaac Sim /
Isaac Lab**, coordinating as **equal peers over Arm Device Connect** — instead of
"head‑nodding," they claim work and ask each other for / offer help over a real
Device Connect swarm.

This is a self‑contained example: it does **not** modify the `strands_robots`
product package or Arm's Device Connect. It reuses the swarm driver from
[`examples/unitree_g1_bed_making_g1_driver.py`](../unitree_g1_bed_making_g1_driver.py).

![Two G1s reach over the bed](media/reach_over_bed.png)

| Walk in (learned policy) | Lean + reach (Pink IK arm + waist) |
| --- | --- |
| ![walk in](media/walk_in.png) | ![lean and reach](media/lean_reach.png) |

A full headless run renders an mp4 to **[`media/isaac_bed_making.mp4`](media/isaac_bed_making.mp4)**.

## What it shows

* **Learned locomotion walk‑in.** Each G1 starts ~0.85 m off its side of the bed
  and **walks in on a learned RL policy** — Unitree's open
  [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) G1 flat‑walk
  policy (LSTM + gait clock), driven with its exact 47‑dim observation and leg PD
  gains. Two G1s stride to the bedside, alternating legs, upright.
* **Established G1 manipulation IK (NVIDIA Pink/Pinocchio).** At the bedside each
  robot makes the bed with **Isaac Lab's `pink_ik` controller** — a weighted‑QP IK
  over the **arm *and* the 3‑DOF waist**, so the torso **leans into** the low reach
  the way a person bends to make a bed.
* **Real, physically‑simulated cloth.** A PhysX particle‑cloth sheet drapes over the
  bed under gravity and is gripped by the robots' hands — not a scripted animation.
* **Friction grasping, no cheating.** The hands have a high‑friction "rubberized"
  material; the cloth is gripped by **contact friction**, with no kinematic grasp.
* **Equal‑peer swarm coordination over Device Connect.** Both G1s pursue the goal
  state *"the bed is made"*; they claim work, emit events, and ask for / offer help.
  With `--broker` both peers register live on the Device Connect dashboard.
* **Optional: real motion from teleoperation data** (`--replay`) — see below.

## Two RL policies, each for what it's good at

The walk and the bedside stance want different things from the legs, so the demo
uses two policies:

* **Walk‑in** — Unitree's `unitree_rl_gym` G1 policy strides across the floor.
* **Bedside stance** — Isaac Lab's pretrained **agile** locomotion policy actively
  balances each robot in place while it settles. (The walk policy is a gait‑clock
  walker: at zero command it keeps marching and drifts, so it travels but does not
  hold a manipulation stance.)

Then, for the manipulation itself, the **pelvis is held** at its arrived pose. This
is deliberate and matches NVIDIA's own approach: a free‑floating humanoid cannot
balance through a bend‑over‑the‑bed reach with the publicly available policies, so
Isaac Lab ships a **fixed‑base** G1 manipulation env
(`FixedBaseUpperBodyIKG1EnvCfg`) for exactly this. The robot does the real learned
walk‑in floating, then plants firmly and the Pink IK arm + waist do the bed‑making.

## Adapting NVIDIA's Pink IK to the Inspire hand

Isaac Lab's `pink_ik` controller ships configured for the **Dex3** 3‑finger G1, but
this demo requires the **Inspire** 5‑finger USD. The controller maps the URDF onto
the sim robot **by joint name over the full joint set**, so three things were needed
(all bundled, self‑contained):

1. **A matching URDF.** `unitree_ros`'s
   `g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf` has the **same 53 actuated joint
   names** as the Isaac Lab Inspire USD. We ship a **kinematics‑only** copy
   (`assets/g1_inspire_kin.urdf`, visual/collision geometry stripped — Pinocchio
   needs only the kinematic tree, and it avoids a mesh‑loading binding bug).
2. **Import order.** `eigenpy`/`pinocchio` must be imported **before** the Isaac
   app launches, or the app shadows eigenpy's string‑vector converter and the
   controller fails to build. `demo.py` does this at the top.
3. The waist (3 DOF) is added to the IK so the torso can lean into the reach.

## Run it

On the DGX Spark, with Isaac Lab:

```bash
cd ~/workspaces/git/IsaacLab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"   # aarch64 caveat
PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
    ~/workspaces/git/robots/examples/isaac_bed_making/demo.py --loopback --render
```

Modes:

| Flag | Effect |
| --- | --- |
| `--loopback` | Coordinate via an in‑process bus (offline, default). |
| `--broker` | Register both peers on the real Device Connect NATS fabric (`.credentials/`). |
| `--no-device-connect` | Skip Device Connect entirely. |
| `--gui` | Open the Isaac Sim window to watch live (instead of headless). |
| `--render` | Capture frames and encode an mp4 into `artifacts/isaac_bed_making/`. |
| `--no-walk` | Skip the learned approach‑walk (start at the bedside). |
| `--walk-only` | Stop after the approach‑walk (to inspect locomotion). |
| `--replay` | Make the bed with the **dataset motion replay** instead of closed‑loop Pink IK. |
| `--replay-speed F` / `--lag S` | Replay playback speed / robot‑1 trail (with `--replay`). |

## Optional: real motion from teleoperation data (`--replay`)

With `--replay`, the waist + both arms are driven from a recorded trajectory of a
*real* teleoperated Unitree G1 making a bed — Source:
[`unitreerobotics/G1_WBT_Brainco_Make_The_Bed`](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Make_The_Bed)
(LeRobot v3.0, Apache‑2.0). Because the sim uses the actual Unitree G1 model, the
recorded joint angles transfer 1:1. `tools/extract_trajectory.py` pulls one episode
and writes the compact `data/bed_making_traj.npz` the demo replays. Regenerate it
(needs `pip install pyarrow huggingface_hub numpy`; not needed just to run the demo):

```bash
python examples/isaac_bed_making/tools/extract_trajectory.py --episode 4
```

## Rendering: cloth and robot together (the gotcha)

The robot articulation only renders its motion with **Fabric on** (Isaac Lab pushes
link poses to the renderer only when `use_fabric=True`), but PhysX does **not** sync
particle‑cloth deformation to Fabric. Per NVIDIA's forums, *mesh* updates do cross
Fabric (only point‑instancer ones don't), and our bedsheet is a `UsdGeom.Mesh`. So
the demo runs with `use_fabric=True` and **blits the live cloth positions** (read
from a PhysX tensor cloth‑view) **into the Fabric mesh points** each render — both
the moving robot and the deforming sheet show at once.

The cloth is built as a **quad** grid (not triangles): Isaac's auto particle‑cloth
turns every mesh edge into a stiff stretch spring, so a triangulated grid locks into
a rigid plate, while quads leave the diagonal as a soft shear spring and it drapes.

## Files

| File | Role |
| --- | --- |
| `demo.py` | Entry point: scene, learned walk‑in, base pin, Pink‑IK bed‑making, Device Connect, render. |
| `locomotion.py` | `RLGymWalker` (Unitree rl_gym walk) + `LocomotionPolicy` (agile balance stance) + floating‑base setup. |
| `manipulation.py` | `PinkArmIK` (NVIDIA Pink IK, arm + waist) + `apply_hand_friction` (rubberized hands). |
| `cloth.py` | PhysX particle‑cloth bedsheet (quad grid) + tensor‑view read + Fabric/usdrt blit for rendering. |
| `scene.py` | Scene geometry — two walk‑in G1s + bed + headboard + pillows + camera + sheet parameters. |
| `replay.py` | (`--replay`) Drives waist + arms (+ Inspire fingers) from the recorded dataset trajectory. |
| `coordination.py` | In‑process Device Connect swarm (loopback / real broker), wrapping the shared swarm driver. |
| `behavior.py` | Per‑robot autonomous decision state machine (scaffolding for emergent ordering). |
| `assets/g1_inspire_kin.urdf` | Kinematics‑only Inspire G1 URDF for the Pink IK solver. |
| `policies/g1_rlgym_walk.pt` | Unitree `unitree_rl_gym` G1 walk policy (BSD‑3) + its `.yaml` deploy config. |
| `data/bed_making_traj.npz` | Extracted real bed‑making trajectory (used by `--replay`). |

## Status

The learned walk‑in, the render reconciliation (robot + cloth both render), the
upright base‑pin manipulation, the NVIDIA Pink IK (arm + waist) reach, the friction
grasp, the cloth physics, and the Device Connect swarm all work end‑to‑end and are
verified by watching rendered frames.

**Known next step:** the sheet does not yet end up spread convincingly flat — it
starts folded back at the foot and the current grasp‑and‑pull doesn't fully flatten
it over the mattress. This is cloth‑geometry / manipulation‑strategy tuning (the IK
itself converges); a both‑hands grab + larger spread is the planned follow‑up.

Other roadmap items (see the issue #2 continuation comments): head‑mounted‑camera
coverage reported over Device Connect; a tolerant "good‑enough" goal; and, as a
stretch, an imitation‑learning policy trained on the dataset rather than replayed,
and a fully Isaac‑Lab‑native (`unitree_rl_lab`) walk once its rsl_rl version is
reconciled.
