# Two G1s make a bed — pure mjlab / MuJoCo-Warp (no NVIDIA / Isaac / Newton)

Two Unitree G1 humanoids flank a correctly-proportioned wide low bed and **make it together**,
coordinating over the **Model Hardware Standard (MHS)** — rendered entirely on
**[mjlab](https://github.com/mujocolab/mjlab) / MuJoCo-Warp**. No Isaac Sim, no Isaac Lab, no Newton,
nothing NVIDIA-proprietary. Both robots are driven by the trained whole-body bed-reach policy
(`policy/bed_reach_g1.onnx`); the second robot runs the same single-hand policy through a sagittal
mirror, so one policy drives a bimanual pair.

![the two robots making the bed](media/frames/frame_05_release.png)

The result — `media/bed_making.mp4` — reads as *two humanoids making a bed*: a **2.0 × 1.8 × 0.61 m**
mattress (wide, low, long), a headboard, two propped pillows, and a light-blue cover that drapes off
the sides and is drawn headward toward the pillows.

## The demo (this is the `examples/mjlab_bed_making/demo/` deliverable)

```bash
# from the repo root, with the mjlab venv on PATH (see "Install" below)
python examples/mjlab_bed_making/demo/two_robot_bed.py            # renders media/bed_making.mp4
python examples/mjlab_bed_making/demo/two_robot_bed.py --probe    # fast, no render: logs the reach envelope
python examples/mjlab_bed_making/demo/two_robot_bed.py --coord broker --nats-url nats://127.0.0.1:4222
```

| file | what it is |
| --- | --- |
| `demo/bed_scene.py` | Builds ONE MuJoCo model with **two** mjlab-configured G1s (attached with `MjSpec.attach`, each keeping its own floating base, actuators, IMU sensors) + the bed. **Bed geometry is copied verbatim from `examples/isaac_bed_making/scene.py`** — the authoritative target. Also holds the sagittal-mirror tables. |
| `demo/g1_policy.py` | Rebuilds the exact 103-dim observation mjlab feeds the policy, out of raw MuJoCo state, and runs the ONNX per robot. The +y robot is driven through the sagittal mirror. |
| `demo/two_robot_bed.py` | The rollout + render: scripts the pelvis-frame reach, grip-locks the cover, wires the MHS coordination, and writes the MP4 + trace. |

### What is the trained policy, and what is an abstraction

The issue asks for honesty here, so:

* **Trained (the hard, valuable part).** Both robots' **balanced whole-body reach onto the bed**, the
  **hold under load**, and the **~0.1 m headward hand draw** — done while staying planted at the
  bedside (the base never drifts or falls across the 9 s clip). This is the loco-manipulation-under-load
  skill the earlier Isaac reach policy never had; it comes straight from the FALCON station-keeping +
  force-curriculum training in this package (see `DESIGN.md`). The `--probe` mode logs the measured
  reach envelope for both robots — they are exact mirror images, which validates both the observation
  reconstruction and the mirror.
* **Abstraction (documented).** The **grip** and the cover's **full bed-making travel** are a
  *grip-lock*: once both hands are on the head edge, the gathered cover feeds headward toward the
  pillows as they draw. The trained hand draw is ~0.1 m per stroke (the policy's measured limit); the
  cover's full travel models the gathered-cover accordion unspooling. A single-stroke full draw across
  a **1.8 m-wide** bed from a stance 0.2 m off the side is out of reach for the current checkpoint — it
  would need the wider `ReachCommand` box + retrain that `DESIGN.md` describes. Per the issue, the bed
  proportions and the two-robot MHS coordination are real and not abstracted; the draw distance is.

### The second robot: a sagittal mirror, not a second policy

The trained policy reaches with the **left** hand. The +y-side robot needs its **right** hand to draw
the same head edge, so `g1_policy.py` runs the identical policy in a mirrored frame: its state is
reflected into the canonical (left-hand) frame, the policy runs, and the action is reflected back
(left/right joints swapped, roll/yaw joints negated). This is a standard way to run a single-handed
policy bimanually. The two robots are then exact mirror images across the bed's centre-line.

### MHS coordination

The two robots come up as **equal peers** on the MHS fabric using the **engine-agnostic coordination
layer that already exists** in `examples/isaac_bed_making/` (`coordination.py`, `swarm_driver.py`,
`mhs_trace.py`) — **imported, not copied**; those modules import nothing from any simulator. During
the make, the peers claim their head-side corners (`pickUpBedSheet`), emit `claimCorner` / `cornerHeld`
events, trade `askForHelp` / `offerHelp` (peer 0 asks, peer 1's `@on` handler auto-offers), place their
corners (`putDownBedSheet`), and reach `goalReached` at 4/4. Every message is recorded with its plane /
transport / profile and its simulator effect, and written to `media/mhs_trace.json`.

`--coord loopback` (default) fans events in-process — offline, no broker — and is labelled `loopback`
in the trace so it is never mistaken for a real fabric. `--coord broker` mounts both peers on a real
NATS fabric (needs the `mhs` SDK + a `nats-server`), the path any other fleet client would take. (Note:
the shared loopback path records each `@emit` publish twice — once at the driver, once at the
in-process bus — so `event.publish` counts are 2× the logical events on that path; the broker path
records once. This is the reused layer's behaviour, left unmodified.)

## Install (aarch64 / GB10, all open packages)

Use plain `pip`, **not** `uv` — mjlab's `uv` sources force an x86 torch that will not run on the GB10.

```bash
python3 -m venv mjlab-venv
mjlab-venv/bin/pip install --upgrade pip
mjlab-venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
mjlab-venv/bin/pip install mjlab onnxruntime mediapy   # mjlab 1.5.2 pulls mujoco-warp 3.10.0.2, warp 1.15.0
```

The demo needs `mjlab` (for the G1 asset + actuators), `mujoco`, `onnxruntime`, `mediapy`, and `numpy`.
The MHS coordination degrades gracefully without the `mhs` SDK on the `loopback` path (the shim in
`examples/isaac_bed_making/mhs_sidecar.py` provides pass-through decorators); `broker` mode needs the
`mhs` SDK installed.

---

# The training task (how the policy was made)

The reach policy is a Unitree G1 RL task that trains the robot to **reach onto a bed and hold the draw
of a sheet while staying balanced** — the loco-manipulation-under-load skill — implemented entirely on
mjlab / MuJoCo-Warp. An earlier reach policy, trained on short free-space reaches with **no payload**,
was fragile: asked to hold a target while a sheet pulled on its hand, it reached by leaning and stepped
backward to recover, drifting off its stance and dragging the cover off the bed. Every custom term
here targets that failure, following **FALCON** ([arXiv 2505.06776](https://arxiv.org/abs/2505.06776),
validated on a real G1) and **HuB** ([arXiv 2505.07294](https://arxiv.org/abs/2505.07294)):

| term (`mdp/bed_reach_terms.py`) | what it fixes |
| --- | --- |
| **`ReachCommand`** (pelvis frame) | a world-frame target lets the robot cut its own error by stepping back — the drift feedback loop. Commanding in the pelvis frame removes it. |
| **`stance_root_xy`** (−) | penalizes the pelvis leaving the midpoint of the feet — the direct anti-lean-and-step term. |
| **`com_over_support`** (+) | keeps the centre of mass over the support polygon. |
| **`apply_reach_force`** + **`force_curriculum`** | applies a survival-gated external wrench on the reaching wrist (`xfrc_applied`), so the policy trains under the very drag it must resist. The cloth is a deploy-time object; the *load* is learned here. |
| **`drifted`** termination | ends the episode once the base walks past a drift radius — the cheapest, strongest anti-drift signal. |

The task **extends mjlab's proven G1 velocity task** (which already gives balance + locomotion) rather
than rebuilding it, so the robot starts from a working stand-and-walk base and only has to learn the
reach-and-hold on top.

## Train it

```bash
# mjlab auto-imports mjlab.tasks.*; expose this package there with a symlink (no site-packages edits)
SP=$(mjlab-venv/bin/python -c "import mjlab, os; print(os.path.dirname(mjlab.__file__))")
ln -sfn "$PWD" "$SP/tasks/bed_reach"

mjlab-venv/bin/list-envs | grep BedReach          # -> Mjlab-BedReach-Unitree-G1

WANDB_MODE=offline MUJOCO_GL=egl mjlab-venv/bin/train \
    Mjlab-BedReach-Unitree-G1 \
    --env.scene.num-envs 1024 --agent.max-iterations 2000 \
    --agent.num-steps-per-env 24 --agent.logger tensorboard --video False
```

Watch `reach_error` (should trend down) and `Curriculum/force_curriculum` (ramps up once the policy
survives long enough under load). A converged run exports `policy.onnx` under
`logs/rsl_rl/g1_bed_reach/<run>/`, loadable with onnxruntime (obs[103] → action[29]). The shipped
`policy/bed_reach_g1.onnx` is a 3000-iteration run: it trained under the full ±25–30 N end-effector
drag with a mean episode length near the cap, i.e. it survives full episodes under load instead of
leaning-and-stepping-off.

## Pins (aarch64 / GB10, all open packages)

`mjlab==1.5.2`, `mujoco-warp==3.10.0.2`, `warp-lang==1.15.0`, `rsl-rl-lib==5.4.0`,
`torch==2.13.0+cu130`, Python 3.12.

## Files

- `demo/` — the two-robot bed-making demo (see the table above).
- `bed_reach_env_cfg.py` — the training env: extends the flat G1 velocity task with the reach command,
  FALCON rewards, drift termination, and the force curriculum.
- `mdp/bed_reach_terms.py` — the custom MDP terms above.
- `rl_cfg.py` — PPO runner config (G1 velocity hyperparameters).
- `policy/bed_reach_g1.onnx` — the trained policy (obs[103] → action[29]).
- `media/` — `bed_making.mp4`, representative `frames/`, and the captured `mhs_trace.json`.
- `DESIGN.md` — the full FALCON-based training design and roadmap.
