# Two Unitree G1s make a bed in NVIDIA Isaac Sim — coordinated over Arm Device Connect

Two **Unitree G1 humanoids** (with **Inspire 5‑finger hands**) autonomously make a
bed in **NVIDIA Isaac Sim / Isaac Lab**, coordinating as **equal peers over Arm
Device Connect** — instead of "head‑nodding," they ask each other for help and
offer help over a real Device Connect swarm.

This is a self‑contained example: it does **not** modify the `strands_robots`
product package or Arm's Device Connect. It reuses the swarm driver from
[`examples/unitree_g1_bed_making_g1_driver.py`](../unitree_g1_bed_making_g1_driver.py).

![Two G1s drape the sheet over the bed](media/02_robots_drape.png)

| Sheet drapes onto the bed | The bed, made |
| --- | --- |
| ![sheet settled](media/01_sheet_settled.png) | ![bed made](media/03_bed_made.png) |

A full headless run renders **107 frames → [`media/isaac_bed_making.mp4`](media/isaac_bed_making.mp4)**.

## What it shows

* **Real, physically‑simulated cloth.** A thick PhysX particle‑cloth "duvet"
  drapes over the bed under gravity and is gripped by the robots' hands — not a
  scripted animation.
* **Friction grasping, no cheating.** The hands have a high‑friction "rubberized"
  material; the cloth is gripped by **contact friction**, with **no kinematic
  attachment / pinning**. Cloth‑on‑bed friction is tuned independently so the
  sheet drapes and grips without sliding off.
* **Genuine arm control.** Each arm is driven by Isaac Lab's world‑frame
  differential IK (not a pre‑baked joint trajectory).
* **Equal‑peer swarm coordination over Device Connect.** Both G1s pursue the goal
  state *"the bed is made"*; they claim work, emit events, and ask for / offer
  help. With `--broker` both peers register live on the Device Connect dashboard
  with callable functions and an event stream.

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
| `--render` | Capture frames and encode an mp4 into `artifacts/isaac_bed_making/`. |

## How it works (and two gotchas worth knowing)

* **Cloth = a *quad* grid, not triangles.** Isaac's auto particle‑cloth turns
  every mesh edge into a stiff stretch spring, so a *triangulated* grid makes the
  cell diagonals inextensible and the sheet locks into a rigid plate. With
  **quads**, the diagonal becomes a soft *shear* spring and the cloth drapes.
  Springs are kept elastic for the same reason.
* **The GPU pipeline doesn't sync cloth deformation to the renderer.** With
  Fabric on (or off) on the GPU PhysX pipeline, deformed particle positions are
  not written back to the USD mesh, so a headless camera renders the *flat,
  authored* mesh while the cloth actually drapes in the backend. The demo runs
  with `use_fabric=False` and reads the live positions from a **PhysX tensor
  cloth‑view**, blitting them into the visual mesh each frame.

## Files

| File | Role |
| --- | --- |
| `demo.py` | Entry point: builds the scene, runs the DC‑coordinated bed‑making sequence, renders. |
| `scene.py` | Scene geometry — two planted G1s + bed + camera + sheet parameters. |
| `cloth.py` | PhysX particle‑cloth bedsheet (quad grid, elastic, thick) + tensor‑view read/blit for rendering. |
| `manipulation.py` | World‑frame differential‑IK arm driver + `apply_hand_friction` (rubberized hands). |
| `coordination.py` | In‑process Device Connect swarm (loopback / real broker), wrapping the shared swarm driver. |
| `behavior.py` | Per‑robot autonomous decision state machine (scaffolding for emergent ordering). |
| `coverage.py` | "Good‑enough" coverage metric scaffolding (to be reworked for head‑mounted cameras). |

## Status

This is a **working baseline**: the cloth physics, Inspire hands, friction
grasping, Device Connect swarm and rendering all work end‑to‑end. The *manipulation
choreography is intentionally minimal* — the next milestone replaces it with
**learned manipulation from the real Unitree
[UnifoLM‑WBT bed‑making dataset](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Make_The_Bed)**.
See the issue #2 continuation comment for the full roadmap.
