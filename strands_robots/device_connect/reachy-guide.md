# Reachy Mini — Cloud Zenoh Mesh Guide

Connect a Reachy Mini Lite (USB) to the AWS-hosted Zenoh mesh via Device Connect so any agent on the mesh can discover and control it.

## Prerequisites

- Reachy Mini Lite plugged in via USB
- Python 3.12+, `uv` installed

## 1. Start the Reachy Daemon

The daemon bridges USB serial to a local REST/WebSocket API on port 9002.

```bash
# From the repo root (uses uv inline script, auto-installs reachy-mini)
uv run --python 3.13 --with reachy-mini start_reachy_daemon.py```

Expected output:

```
Starting reachy-mini-daemon on /dev/cu.usbmodemXXXX (API port 9002)
```

Leave this running in a separate terminal.

## 2. Clone and Set Up the Strands Robots SDK

```bash
git clone --branch feat/device-connect-integration-draft \
  https://github.com/atsyplikhin/robots.git
cd robots
./strands_robots/device_connect/setup.sh
source .venv/bin/activate
uv pip install websockets  # required dependency not yet in setup
```

## 3. Connect to the Cloud Zenoh Mesh

Set environment variables (every terminal that talks to the mesh):

```bash
export ZENOH_CONNECT=tcp/zenoh-nlb-2cb0b84309701828.elb.us-east-1.amazonaws.com:7447
export ZENOH_MODE=client
export DEVICE_CONNECT_ALLOW_INSECURE=true
```

Start the Device Connect runtime (bridges the local daemon to the cloud mesh):

```bash
python -c "
import asyncio, os
from strands_robots.device_connect import ReachyMiniDriver
from device_connect_sdk import DeviceRuntime

driver = ReachyMiniDriver(host='localhost', api_port=9002)
runtime = DeviceRuntime(
    driver=driver,
    device_id='reachy-mini-1',
    messaging_urls=[os.environ['ZENOH_CONNECT']],
    allow_insecure=True,
)
asyncio.run(runtime.run())
"
```

Expected output:

```
INFO - Using ZENOH messaging backend
INFO - Connected to ZENOH broker: ['tcp/zenoh-nlb-...amazonaws.com:7447']
INFO - Driver connected: reachy_mini
INFO - Device registered: registration_id=...
INFO - Subscribed to commands on device-connect.default.reachy-mini-1.cmd
```

Leave this running. The robot is now on the mesh as `reachy-mini-1`.

## 4. Invoke Commands from Any Mesh Client

From another terminal (with the same env vars and venv activated):

```bash
source robots/.venv/bin/activate
export ZENOH_CONNECT=tcp/zenoh-nlb-2cb0b84309701828.elb.us-east-1.amazonaws.com:7447
export ZENOH_MODE=client
export DEVICE_CONNECT_ALLOW_INSECURE=true
```

### Move antennas

```python
from device_connect_agent_tools import connect, invoke_device
connect()
r = invoke_device('reachy-mini-1', 'antennas', {'left': 30, 'right': -30})
print('RESULT:', r)
# {'success': True, 'result': {'status': 'success', 'left': 30, 'right': -30}}
```

### Look (head pose)

```python
invoke_device('reachy-mini-1', 'look', {'pitch': -15, 'yaw': 15, 'roll': 0})
# pitch: up/down (negative = look up), yaw: left/right, roll: tilt
```

### Expressions

```python
invoke_device('reachy-mini-1', 'nod')    # yes gesture
invoke_device('reachy-mini-1', 'shake')  # no gesture
invoke_device('reachy-mini-1', 'happy')  # antenna wiggle
```

### Sequence example

```python
from device_connect_agent_tools import connect, invoke_device
import time

connect()
invoke_device('reachy-mini-1', 'look', {'pitch': -10, 'yaw': 15})
time.sleep(1)
invoke_device('reachy-mini-1', 'nod')
time.sleep(2)
invoke_device('reachy-mini-1', 'look', {'pitch': 0, 'yaw': 0, 'roll': 0})
print('Done!')
```

## Available RPCs

| RPC | Parameters | Description |
|-----|-----------|-------------|
| `look` | `pitch`, `roll`, `yaw`, `x`, `y`, `z` | Set head pose (degrees / mm) |
| `antennas` | `left`, `right` | Set antenna angles (degrees) |
| `body` | `yaw` | Set body yaw (degrees) |
| `nod` | — | Yes gesture |
| `shake` | — | No gesture |
| `happy` | — | Antenna wiggle |
| `getJoints` | — | Current joint positions |
| `getImu` | — | IMU sensor data |
| `enableMotors` | `motor_ids` (optional) | Torque on |
| `disableMotors` | `motor_ids` (optional) | Torque off |
| `wakeUp` | — | Enable motors + wake animation |
| `sleep` | — | Sleep animation + disable motors |
| `stopMotion` | — | Stop all motion |
| `getDaemonStatus` | — | Daemon status and motor state |
| `playMove` | `move_name`, `library` | Play recorded move (`emotions` or `dance`) |
| `listMoves` | `library` | List available moves |

## Process Summary

You need **three terminals**:

| Terminal | Command | Purpose |
|----------|---------|---------|
| 1 | `python start_reachy_daemon.py` | USB serial daemon (port 9002) |
| 2 | Device Connect runtime script (step 3) | Bridges daemon to cloud Zenoh mesh |
| 3 | `invoke_device(...)` calls (step 4) | Send commands to the robot |

## Troubleshooting

- **`No USB serial device found`** — Check that Reachy is plugged in (`ls /dev/cu.usbmodem*`)
- **`ModuleNotFoundError: websockets`** — Run `uv pip install websockets`
- **`KeyboardInterrupt` on import** — The `cv2` import can hang; make sure the daemon terminal is separate from the Device Connect terminal
- **Connection refused on port 9002** — Daemon not running; start it first (step 1)
