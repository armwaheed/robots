# Reachy Mini — Device Connect NATS Guide

Connect a Reachy Mini on a Raspberry Pi to a remote Device Connect NATS broker.

This guide covers the working setup we validated on the Pi at `10.118.156.11`:

- the existing Reachy browser-control demo continues to own the local robot daemon
- a separate Device Connect sidecar runs alongside it
- the sidecar bridges Reachy RPCs into Device Connect over NATS

## Where this code belongs

The Pi-side Device Connect bridge belongs in the `robots` repo, not in
`conference-room`.

- `conference-room` owns the app-specific Reachy browser demo:
  `apps/reachy_mini_browser_control/...`
- `robots` owns the reusable Strands + Device Connect Reachy integration:
  `strands_robots/device_connect/reachy_mini_driver.py`
  `strands_robots/device_connect/reachy_transport.py`
  `strands_robots/device_connect/run_reachy_nats.py`

If you need to document how the local Pi demo starts Reachy, keep that in the
`conference-room` README. If you need to document how Reachy registers into
Device Connect, keep that here.

## Two valid deployment patterns

### Option A: Reuse an existing local Reachy daemon

Use this when the Pi is already running the browser-control demo or another
local Reachy app that exposes the daemon on `127.0.0.1:8000`.

This is the safest option if you do not want to disturb an existing demo.

### Option B: Start a dedicated Reachy daemon from this repo

Use this when no local Reachy daemon is already running.

That path is what Sourav described originally:

```bash
source .venv/bin/activate
python strands_robots/device_connect/start_reachy_daemon.py
```

It starts a local daemon on port `9002` and is fine for a standalone setup.

## Recommended setup on a Pi with an existing Reachy demo

### 1. Prepare the `robots` repo

```bash
git clone --branch feat/device-connect-integration-draft \
  https://github.com/atsyplikhin/robots.git
cd robots
./strands_robots/device_connect/setup.sh
source .venv/bin/activate
uv pip install websockets
```

### 2. Verify the local Reachy daemon

If the Pi already runs `reachy_mini_browser_control`, verify that the local
daemon is healthy:

```bash
curl http://127.0.0.1:8000/api/daemon/status
```

Expected output includes:

```json
{
  "state": "running",
  "wireless_version": false
}
```

If nothing is listening on port `8000`, use **Option B** above to start a
dedicated daemon on `9002` instead.

### 3. Start the Device Connect sidecar

If you already have a daemon on `127.0.0.1:8000`, run the sidecar against that
daemon:

```bash
cd /path/to/robots
source .venv/bin/activate

export MESSAGING_BACKEND=nats
export NATS_URL='nats://nats-nlb-52ebd3849ae4cb02.elb.us-east-1.amazonaws.com:4222'
export MESSAGING_URLS="$NATS_URL"
export DEVICE_CONNECT_ALLOW_INSECURE=true

python strands_robots/device_connect/run_reachy_nats.py
```

By default, the runner uses:

- `DEVICE_ID=reachy-mini-1`
- `REACHY_HOST=127.0.0.1`
- `REACHY_PORT=8000`
- `REACHY_TRANSPORT_MODE=websocket`

You can override them if needed:

```bash
export DEVICE_ID='reachy-mini-1'
export REACHY_HOST='127.0.0.1'
export REACHY_PORT='8000'
export REACHY_TRANSPORT_MODE='websocket'
python strands_robots/device_connect/run_reachy_nats.py
```

Expected successful logs:

```text
INFO - Using NATS messaging backend
INFO - Connected to NATS broker: ['nats://nats-nlb-52ebd3849ae4cb02.elb.us-east-1.amazonaws.com:4222']
INFO - Driver connected: reachy_mini
INFO - Device registered: ...
INFO - Subscribed to commands on device-connect.default.reachy-mini-1.cmd
```

### 4. Sourav's original direct-runtime form

If you prefer not to use the helper runner, this is the equivalent direct form.

When reusing an existing daemon on the Pi, change the original `api_port=9002`
example to `api_port=8000`:

```bash
cd /path/to/robots
source .venv/bin/activate
export MESSAGING_BACKEND=nats
export NATS_URL='nats://nats-nlb-52ebd3849ae4cb02.elb.us-east-1.amazonaws.com:4222'
export MESSAGING_URLS="$NATS_URL"
export DEVICE_CONNECT_ALLOW_INSECURE=true

python -c "
import asyncio, os
from strands_robots.device_connect import ReachyMiniDriver
from device_connect_sdk import DeviceRuntime

driver = ReachyMiniDriver(
    host='localhost',
    api_port=8000,
    transport_mode='websocket',
)
runtime = DeviceRuntime(
    driver=driver,
    device_id='reachy-mini-1',
    messaging_urls=[os.environ['NATS_URL']],
    messaging_backend='nats',
    allow_insecure=True,
)
asyncio.run(runtime.run())
"
```

Use `api_port=9002` only if you started the dedicated daemon from
`start_reachy_daemon.py`.

## Verify from a client

From another terminal:

```bash
cd /path/to/robots
source .venv/bin/activate
export MESSAGING_BACKEND=nats
export NATS_URL='nats://nats-nlb-52ebd3849ae4cb02.elb.us-east-1.amazonaws.com:4222'
export MESSAGING_URLS="$NATS_URL"
export DEVICE_CONNECT_ALLOW_INSECURE=true
```

Discover devices:

```bash
python -c "
from device_connect_agent_tools import connect, discover_devices
from device_connect_agent_tools.connection import disconnect

connect()
try:
    print(discover_devices())
finally:
    disconnect()
"
```

Invoke RPCs:

```bash
python -c "
from device_connect_agent_tools import connect, invoke_device
from device_connect_agent_tools.connection import disconnect

connect()
try:
    print(invoke_device('reachy-mini-1', 'look', {'pitch': -15, 'yaw': 15}))
    print(invoke_device('reachy-mini-1', 'nod'))
finally:
    disconnect()
"
```

## File inventory

These are the reusable Reachy Device Connect files in this repo:

- `strands_robots/device_connect/reachy_mini_driver.py`
  Reachy-specific Device Connect RPC driver
- `strands_robots/device_connect/reachy_transport.py`
  REST/WebSocket and Zenoh transport helpers
- `strands_robots/device_connect/run_reachy_nats.py`
  Small sidecar runner for NATS registration
- `strands_robots/device_connect/start_reachy_daemon.py`
  Helper for standalone daemon startup when you are not reusing an existing Pi
  daemon

## Troubleshooting

- `Authorization Violation`
  The NATS credentials are valid enough to authenticate, but not valid for the
  broker you are pointing at.
- `permissions violation for publish to "device-connect.default.registry"`
  The credentials authenticate, but the broker ACLs do not allow Device Connect
  registration.
- `Connection refused` on port `8000`
  No local Reachy daemon is running there. Either start the browser-control demo
  or run `start_reachy_daemon.py` and point the sidecar at `9002`.
- `No USB serial device found`
  Only relevant when starting a dedicated daemon directly from this repo.
- `ModuleNotFoundError: websockets`
  Run `uv pip install websockets`.
