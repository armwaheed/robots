"""Run Reachy Mini Device Connect against a NATS broker.

This sidecar is intended for cases where the robot already has an existing
local control/demo stack. It talks to the Reachy daemon over REST/WebSocket and
bridges structured RPCs into Device Connect over NATS.
"""

import asyncio
import os

from device_connect_sdk import DeviceRuntime

try:
    from strands_robots.device_connect.reachy_mini_driver import ReachyMiniDriver
except ImportError:
    from reachy_mini_driver import ReachyMiniDriver  # type: ignore


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


async def main() -> None:
    nats_url = _env("NATS_URL", "nats://localhost:4222")
    nats_credentials_file = os.environ.get("NATS_CREDENTIALS_FILE", "").strip() or None
    device_id = _env("DEVICE_ID", "reachy-mini-1")
    host = _env("REACHY_HOST", "127.0.0.1")
    prefix = _env("REACHY_PREFIX", "reachy_mini")
    transport_mode = _env("REACHY_TRANSPORT_MODE", "websocket")
    api_port = int(_env("REACHY_PORT", "8000"))

    driver = ReachyMiniDriver(
        host=host,
        prefix=prefix,
        api_port=api_port,
        transport_mode=transport_mode,
    )
    runtime_kwargs = {
        "driver": driver,
        "device_id": device_id,
        "messaging_backend": "nats",
        "messaging_urls": [nats_url],
        "allow_insecure": True,
    }
    if nats_credentials_file:
        runtime_kwargs["nats_credentials_file"] = nats_credentials_file
    runtime = DeviceRuntime(**runtime_kwargs)
    await runtime.run()


if __name__ == "__main__":
    asyncio.run(main())
