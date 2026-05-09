"""Compatibility imports for Device Connect package naming.

The Arm Device Connect repository currently ships the edge runtime as
``device_connect_edge``.  Older strands-robots code and tests used the
``device_connect_sdk`` import path, so keep both working while the ecosystem
settles on one name.
"""

try:
    from device_connect_edge import DeviceRuntime
    from device_connect_edge.drivers import DeviceDriver, emit, on, periodic, rpc
    from device_connect_edge.types import DeviceIdentity, DeviceStatus
except ModuleNotFoundError:
    from device_connect_sdk import DeviceRuntime
    from device_connect_sdk.drivers import DeviceDriver, emit, on, periodic, rpc
    from device_connect_sdk.types import DeviceIdentity, DeviceStatus

__all__ = [
    "DeviceRuntime",
    "DeviceDriver",
    "DeviceIdentity",
    "DeviceStatus",
    "emit",
    "on",
    "periodic",
    "rpc",
]
