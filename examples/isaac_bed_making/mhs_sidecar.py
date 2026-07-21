#!/usr/bin/env python3
# Copyright (c) 2024-2026, Strands Robots contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""MHS shim for the Isaac Sim bed-making swarm (examples/isaac_bed_making).

The two-humanoid swarm driver (``swarm_driver.py``) began on Arm Device Connect
(``device_connect_edge``). That driver framework was absorbed into the Model Hardware
Standard (MHS) Python SDK (``mhs-python-sdk``, import root ``mhs``), so this file lets
``swarm_driver.py`` run on MHS by importing its device surface from here instead of from
``device_connect_edge``.

The migration is almost a straight rename — MHS kept Device Connect's driver model:

* ``DeviceDriver``          -> ``mhs.MhsDriver``
* ``@on(event_name=...)``   -> ``mhs.on`` verbatim (same ``_sub_event_name`` marker, same
                               ``handler(device_id, event_name, payload)`` dispatch)
* ``@periodic(interval=..., wait_for_completion=...)`` -> ``mhs.periodic`` verbatim

Only two decorators differ, and only in one keyword: Device Connect's ``@rpc`` / ``@emit``
took a ``labels={...}`` dict of advisory dashboard hints; MHS's take ``tags=[...]``. The two
wrappers below accept ``labels=`` and forward it as ``tags`` so ``swarm_driver.py`` is
unchanged apart from this import.

``DeviceIdentity`` / ``DeviceStatus`` become plain data holders: MHS carries identity as a
dict on the manifest rather than as objects, so these keep the driver's ``identity`` /
``status`` properties constructing exactly as before.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# ── SDK import-shim ───────────────────────────────────────────────────────────────────────────
try:
    from mhs import MhsDriver as DeviceDriver, on, periodic
    from mhs import rpc as _mhs_rpc, emit as _mhs_emit

    HAVE_MHS = True
except Exception:  # mhs-python-sdk not installed (e.g. offline unit tests of the @on handlers)
    HAVE_MHS = False

    class DeviceDriver:  # type: ignore  minimal stand-in so the driver subclass imports
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def _passthrough(*dargs: Any, **dkwargs: Any):
        def _decorate(fn: Callable) -> Callable:
            return fn
        return _decorate

    _mhs_rpc = _mhs_emit = _passthrough  # type: ignore

    def on(device_id: Optional[str] = None, device_type: Optional[str] = None,  # type: ignore
           event_name: Optional[str] = None):
        def _decorate(fn: Callable) -> Callable:
            fn._sub_event_name = event_name  # the loopback bus reads this marker
            return fn
        return _decorate

    def periodic(*dargs: Any, **dkwargs: Any):  # type: ignore
        return _passthrough()


# ── @rpc / @emit: accept Device Connect's labels= and forward as MHS tags= ──────────────────────
def _labels_to_tags(labels: Optional[Dict[str, Any]]) -> List[str]:
    """Render a Device Connect ``labels`` dict as MHS ``tags`` (``"key:value"`` strings)."""
    return [f"{k}:{v}" for k, v in (labels or {}).items()]


def rpc(name: Optional[str] = None, *, labels: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None, estop: bool = False,
        description: Optional[str] = None) -> Callable:
    """MHS ``@rpc`` that also accepts Device Connect's ``labels=`` (mapped to ``tags``)."""
    merged = list(tags or []) + _labels_to_tags(labels)
    return _mhs_rpc(name, description=description, estop=estop, tags=(merged or None))


def emit(name: Optional[str] = None, *, labels: Optional[Dict[str, Any]] = None,
         description: Optional[str] = None) -> Callable:
    """MHS ``@emit`` that also accepts Device Connect's ``labels=`` (advisory; dropped)."""
    return _mhs_emit(name, description=description)


# ── identity / status shims ─────────────────────────────────────────────────────────────────────
class DeviceIdentity:
    """What a device advertises about itself (device_type, model, manufacturer, description, ...)."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class DeviceStatus:
    """A device's runtime status / availability."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)
