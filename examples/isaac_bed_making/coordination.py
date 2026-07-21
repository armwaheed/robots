"""In-process MHS swarm coordination for the Isaac Sim demo.

Wraps :class:`BedMakingG1Driver` from ``swarm_driver.py`` — a self-contained copy of the
equal-peer swarm driver, bundled here so the Isaac demo has **no dependency on the MuJoCo
demo** — so the Isaac Sim main loop, which is synchronous, can drive two G1 peers that
coordinate over the Model Hardware Standard (MHS) on a background asyncio thread.
Two transports:

* ``loopback`` — no broker; an in-process bus fans events between peers. Offline.
* ``broker``   — both peers mount on an MHS NATS fabric, so they appear to any
  fleet client on that fabric with callable procedures and a live event stream.
  Credentials, if the broker needs them, come from the SDK's environment
  conventions (``NATS_CREDENTIALS_FILE``, or ``NATS_JWT`` + ``NATS_NKEY_SEED``).

The sim loop calls :meth:`SwarmCoordinator.invoke` (blocking) and
:meth:`SwarmCoordinator.snapshot`; all driver/runtime work happens on the
coordinator's own event loop thread.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.isaac_bed_making.mhs_trace import (  # noqa: E402
    BROADCAST,
    EVENT_PUBLISH,
    RPC_CALL,
    RPC_RETURN,
    MessageTrace,
)
from examples.isaac_bed_making.swarm_driver import (  # noqa: E402
    GOAL_STATE,
    SHEET_TO_BED,
    BedMakingG1Driver,
)

DEFAULT_NATS_URL = "nats://localhost:4222"

# device_id the demo's own fleet-plane client mounts under. It is a THIRD participant on the
# fabric alongside the two robots — the orchestrator/console role — so procedure calls are
# addressed to a device over the broker rather than made as local method calls.
ORCHESTRATOR_ID = "bed-making-orchestrator"


class _LoopbackBus:
    """Fan every emitted event out to the other peers' ``@on`` handlers, in process.

    Stands in for the broker on the offline path: same driver surface, same payloads, same handlers
    — the bytes just never leave the process. It is NOT an MHS transport, and the trace labels it
    ``loopback`` so a reader is never misled into thinking a broker was involved."""

    def __init__(self, trace: MessageTrace) -> None:
        self._subs: Dict[str, List[Tuple[str, Any]]] = {}
        self._trace = trace

    def register(self, driver: BedMakingG1Driver) -> None:
        for _, method in inspect.getmembers(driver, predicate=inspect.ismethod):
            event_name = getattr(method, "_sub_event_name", None)
            if event_name:
                self._subs.setdefault(event_name, []).append((driver.device_id, method))
        driver.set_emit_sink(self._make_sink(driver.device_id))

    def _make_sink(self, source_id: str):
        async def _sink(event_name: str, payload: dict) -> None:
            self._trace.record(EVENT_PUBLISH, src=source_id, dst=BROADCAST,
                               name=event_name, payload=payload)
            for sub_id, handler in self._subs.get(event_name, []):
                if sub_id == source_id:
                    continue
                try:
                    await handler(source_id, event_name, payload)
                except Exception:
                    pass

        return _sink


class SwarmCoordinator:
    """Runs two G1 MHS peers on a background event loop."""

    def __init__(self, mode: str = "loopback", nats_url: str = DEFAULT_NATS_URL) -> None:
        self.mode = mode
        self.nats_url = nats_url
        self.tenant = os.environ.get("MHS_TENANT", "_")
        self.goal_state = GOAL_STATE
        self.sheet_to_bed = SHEET_TO_BED
        # Both peers run the WIRE profile: transport + application API, no buffer or format layer.
        # They hold no shared-memory device state, so the direct plane never comes into it.
        self.trace = MessageTrace(
            plane="fleet" if mode == "broker" else "in-process",
            transport="nats" if mode == "broker" else "loopback",
            profile="WIRE")
        self.peers: List[BedMakingG1Driver] = []
        self._mounted: List[Any] = []
        self._client: Optional[Any] = None
        self._fleet: Optional[Any] = None       # fleet-plane client; broker mode only
        self._fleet_stack: Optional[Any] = None  # keeps connect_fleet's context open
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mhs-swarm")
        self._ready = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self, device_ids: Optional[List[str]] = None) -> List[str]:
        """Start the loop thread and register the two peers. Returns device ids."""
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._async_start(device_ids), self._loop)
        ids = fut.result(timeout=30)
        self._ready.set()
        return ids

    async def _async_start(self, device_ids: Optional[List[str]]) -> List[str]:
        ids = device_ids or ["beta-unitree-g1-humanoid-0", "beta-unitree-g1-humanoid-1"]
        if self.mode == "broker":
            import mhs.transport_layer.nats  # noqa: F401  registers the "nats" backend
            from mhs.mhs_api.mount import mount
            from mhs.mhs_api.security_config import SecurityConfig
            from mhs.transport_layer.auth import load_credentials_from_env
            from mhs.transport_layer.messaging import create_client

            # The SDK is secure-by-default on remote transports and refuses to
            # mount without a commissioned Security State bundle. This is a sim
            # demo on a trusted lab fabric, so run unsecured unless the operator
            # states a posture via MHS_SECURITY_MODE (then defer to the env).
            security = None if os.environ.get("MHS_SECURITY_MODE") else SecurityConfig.off()

            # One broker connection carries both peers (the SDK's multi-device
            # pattern): mount() registers each driver's @rpc procedures, wires
            # @emit to the fabric, and subscribes its @on handlers.
            self._client = create_client("nats")
            await self._client.connect(
                [self.nats_url], credentials=load_credentials_from_env()
            )
            for device_id in ids:
                drv = BedMakingG1Driver(device_id=device_id, role="peer", auto_offer=True)
                drv.set_trace(self.trace)
                md = await mount(
                    driver=drv, device_id=device_id,
                    messaging=self._client, tenant=self.tenant,
                    security=security,
                )
                self.peers.append(drv)
                self._mounted.append(md)
            # A third participant on the fabric: a fleet-plane client, exactly what an operator
            # console or an orchestrating agent would hold. Routing the demo's procedure calls
            # through it means they are addressed to a device_id and carried by the broker — the
            # same path any other client would take — instead of being local method calls that
            # merely look like MHS.
            from contextlib import AsyncExitStack

            from mhs.mhs_api.run import connect_fleet
            # discovery="presence" finds devices from their own presence announcements on the
            # fabric. The default ("auto") prefers a registry service, which a bare broker does not
            # run — asking for it just times out against `mhs._.discovery` with no responders.
            self._fleet_stack = AsyncExitStack()
            self._fleet = await self._fleet_stack.enter_async_context(
                connect_fleet(broker=self.nats_url, tenant=self.tenant,
                              device_id=ORCHESTRATOR_ID, discovery="presence",
                              allow_insecure=security is not None))
            found = sorted(r["device_id"] for r in (await self._fleet.discover("*"))["results"])
            self.trace.record(RPC_CALL, src=ORCHESTRATOR_ID, dst=BROADCAST, name="discover",
                              payload={"selector": "*"},
                              effect=f"found {len(found)} peers on the fabric")
            print(f"[mhs] fleet-plane discovery found: {found}", flush=True)
        else:
            bus = _LoopbackBus(self.trace)
            for device_id in ids:
                drv = BedMakingG1Driver(device_id=device_id, role="peer", auto_offer=True)
                drv.set_trace(self.trace)
                bus.register(drv)
                self.peers.append(drv)
        return [p.device_id for p in self.peers]

    def stop(self) -> None:
        async def _shutdown() -> None:
            if self._fleet_stack is not None:
                try:
                    await self._fleet_stack.aclose()
                except Exception:
                    pass
            if self._mounted:
                from mhs.mhs_api.mount import unmount

                for md in self._mounted:
                    try:
                        await unmount(md)
                    except Exception:
                        pass
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:
                    pass

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ── synchronous bridge for the sim loop ───────────────────────────────
    def invoke(self, peer_idx: int, fn_name: str, timeout: float = 15.0,
               effect: str = "", **params) -> dict:
        """Call a peer's MHS ``@rpc`` procedure and return its result.

        In ``broker`` mode the call goes out over the FLEET PLANE: it is addressed to the device by
        id and travels through the broker, so it is the same path any other fleet client or
        orchestrator on that fabric would use. In ``loopback`` mode there is no fabric, so the
        procedure body is awaited in process; the trace labels those hops ``loopback`` rather than
        pretending a broker was involved.

        ``effect`` records what the simulator does as a result of this call, so the message-flow
        trace shows cause and consequence rather than just traffic."""
        device_id = self.peers[peer_idx].device_id
        call = self.trace.record(RPC_CALL, src="sim", dst=device_id, name=fn_name,
                                 payload=params, effect=effect)

        async def _call() -> dict:
            if self._fleet is not None:
                result = await self._fleet.invoke(device_id, fn_name, **params)
            else:
                method = getattr(self.peers[peer_idx], fn_name)
                result = await method(**params)
            return result if isinstance(result, dict) else {"result": result}

        result = asyncio.run_coroutine_threadsafe(_call(), self._loop).result(timeout=timeout)
        self.trace.record(RPC_RETURN, src=device_id, dst="sim", name=fn_name,
                          payload={"ok": result.get("ok", True)},
                          effect=f"in reply to #{call.seq}" if call else "")
        return result

    def snapshot(self) -> List[dict]:
        return [p.agent.snapshot() for p in self.peers]

    def event_history(self, peer_idx: int = 0, limit: int = 50) -> List[dict]:
        return self.peers[peer_idx].agent.event_history(limit=limit)

    def help_history(self, peer_idx: int = 0, limit: int = 50) -> List[dict]:
        return self.peers[peer_idx].agent.help_history(limit=limit)
