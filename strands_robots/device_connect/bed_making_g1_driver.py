"""Device Connect driver for a Unitree G1 swarm peer in the bed-making demo.

Both G1s register with Device Connect as **equal peers** and coordinate
entirely through DC events — there are no master/worker calls. When one
peer claims, picks up, places, or releases a sheet corner, it emits a DC
event; the other peer subscribes to the same event and updates its local
view of the swarm so it can react (pick a different corner, walk over to
support, etc.).

Schema:

- ``claimCorner(corner)`` — "I intend to grab this corner next." Emitted
  before walking to the pickup station so the peer can yield or pick a
  different corner before any physical action happens.
- ``cornerHeld(corner, x, y, z)`` — "I am holding this corner right now;
  here is its current world position." Emitted continuously while held.
- ``cornerPlaced(corner, bed_corner)`` — "I have placed this corner on the
  bed at ``bed_corner``."
- ``cornerReleased(corner)`` — "I have let go." The corner is now free.
- ``askAssist(corner, reason)`` — "I need help with this corner."
- ``state(status, role, holding, placed)`` — periodic heartbeat with the
  whole swarm-relevant state.

The driver subscribes to all of the above for **other** device ids. When
an event arrives, the driver mutates its bound :class:`SwarmAgent` (passed
in as ``agent``) under a lock. The simulation main thread reads the agent
state and applies it to MuJoCo.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from strands_robots.device_connect._compat import (
    DeviceDriver,
    DeviceIdentity,
    DeviceStatus,
    emit,
    on,
    periodic,
    rpc,
)

logger = logging.getLogger(__name__)


@dataclass
class PeerView:
    """What this agent knows about another swarm peer."""

    device_id: str
    status: str = "online"
    holding_corner: Optional[str] = None
    holding_position: Optional[Tuple[float, float, float]] = None
    last_event: Optional[str] = None


@dataclass
class SwarmState:
    """Mutable, thread-safe-ish view of one swarm peer's own state."""

    device_id: str
    role: str = "peer"
    status: str = "online"
    held_corner: Optional[str] = None
    placed_corners: set = field(default_factory=set)
    peer_claims: Dict[str, str] = field(default_factory=dict)
    peers: Dict[str, PeerView] = field(default_factory=dict)
    last_event: Optional[str] = None
    rpc_counts: Dict[str, int] = field(default_factory=dict)


class SwarmAgent:
    """Per-robot decision state shared between the DC driver and the sim loop.

    The sim main thread reads ``state`` to decide MuJoCo actions. The DC
    driver mutates ``state`` from the asyncio thread when peer events arrive.
    All mutations are guarded by ``lock``. Both threads should hold the lock
    when reading/writing nested data.
    """

    def __init__(self, *, device_id: str, role: str = "peer") -> None:
        self.lock = threading.Lock()
        self.state = SwarmState(device_id=device_id, role=role)

    # Mutators called from the DC driver's @on handlers (asyncio thread).

    def note_peer_claim(self, peer_id: str, corner: str) -> None:
        with self.lock:
            self.state.peer_claims[peer_id] = corner
            self._peer(peer_id).status = "claiming"
            self._peer(peer_id).last_event = f"claim:{corner}"

    def note_peer_held(self, peer_id: str, corner: str, pos: Tuple[float, float, float]) -> None:
        with self.lock:
            view = self._peer(peer_id)
            view.holding_corner = corner
            view.holding_position = pos
            view.status = "holding"
            view.last_event = f"hold:{corner}"
            self.state.peer_claims[peer_id] = corner

    def note_peer_placed(self, peer_id: str, corner: str, bed_corner: str) -> None:
        with self.lock:
            self.state.placed_corners.add(corner)
            self.state.peer_claims.pop(peer_id, None)
            view = self._peer(peer_id)
            view.holding_corner = None
            view.holding_position = None
            view.status = "placed"
            view.last_event = f"placed:{corner}@{bed_corner}"

    def note_peer_released(self, peer_id: str, corner: str) -> None:
        with self.lock:
            self.state.peer_claims.pop(peer_id, None)
            view = self._peer(peer_id)
            view.holding_corner = None
            view.holding_position = None
            view.status = "online"
            view.last_event = f"released:{corner}"

    def note_peer_assist(self, peer_id: str, corner: str, reason: str) -> None:
        with self.lock:
            view = self._peer(peer_id)
            view.last_event = f"askAssist:{corner}:{reason}"

    def note_peer_state(self, peer_id: str, status: str, holding: Optional[str]) -> None:
        with self.lock:
            view = self._peer(peer_id)
            view.status = status
            view.holding_corner = holding

    # Mutators called from the sim main thread.

    def set_status(self, status: str) -> None:
        with self.lock:
            self.state.status = status

    def set_held_corner(self, corner: Optional[str]) -> None:
        with self.lock:
            self.state.held_corner = corner

    def record_placement(self, corner: str) -> None:
        with self.lock:
            self.state.placed_corners.add(corner)

    def tally_rpc(self, name: str) -> None:
        with self.lock:
            self.state.rpc_counts[name] = self.state.rpc_counts.get(name, 0) + 1

    # Snapshots for the driver/dashboard.

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "device_id": self.state.device_id,
                "role": self.state.role,
                "status": self.state.status,
                "held_corner": self.state.held_corner,
                "placed_corners": sorted(self.state.placed_corners),
                "peer_claims": dict(self.state.peer_claims),
                "peers": {pid: vars(view) for pid, view in self.state.peers.items()},
                "last_event": self.state.last_event,
                "rpc_counts": dict(self.state.rpc_counts),
            }

    def _peer(self, peer_id: str) -> PeerView:
        view = self.state.peers.get(peer_id)
        if view is None:
            view = PeerView(device_id=peer_id)
            self.state.peers[peer_id] = view
        return view


class BedMakingG1Driver(DeviceDriver):
    """DC driver for one Unitree G1 swarm peer.

    Each instance is bound to a :class:`SwarmAgent` and exposes the
    bed-making swarm event surface. Constructor accepts a ``role`` label
    (purely cosmetic — used to flavour the identity description) and a
    ``device_id``.
    """

    device_type = "unitree_g1_bed_making"

    def __init__(
        self,
        *,
        device_id: str,
        role: str = "peer",
        agent: Optional[SwarmAgent] = None,
    ) -> None:
        super().__init__()
        self._device_id = device_id
        self._agent = agent if agent is not None else SwarmAgent(device_id=device_id, role=role)

    @property
    def identity(self) -> DeviceIdentity:
        return DeviceIdentity(
            device_type=self.device_type,
            manufacturer="Unitree",
            model="G1 EDU",
            description=(
                f"Unitree G1 humanoid — swarm peer ({self._agent.state.role}) in the "
                "Strands Robots two-humanoid MuJoCo bed-making demo."
            ),
        )

    @property
    def status(self) -> DeviceStatus:
        snap = self._agent.snapshot()
        my_status = snap["status"]
        busy = my_status not in {"online", "available"}
        # The Device Connect dashboard renders the live status pill GREEN with
        # a flashing dot when the device's availability is "available" — that
        # is the convention used by the other registered devices (tokyo, sf,
        # reachy, etc.). Using "online" or "idle" renders the pill grey even
        # though the device is heartbeating, so use "available" as the
        # default ready state.
        return DeviceStatus(
            availability="busy" if busy else "available",
            busy_score=1.0 if busy else 0.0,
            location="bed_making_simulator",
        )

    async def connect(self) -> None:
        logger.info(
            "BedMakingG1Driver connected device_id=%s role=%s",
            self._device_id,
            self._agent.state.role,
        )

    async def disconnect(self) -> None:
        logger.info("BedMakingG1Driver disconnected device_id=%s", self._device_id)

    # ── Swarm-visible RPCs ────────────────────────────────────────────────

    @rpc()
    async def getStatus(self) -> dict:
        """Return the agent's snapshot for the dashboard."""
        self._agent.tally_rpc("getStatus")
        return self._agent.snapshot()

    @rpc()
    async def emergencyStopAll(self, reason: str = "") -> dict:
        """Broadcast an emergency stop so all peers go idle."""
        self._agent.tally_rpc("emergencyStopAll")
        await self.emergencyStop(reason=reason)
        return {"reason": reason}

    # ── Swarm coordination events (emitted by this peer) ──────────────────

    @emit()
    async def claimCorner(self, corner: str = "") -> None:
        """Announce this peer intends to grab ``corner`` next."""

    @emit()
    async def cornerHeld(self, corner: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Announce this peer is currently holding ``corner`` at (x, y, z)."""

    @emit()
    async def cornerPlaced(self, corner: str = "", bed_corner: str = "") -> None:
        """Announce this peer has placed ``corner`` over ``bed_corner``."""

    @emit()
    async def cornerReleased(self, corner: str = "") -> None:
        """Announce this peer has let go of ``corner`` — it is free again."""

    @emit()
    async def askAssist(self, corner: str = "", reason: str = "") -> None:
        """Ask peers to assist with ``corner`` (e.g., to hold it in place)."""

    @emit()
    async def heartbeat(self, status: str = "online", holding: str = "") -> None:
        """Periodic state heartbeat broadcast every few seconds."""

    @emit()
    async def emergencyStop(self, reason: str = "") -> None:
        """Network-wide stop signal."""

    # ── Peer event subscribers ────────────────────────────────────────────

    @on(event_name="claimCorner")
    async def onPeerClaim(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        if corner:
            self._agent.note_peer_claim(device_id, corner)
            logger.info("%s saw peer %s claim %s", self._device_id, device_id, corner)

    @on(event_name="cornerHeld")
    async def onPeerHeld(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        if not corner:
            return
        pos = (
            float(payload.get("x", 0.0)),
            float(payload.get("y", 0.0)),
            float(payload.get("z", 0.0)),
        )
        self._agent.note_peer_held(device_id, corner, pos)

    @on(event_name="cornerPlaced")
    async def onPeerPlaced(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        bed_corner = payload.get("bed_corner", "")
        if corner:
            self._agent.note_peer_placed(device_id, corner, bed_corner)
            logger.info("%s saw peer %s place %s on %s", self._device_id, device_id, corner, bed_corner)

    @on(event_name="cornerReleased")
    async def onPeerReleased(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        if corner:
            self._agent.note_peer_released(device_id, corner)

    @on(event_name="askAssist")
    async def onPeerAsk(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        reason = payload.get("reason", "")
        if corner:
            self._agent.note_peer_assist(device_id, corner, reason)

    @on(event_name="heartbeat")
    async def onPeerHeartbeat(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        status = payload.get("status", "online")
        holding = payload.get("holding", "") or None
        self._agent.note_peer_state(device_id, status, holding)

    @on(event_name="emergencyStop")
    async def onPeerEmergencyStop(self, device_id: str, event_name: str, payload: dict) -> None:
        logger.warning(
            "device=%s emergency stop received from %s reason=%r",
            self._device_id,
            device_id,
            payload.get("reason"),
        )
        self._agent.set_status("stopped")

    # ── Periodic heartbeat ────────────────────────────────────────────────

    @periodic(interval=2.0, wait_for_completion=True)
    async def _emitHeartbeat(self) -> None:
        snap = self._agent.snapshot()
        try:
            await self.heartbeat(status=snap["status"], holding=snap.get("held_corner") or "")
        except Exception as exc:
            logger.debug("heartbeat emit failed: %s", exc)

    # ── Helpers used by the sim main thread ───────────────────────────────

    @property
    def agent(self) -> SwarmAgent:
        return self._agent

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def role(self) -> str:
        return self._agent.state.role
