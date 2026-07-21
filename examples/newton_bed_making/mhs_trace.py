"""Message-flow trace for the two G1s' MHS coordination.

The demo's claim is that the two robots coordinate *over MHS*, not that a script calls two objects
in a fixed order. This module is how that claim is checked rather than asserted: every message the
peers exchange is recorded with the MHS **plane**, **profile** and **transport** it travelled over,
who sent it, who received it, and what the simulator did as a result. The run prints the table and
writes it as JSON, so a reader can diff the trace against the narrative.

The vocabulary is the SDK's own (``mhs`` docs: *Concepts*, and the ``mhs`` CLI's plane split):

**Planes** — which side of MHS a message uses.

* ``fleet`` — broker-backed and remote: ``discover`` / ``invoke`` / ``broadcast`` / ``subscribe``,
  reached through a ``Fleet`` (``connect_fleet``) over a messaging transport. This is the plane
  governed by MHS's auth and tenancy rules, and the one both peers mount on under ``--broker``.
* ``direct`` — a locally attached rig, no broker: ``keys`` / ``get`` / ``set`` / ``describe``
  against a shared-memory segment. **This demo never uses the direct plane** — the robots are
  simulated articulations, not a rig with slots on this host.

**Profile** — which of the four MHS layers a process runs. This demo is **WIRE**: transport plus
application API, no buffer or format layer. It holds no shared-memory device state; it talks to
peers over a transport. (A **BUFFER** process would additionally own local ``MhsDict`` storage.)

**Transports** — ``nats`` under ``--broker``. Under ``--loopback`` the transport is recorded as
``loopback``, which is deliberately *not* an MHS transport: it is this demo's in-process fan-out so
the path runs offline on one machine with no broker and no credentials. Loopback exercises the same
driver surface (the same ``@rpc`` bodies, the same ``@on`` handlers, the same payloads) but the bytes
never leave the process. Anything that must be true of the real fabric should be read off a
``--broker`` trace; that is the honest one.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Message kinds, in the direction they travel.
RPC_CALL = "rpc.call"          # caller -> device: invoke a procedure
RPC_RETURN = "rpc.return"      # device -> caller: the procedure's result
EVENT_PUBLISH = "event.publish"  # device -> fabric: an @emit fired
EVENT_DELIVER = "event.deliver"  # fabric -> device: an @on handler ran

BROADCAST = "*"                # destination for an event with no single addressee


@dataclass(frozen=True)
class Hop:
    """One MHS message, and what it caused."""

    seq: int
    t_ms: float          # milliseconds since the trace started
    kind: str            # RPC_CALL | RPC_RETURN | EVENT_PUBLISH | EVENT_DELIVER
    plane: str           # "fleet" | "direct" | "in-process"
    transport: str       # "nats" | "loopback"
    profile: str         # "WIRE" | "BUFFER"
    src: str             # device_id, or "sim" for the simulation main loop
    dst: str             # device_id, or BROADCAST
    name: str            # procedure or event name
    payload: Dict[str, Any] = field(default_factory=dict)
    effect: str = ""     # what the simulator did because of this message


class MessageTrace:
    """Thread-safe, bounded record of the MHS messages a run exchanged.

    Written from two threads — the simulation main loop (RPC calls) and the coordinator's asyncio
    thread (event publish/deliver) — so every mutation takes the lock.
    """

    def __init__(self, *, plane: str, transport: str, profile: str = "WIRE",
                 maxlen: int = 2000) -> None:
        self.plane = plane
        self.transport = transport
        self.profile = profile
        self._maxlen = maxlen
        self._hops: List[Hop] = []
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    def record(self, kind: str, src: str, dst: str, name: str,
               payload: Optional[Dict[str, Any]] = None, effect: str = "") -> Optional[Hop]:
        """Append one message. Returns the Hop, or None once the cap is reached."""
        with self._lock:
            if len(self._hops) >= self._maxlen:
                return None
            hop = Hop(
                seq=len(self._hops) + 1,
                t_ms=round((time.monotonic() - self._t0) * 1000.0, 1),
                kind=kind, plane=self.plane, transport=self.transport, profile=self.profile,
                src=src, dst=dst, name=name, payload=dict(payload or {}), effect=effect,
            )
            self._hops.append(hop)
            return hop

    def note_effect(self, seq: int, effect: str) -> None:
        """Attach the simulator action a message caused, once it is known.

        Hops are frozen, so this replaces the entry rather than mutating it."""
        with self._lock:
            for idx, hop in enumerate(self._hops):
                if hop.seq == seq:
                    self._hops[idx] = Hop(**{**asdict(hop), "effect": effect})
                    return

    def hops(self) -> List[Hop]:
        with self._lock:
            return list(self._hops)

    def render(self, limit: int = 80) -> str:
        """The trace as a fixed-width table: who sent what, over which plane/transport, to whom,
        and what the robot did about it."""
        hops = self.hops()[:limit]
        if not hops:
            return "(no MHS messages recorded)"
        head = f"{'#':>3}  {'t(ms)':>7}  {'kind':<14} {'plane':<10} {'transport':<9} " \
               f"{'from':<28} {'to':<28} {'message':<16} effect"
        lines = [head, "-" * len(head)]
        for h in hops:
            lines.append(
                f"{h.seq:>3}  {h.t_ms:>7.1f}  {h.kind:<14} {h.plane:<10} {h.transport:<9} "
                f"{_short(h.src):<28} {_short(h.dst):<28} {h.name:<16} {h.effect}")
        if len(self.hops()) > limit:
            lines.append(f"... {len(self.hops()) - limit} more")
        return "\n".join(lines)

    def summary(self) -> Dict[str, int]:
        """Message counts by kind — the one-line "did anything actually flow?" check."""
        out: Dict[str, int] = {}
        for h in self.hops():
            out[h.kind] = out.get(h.kind, 0) + 1
        return out

    def write_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"plane": self.plane, "transport": self.transport, "profile": self.profile,
             "summary": self.summary(), "hops": [asdict(h) for h in self.hops()]},
            indent=2))
        return path


def _short(device_id: str) -> str:
    """Trim the common ``beta-unitree-g1-humanoid-N`` prefix so the table stays readable."""
    return device_id.replace("beta-unitree-g1-humanoid-", "g1-") if device_id else "-"
