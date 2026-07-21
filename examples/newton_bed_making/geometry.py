"""Authoritative scene geometry for the two-G1 Newton bed-making demo (world metres).

This is the SAME bed the Isaac Sim demo uses (``examples/isaac_bed_making/scene.py``) — copied
here, not rescaled, so the two demos frame the identical bed. The long axis is **x**: the head is at
-x (headboard + pillows), the foot at +x. Nothing in this module imports a simulator, so the
constants can be read by tests, the MHS layer, or a different engine without pulling in Isaac Lab.

Layout summary::

    -x  headboard + two propped pillows            foot  +x
        ┌───────────────────────────────────────┐
        │                                        │
   -y   │              MATTRESS                   │  robot 1 stands at +y, yaw -90
 robot0 │           2.0 (x) x 1.8 (y)            │
   +90  │                                        │
        └───────────────────────────────────────┘
        head edge x=-1.0            foot edge x=+1.0
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (x, y, z, w) — the Newton backend's convention

# ── Mattress ────────────────────────────────────────────────────────────────
BED_SIZE: Vec3 = (2.0, 1.8, 0.61)          # 2.0 long (x) x 1.8 wide (y) x 0.61 tall (z)
BED_CENTER: Vec3 = (0.0, 0.0, 0.305)       # bottom on the floor (center z = height / 2)
BED_TOP_Z: float = BED_CENTER[2] + BED_SIZE[2] / 2.0  # 0.61
HEAD_X: float = -BED_SIZE[0] / 2.0         # -1.0  (headboard end)
FOOT_X: float = +BED_SIZE[0] / 2.0         # +1.0
SIDE_Y: float = BED_SIZE[1] / 2.0          # 0.9   (mattress long-side edges at y = ±0.9)

OVERHANG: float = 0.2286                    # 9 inches of sheet past each long side

# ── Headboard + pillows (static props at the head) ──────────────────────────
HEADBOARD_SIZE: Vec3 = (0.12, 1.9, 0.95)
HEADBOARD_CENTER: Vec3 = (HEAD_X, 0.0, 0.55)   # flush against the mattress head, rises to ~1.0

PILLOW_SIZE: Vec3 = (0.5, 0.72, 0.16)
PILLOW_PROP_DEG: float = 70.0               # tip from flat: stood on the foot-side edge, leaning back
PILLOWS: Dict[str, Vec3] = {
    "left": (HEAD_X + 0.22, -0.43, BED_TOP_Z + 0.22),
    "right": (HEAD_X + 0.22, +0.43, BED_TOP_Z + 0.22),
}

# ── Bedsheet (proxy) ─────────────────────────────────────────────────────────
# Head edge (the grab line the robots draw toward the pillows) sits foot-of-the-pillows at ~mid-bed,
# so there is a real headward draw to make; the rest drapes footward with a 9-inch side overhang.
SHEET_LEN: float = 1.55                      # x: head edge -> off the foot
# The proxy cover is a RIGID slab (real coupled cloth needs Kit — see demo.py), so its half-width is
# kept below the robots' stance |y| to avoid the slab intersecting them. A soft sheet would drape the
# full 9-inch overhang; the proxy trades some overhang for physical clearance.
SHEET_WIDTH: float = 2.0                      # y: bed width 1.8 + ~0.10 overhang each side
SHEET_THICKNESS: float = 0.06
SHEET_HEAD_X: float = 0.15                    # world x of the sheet's head edge at rest
SHEET_REST_Z: float = BED_TOP_Z + SHEET_THICKNESS / 2.0
# How far headward the draw carries the sheet's head edge (toward, not past, the pillows).
SHEET_DRAW_DX: float = 0.75

# ── Robots: flank opposite long sides, face the bed, reach onto their near corner ──
# The bed-reach policy is a STANDING reach (no locomotion command), so the robots are placed at
# their bedside marks rather than walked in. MANIP_Y = 1.10 clears the y = ±0.9 mattress side.
MANIP_X: float = 0.40
MANIP_Y: float = 1.25                         # bedside stance; clears the mattress side and the cover overhang
ROBOT_SPAWN_Z: float = 0.74                  # G1_MINIMAL_CFG standing pelvis height (feet on floor)

ROBOTS: Dict[int, Dict] = {
    0: {"pos": (MANIP_X, -MANIP_Y, ROBOT_SPAWN_Z), "yaw_deg": 90.0, "side": "-y", "mirror": False},
    1: {"pos": (MANIP_X, +MANIP_Y, ROBOT_SPAWN_Z), "yaw_deg": -90.0, "side": "+y", "mirror": True},
}

# Bed corners in the world (compass labels match the swarm driver; +x = foot = East).
BED_CORNERS: Dict[str, Vec3] = {
    "NE": (FOOT_X, SIDE_Y, BED_TOP_Z),
    "NW": (HEAD_X, SIDE_Y, BED_TOP_Z),
    "SE": (FOOT_X, -SIDE_Y, BED_TOP_Z),
    "SW": (HEAD_X, -SIDE_Y, BED_TOP_Z),
}

# ── Camera: 3/4 view framing the whole bed + both robots ─────────────────────
CAM_EYE: Vec3 = (3.9, -3.5, 2.7)
CAM_LOOKAT: Vec3 = (0.0, 0.0, 0.5)
CAM_RES: Tuple[int, int] = (1280, 720)


def yaw_to_quat(yaw_deg: float) -> Quat:
    """Quaternion (x, y, z, w) for a rotation of ``yaw_deg`` about +Z.

    NOTE the (x, y, z, w) order: this Isaac Lab Newton build stores/reads root orientation as XYZW
    (an upright robot reads ``(0,0,0,1)``). A WXYZ value here silently spawns the robot rotated about
    X — the policy then sees inverted gravity and collapses.
    """
    h = math.radians(yaw_deg) / 2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


def roty_to_quat(deg: float) -> Quat:
    """Quaternion (x, y, z, w) for a rotation of ``deg`` about +Y (tips a pillow up on its edge)."""
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))
