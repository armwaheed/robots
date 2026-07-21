"""A real deformable bedsheet: a native MuJoCo ``flexcomp`` cloth, authored rumpled.

This replaces the rigid plank-like mocap cover of the first cut of the demo. The sheet is a 2-D flex
(a triangulated grid of point masses with continuum membrane + bending elasticity), so it drapes,
folds, wrinkles, gathers and unspools under contact — nothing here is kinematic.

Two authoring details matter:

* The grid is emitted as ``type="direct"``: we hand MuJoCo the vertex positions, so the sheet's
  *stress-free rest shape* is the rumpled shape it starts in. An unmade sheet that has been shoved
  toward the foot of the bed is not a stretched flat sheet — it is slack, and pulling it flat should
  cost only bending. Authoring the ripples as the rest state gets that right (and stops the sheet
  from exploding flat on the first step, which is what a flat rest shape does).
* The rumple is shallow on purpose. It reads as "unmade" and adds a few centimetres of slack, but the
  draw is mostly the robots *dragging* the sheet up the mattress, not unspooling folds. A sheet
  wrinkled deeply enough to store half a metre of slack in half a metre of bed is a rolled tube, and a
  rolled tube on a slick mattress rolls off the foot — measured, twice.

Collision filtering (contype/conaffinity bits, see CLOTH_BIT/WORLD_BIT): the cloth collides with the
bed, pillows, headboard, floor and the two hand capsules, but NOT with the robots' legs/torsos. The
robots stand 0.2 m off a bed whose sheet overhangs 9 inches, i.e. exactly where the drape hangs;
letting a 2.26 m-wide curtain wrap their shins is not part of the task and would perturb the balance
policy for no visual gain. This is a documented modelling choice, not a physics claim.
"""
from __future__ import annotations

import math

import numpy as np

# Collision bits. Robot geoms keep mjlab's default (contype=1, conaffinity=1).
ROBOT_BIT = 1
CLOTH_BIT = 4
WORLD_BIT = ROBOT_BIT | CLOTH_BIT      # 5: scenery both the robots and the cloth can touch


def sheet_points(
    head_x: float,
    flat_to_x: float,
    bunch_to_x: float,
    top_z: float,
    half_width: float,
    spacing: float,
    ripple_amp: float,
    ripple_len: float,
    foot_drop: float,
    lip_clear: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """Author the rumpled sheet as an (N,3) vertex array on an (nu,nv) grid.

    Along the material's long axis the path runs: a flat head run (the part the robots grab), then a
    rippled bunch (the slack), then a lip that folds down over the foot edge. Across the short axis
    it is flat and extends past the mattress by the 9-inch overhang, which drapes on the first step.
    """
    # ── long axis: walk the path by arc length so edge rest-lengths are uniform ──────────────────
    xs, zs = [], []
    x, phase = head_x, 0.0
    while x < bunch_to_x:
        xs.append(x)
        if x < flat_to_x:                                   # flat head run
            zs.append(top_z)
            x += spacing
        else:                                               # rippled bunch: raised cosine in z
            # z = top + A(1 - cos): the troughs sit ON the mattress instead of through it. Authoring
            # ripples that dip below the surface makes the first solver step fire the sheet off the
            # bed, which is exactly what it did.
            zs.append(top_z + ripple_amp * (1.0 - math.cos(phase)))
            slope = ripple_amp * (2 * math.pi / ripple_len) * math.sin(phase)
            dx = spacing / math.hypot(1.0, slope)
            x += dx
            phase += 2 * math.pi * dx / ripple_len
    # Foot lip: fold down over the foot face. It hangs `lip_clear` outboard of the foot edge so it
    # does not start life inside the bed's collision skirt — 6 mm of authored penetration is enough
    # to blow the solver up on the first step (measured: NaN at t=0.21 s).
    drop = spacing
    while drop <= foot_drop:
        xs.append(bunch_to_x + lip_clear)
        zs.append(top_z - drop)
        drop += spacing
    nu = len(xs)

    # ── short axis: uniform, symmetric about y=0, running past the mattress edge ─────────────────
    nv = int(round(2 * half_width / spacing)) + 1
    ys = np.linspace(-half_width, half_width, nv)

    pts = np.empty((nu * nv, 3))
    for i, (px, pz) in enumerate(zip(xs, zs)):
        for j, py in enumerate(ys):
            pts[i * nv + j] = (px, py, pz)
    return pts, nu, nv


def grid_elements(nu: int, nv: int) -> np.ndarray:
    """Two consistently-wound triangles per grid cell."""
    tris = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = a + 1
            c = a + nv
            d = c + 1
            tris.append((a, b, d))
            tris.append((a, d, c))
    return np.array(tris, dtype=int)


def flexcomp_xml(
    name: str,
    pts: np.ndarray,
    tris: np.ndarray,
    mass: float,
    radius: float,
    young: float,
    poisson: float,
    thickness: float,
    friction: float,
    edge_damping: float,
    rgba: tuple[float, float, float, float],
) -> str:
    """Emit the <flexcomp type="direct"> block for the authored cloth."""
    point = " ".join(f"{v:.4f}" for v in pts.reshape(-1))
    element = " ".join(str(int(v)) for v in tris.reshape(-1))
    return f"""
    <flexcomp name="{name}" type="direct" dim="2" radius="{radius}" mass="{mass}"
              rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
              point="{point}"
              element="{element}">
      <contact selfcollide="none" internal="false" condim="3"
               contype="{CLOTH_BIT}" conaffinity="{CLOTH_BIT}"
               solref="0.008 1" solimp="0.9 0.95 0.001"
               friction="{friction} 0.02 0.001"/>
      <edge damping="{edge_damping}"/>
      <elasticity young="{young}" poisson="{poisson}" thickness="{thickness}" elastic2d="both"/>
    </flexcomp>"""
