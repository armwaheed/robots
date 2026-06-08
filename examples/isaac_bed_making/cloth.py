"""PhysX particle-cloth bedsheet + grasp helpers for the Isaac Sim demo.

Self-contained (only ``omni.physx`` + ``pxr``). Facts learned the hard way on the
DGX Spark and baked in here:

* The cloth mesh is built **by hand** as a numpy/USD **quad** grid. Quads matter:
  Isaac's auto particle-cloth turns every *mesh edge* into a stiff stretch spring,
  so a triangulated grid makes the cell diagonals inextensible and the sheet locks
  into a rigid plate; with quads the diagonal becomes a soft *shear* spring and the
  cloth drapes. Springs are kept elastic for the same reason.
* Particle cloth only simulates when **GPU dynamics** is enabled on the physics
  scene (:func:`enable_gpu_dynamics`). Without it the sheet is inert.
* On the **GPU pipeline the deformed particle positions never sync to the USD
  mesh / Fabric**, so a headless camera renders the flat authored mesh. Read the
  live positions from a PhysX *tensor cloth view* (:func:`make_cloth_view`,
  :func:`view_positions`) and blit them into the mesh (:func:`sync_mesh_from_view`)
  each frame; run with ``SimulationCfg(use_fabric=False)`` so the render reads USD.

A :func:`grasp` auto-attachment helper is provided, but the demo grips the cloth
by **friction** (rubberized hands) instead — see ``manipulation.apply_hand_friction``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, Vt


def enable_gpu_dynamics(stage, scene_path: str) -> None:
    """Enable GPU dynamics + GPU broadphase on the physics scene (required for
    particle cloth). Safe to call once after the World/Sim physics scene exists."""
    prim = stage.GetPrimAtPath(scene_path)
    psa = PhysxSchema.PhysxSceneAPI.Apply(prim)
    psa.CreateEnableGPUDynamicsAttr(True)
    psa.CreateBroadphaseTypeAttr("GPU")
    psa.CreateSolverTypeAttr("TGS")


def find_physics_scene_path(stage) -> Optional[str]:
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            return p.GetPath().pathString
    return None


@dataclass
class Bedsheet:
    """Handle to a particle-cloth bedsheet and its grid topology."""

    prim_path: str
    mesh: object
    particle_system_path: str
    nx: int
    ny: int
    size: Tuple[float, float]
    # vertex ids of the four corners, keyed by compass label
    corner_vids: Dict[str, int]

    def vid(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i


def build_bedsheet(
    stage,
    scene_path: str,
    prim_path: str = "/World/Sheet",
    *,
    size: Tuple[float, float] = (2.2, 2.0),
    resolution: Tuple[int, int] = (44, 40),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.95),
    color: Tuple[float, float, float] = (0.85, 0.82, 0.72),
    # ── The "MuJoCo recipe" (validated 2026-06-06 in isolation, no robots) ──────
    # The shipped sheet draped as a stiff cantilever and "fluttered in the wind".
    # Root cause was NOT the engine — it was that the sheet was built HEAVY (~35 kg)
    # and FINE (~1750 verts), so a soft-spring lattice with that much inertia sloshes
    # in a limit cycle that no damping kills. MuJoCo's flex sheet (the one that looked
    # good) is the opposite: ~0.18 kg total, 143 verts, ~4.4 cm thick. Porting that
    # recipe to this SAME PhysX particle cloth — featherlight + coarse (set via the
    # scene's particle_mass / resolution) + low bend + more solver iters — drops the
    # frame-to-frame motion from ±5 cm to ±1 cm and the overhang folds DOWN over the
    # edge. (Quads still matter: a triangulated particle grid locks into a rigid plate
    # because every mesh edge becomes a stretch spring — see the mesh build below.)
    stretch_stiffness: float = 300.0,   # elastic (1500 was rigid; 150–300 look identical)
    bend_stiffness: float = 0.3,        # low so it folds over the edge; >2 = rigid cantilever,
                                        # <0.1 = the free edge curls/rolls
    shear_stiffness: float = 10.0,
    damping: float = 1.5,
    # PBD particle-material: cloth-surface friction kept LOW so the overhang slides
    # down the mattress side instead of catching. drag MUST be ~0 — even 0.1 acts like
    # a parachute on the flat falling flap and it never drapes (0.4 floated it dead flat).
    friction: float = 0.2,
    drag: float = 0.0,
    # More solver position iterations make each step CONVERGE — a too-low count leaves
    # the cloth in a buzzing limit cycle (the overhang "breathes" forever). 16 is the
    # Omniverse default; 48 settles it.
    solver_iterations: int = 48,
    # Per-particle mass. Keep the sheet FEATHERLIGHT: at the coarse scene resolution
    # this totals ~0.2 kg (MuJoCo was 0.18 kg). A heavy sheet sloshes; a light one
    # hangs dead still. (This is per particle, so it scales with vert count — pair it
    # with a COARSE resolution in the scene.)
    particle_mass: float = 0.001,
    thickness: float = 0.0,
    fold: bool = False,
    fold_start: float = 0.66,
    accordion: bool = False,
    accordion_start: float = 0.3,
    accordion_gather: float = 0.55,
    accordion_waves: int = 4,
    accordion_amp: float = 0.09,
) -> Bedsheet:
    """Create a draping particle-cloth sheet as a quad grid mesh.

    ``origin`` is the world translate of the sheet centre. With ``fold=False`` the
    sheet starts flat (+Z-up) and drapes under gravity. With ``fold=True`` the
    **foot end is folded back over itself**: the length up to ``fold_start`` lies
    flat, and the remaining foot fraction folds back over the top (as a turned-back
    sheet at the foot of the bed, like the Figure Helix clip) — so the bed starts
    unmade and the robots arrange it. The four cloth corners are still tracked.
    """
    from omni.physx.scripts import particleUtils, physicsUtils

    Wx, Wy = size
    nx, ny = resolution
    ox, oy, oz = origin
    layer_dz = max(thickness, 0.04)  # height the folded-back flap sits above the sheet
    x_fold = fold_start * Wx - Wx / 2.0  # local x of the fold line

    # Author the points directly in WORLD space (bake the origin in) and keep the
    # mesh transform identity — see the rendering note below: with use_fabric=True
    # the renderer reads the mesh's world transform from Fabric, so a non-identity
    # xformOp would be applied ON TOP of the world-space deformed points we blit each
    # frame, floating the sheet up by the origin. Identity transform + world points
    # means what we blit is exactly what renders.
    # Accordion: the head-side fraction up to ``accordion_start`` lies flat (the part
    # the robots grip); the foot fraction is GATHERED into a pleated ruffle — its flat
    # x-extent is compressed by ``accordion_gather`` and the slack taken up by
    # ``accordion_waves`` vertical pleats of height ``accordion_amp``. That slack is the
    # point: pulling the flat head edge toward the head simply UNSPOOLS the accordion
    # instead of dragging a sheet that is stuck flat to the mattress, so the friction
    # grip doesn't have to overpower the whole cover (and the robots aren't yanked over).
    acc_fold_x = accordion_start * Wx - Wx / 2.0
    acc_extent = (1.0 - accordion_start) * Wx

    pts: List[Gf.Vec3f] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            u, v = i / nx, j / ny
            if accordion and u > accordion_start:
                t = (u - accordion_start) / (1.0 - accordion_start)
                x = acc_fold_x + t * acc_extent * accordion_gather
                z = accordion_amp * (0.5 - 0.5 * math.cos(2.0 * math.pi * accordion_waves * t))
            elif fold and u > fold_start:
                # Fold the foot flap back over the top toward the head.
                x = x_fold - (u - fold_start) * Wx
                z = layer_dz
            else:
                x, z = u * Wx - Wx / 2.0, 0.0
            pts.append(Gf.Vec3f(x + ox, (v * Wy - Wy / 2.0) + oy, z + oz))
    idx: List[int] = []
    counts: List[int] = []

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    # Build the sheet as QUADS, not triangles. Isaac's auto particle-cloth turns
    # every *mesh edge* into a stiff stretch spring; a triangulated grid adds a
    # diagonal edge to each cell, so those diagonals become inextensible stretch
    # springs and lock the sheet into a rigid plate. With quads the diagonal is
    # NOT a mesh edge, so the solver adds it as a SOFT shear spring and the cloth
    # drapes. (MuJoCo's flex gets triangle elasticity for free; PBD does not.)
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            idx += [a, b, c, d]
            counts += [4]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateDisplayColorAttr().Set([color])
    # Identity transform — the points above are already in world space. (Authoring a
    # translate here and world-space points would double-transform the sheet under
    # Fabric and float it off the bed.)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    # Particle radius. Default: neighbours just touch at rest (radius = half the
    # grid spacing) — the canonical Omniverse recipe. A `thickness` override
    # fattens the particles into a "duvet"-like sheet with a thicker collision
    # profile the hands can catch more easily; capped just under the spacing so
    # particles don't overlap at rest (use a coarser grid for a thicker sheet).
    spacing = Wy / ny
    rest_offset = min(thickness / 2.0, 0.49 * spacing) if thickness > 0.0 else 0.5 * spacing
    contact_offset = rest_offset * 1.5
    ps_path = Sdf.Path(prim_path + "_particles")
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=ps_path,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
        particle_contact_offset=contact_offset,
        solid_rest_offset=rest_offset,
        fluid_rest_offset=0.0,
        solver_position_iterations=solver_iterations,
        simulation_owner=Sdf.Path(scene_path),
    )
    pmat = Sdf.Path(prim_path + "_material")
    particleUtils.add_pbd_particle_material(stage, pmat, drag=drag, lift=0.0, friction=friction)
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(ps_path), pmat)

    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=Sdf.Path(prim_path),
        dynamic_mesh_path=None,
        particle_system_path=ps_path,
        spring_stretch_stiffness=stretch_stiffness,
        spring_bend_stiffness=bend_stiffness,
        spring_shear_stiffness=shear_stiffness,
        spring_damping=damping,
        self_collision=True,
        self_collision_filter=True,
    )
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).GetMassAttr().Set(particle_mass * len(pts))

    # Map the four sheet corners onto compass labels (matches the swarm driver's
    # SHEET_TO_BED: A=NW, B=NE, C=SE, D=SW). +x = East, +y = North.
    corner_vids = {
        "NW": vid(0, ny),
        "NE": vid(nx, ny),
        "SE": vid(nx, 0),
        "SW": vid(0, 0),
    }
    return Bedsheet(prim_path, mesh, ps_path.pathString, nx, ny, size, corner_vids)


def grasp(stage, cloth_path: str, body_path: str, attach_path: str,
          bind_offset: float = 0.10) -> None:
    """Create a PhysX auto-attachment binding the cloth to a rigid body
    (a robot hand) where they overlap — i.e. grab whatever cloth is in the hand.

    ``bind_offset`` widens the overlap radius so cloth particles within that many
    metres of the palm are attached, letting the grasp catch the sheet even when
    the IK leaves the palm a few cm short of the surface."""
    if stage.GetPrimAtPath(attach_path):
        return
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, Sdf.Path(attach_path))
    att.GetActor0Rel().SetTargets([Sdf.Path(cloth_path)])
    att.GetActor1Rel().SetTargets([Sdf.Path(body_path)])
    api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    # Particle cloth binds through the "deformable vertex" path of the auto
    # attachment; widen its overlap offset. Wrapped defensively in case the
    # attribute set differs across PhysX schema versions.
    try:
        api.CreateEnableDeformableVertexAttachmentsAttr(True)
        api.CreateDeformableVertexOverlapOffsetAttr(float(bind_offset))
        api.CreateEnableRigidSurfaceAttachmentsAttr(True)
    except Exception:
        pass


def release(stage, attach_path: str) -> None:
    """Remove a grasp attachment (let go of the cloth)."""
    if stage.GetPrimAtPath(attach_path):
        stage.RemovePrim(attach_path)


# ── live deformed positions + rendering ────────────────────────────────────
# On the GPU PhysX pipeline, particle-cloth deformation is NOT written back to
# the USD mesh (with Fabric on it isn't synced at all), so a headless camera
# renders the stale, flat authored mesh. The cloth IS draping in the backend —
# we read the live particle positions via the PhysX *tensor* cloth view and blit
# them into the visual mesh ourselves. Requires SimulationCfg(use_fabric=False)
# so the renderer reads USD.
def make_cloth_view(prim_path: str = "/World/Sheet", backend: str = "torch"):
    """Create a PhysX tensor view over a particle cloth (call after the first
    sim step so the cloth is registered in the physics scene)."""
    import omni.physics.tensors as tensors

    sv = tensors.create_simulation_view(backend)
    sv.set_subspace_roots("/")
    return sv.create_particle_cloth_view(prim_path.replace(".*", "*"))


def view_positions(view, idx: int = 0):
    """Return the live deformed particle positions (world frame) as (N,3) numpy."""
    import numpy as np

    pos = view.get_positions()  # (count, max_particles*3), flat per cloth
    row = pos[idx]
    if hasattr(row, "detach"):
        row = row.detach().cpu().numpy()
    return np.asarray(row, dtype=float).reshape(-1, 3)


def sync_mesh_from_view(view, mesh, idx: int = 0):
    """Blit live cloth positions into the visual mesh so the renderer shows the
    real (deformed) cloth. Returns the (N,3) positions for any geometry logic."""
    import numpy as np
    from pxr import Gf, Vt

    p = view_positions(view, idx)
    # Positions are world-space; zero the mesh translate so they aren't
    # double-transformed by the leftover build-time AddTranslateOp.
    attr = mesh.GetPrim().GetAttribute("xformOp:translate")
    if attr and tuple(attr.Get()) != (0.0, 0.0, 0.0):
        attr.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    return p


# ── Fabric (use_fabric=True) rendering path ─────────────────────────────────
# With use_fabric=True the RTX renderer reads geometry from Fabric, not USD, so a
# USD points blit is ignored and the cloth renders flat. We must run fabric on for
# the *robot* to render its motion: Isaac Lab's SimulationContext.forward() only
# calls update_articulations_kinematic() when fabric is enabled, so with fabric off
# the articulation renders frozen at its spawn pose. NVIDIA confirms (forums:
# "Using Fabric with Particles") that point-instancer changes can't go through
# Fabric but **mesh updates do** — our bedsheet is a UsdGeom.Mesh, so we write its
# deformed points straight into Fabric/usdrt each render. PhysX does NOT auto-sync
# particle-cloth positions to Fabric, so we still read them from the tensor cloth
# view. Fabric stores each prim's WORLD transform directly, so a non-identity mesh
# xform would be applied on top of the world-space points we write — floating the
# sheet up by the origin (the "sheet loads a few feet in the air" bug). We therefore
# build the mesh with its points already in world space and an IDENTITY transform
# (see build_bedsheet), so the deformed positions we blit render exactly where they
# are — no resetXformStack or per-frame transform fixup needed.
def make_fabric_points(prim_path: str = "/World/Sheet"):
    """Attach the Fabric/usdrt stage and return the cloth mesh's ``points``
    attribute for in-place per-frame updates (use with :func:`sync_fabric_from_view`
    when running ``SimulationCfg(use_fabric=True)``). Call after ``sim.reset()`` and
    the first sim step. Returns ``None`` if the prim isn't in Fabric yet."""
    import omni.usd
    import usdrt

    rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    prim = rt_stage.GetPrimAtPath(prim_path)
    if not (prim and prim.IsValid()):
        return None
    attr = prim.GetAttribute("points")
    if attr and attr.IsValid():
        return attr
    return prim.CreateAttribute("points", usdrt.Sdf.ValueTypeNames.Point3fArray, True)


def sync_fabric_from_view(view, fabric_points, idx: int = 0):
    """Blit live deformed cloth positions into the Fabric mesh ``points`` so the
    RTX renderer (``use_fabric=True``) shows the real cloth. Returns the (N,3)
    positions. ``fabric_points`` comes from :func:`make_fabric_points`."""
    import numpy as np
    import usdrt

    p = view_positions(view, idx)
    fabric_points.Set(usdrt.Vt.Vec3fArray(np.ascontiguousarray(p, dtype=np.float32)))
    return p


def corner_world_positions(view, sheet: "Bedsheet"):
    """Return {label: (x,y,z)} live world positions of the four sheet corners,
    read from the tensor cloth ``view`` (see :func:`make_cloth_view`)."""
    pts = view_positions(view)
    out = {}
    if pts.shape[0] == 0:
        return out
    for label, vid in sheet.corner_vids.items():
        if vid < pts.shape[0]:
            out[label] = tuple(round(float(c), 4) for c in pts[vid])
    return out


# ── Render-side VISUAL THICKNESS: a closed double-layer "shell" ─────────────
# The particle cloth is a single-layer membrane (one quad grid, N particles), so
# the renderer draws it as a zero-thickness sheet of paper — physically correct
# for the sim, but it *looks* thin. A real made bed reads as a substantial,
# thick cover. We give it visual body WITHOUT touching the physics: the membrane
# stays the simulated cloth, and we drive a SEPARATE visual mesh — the same grid
# extruded into a slab — from the same live particle positions each frame.
#
# Construction (static topology, points refreshed per frame):
#   * TOP layer    = membrane + n·(h/2)   (n = per-vertex surface normal)
#   * BOTTOM layer = membrane − n·(h/2)
#   * SIDE WALLS   = quads stitching the top and bottom rims around the perimeter
# so the result is a closed quad slab of thickness ``h`` that hugs the draped
# membrane (offset along the normal, so the thickness stays uniform even where the
# overhang folds down vertical). The membrane mesh is hidden (the slab encloses it
# anyway); hiding it is safe — particle cloth simulates off the particle system,
# not the mesh's render visibility, and if the sim ever *did* die the slab (built
# FROM the live particle positions) would render flat, so the mp4 still catches it.
@dataclass
class ShellMesh:
    """Handle to the visual double-layer slab that thickens a :class:`Bedsheet`."""

    prim_path: str
    mesh: object
    nx: int
    ny: int
    n_particles: int   # membrane vertex count = (nx+1)*(ny+1)
    thickness: float


def _grid_vertex_normals(grid):
    """Per-vertex unit normals for a draped quad grid.

    ``grid`` is (ny+1, nx+1, 3) world positions (row j = +y, col i = +x). The
    normal is cross(∂/∂i, ∂/∂j) so a flat sheet in the xy-plane yields +z; where
    the sheet folds down over an edge the normal rotates with it, keeping the
    extruded thickness perpendicular to the surface everywhere."""
    import numpy as np

    du = np.gradient(grid, axis=1)   # tangent along +x (columns)
    dv = np.gradient(grid, axis=0)   # tangent along +y (rows)
    n = np.cross(du, dv)
    mag = np.linalg.norm(n, axis=2, keepdims=True)
    return n / np.where(mag < 1e-9, 1.0, mag)


def shell_points(membrane_pts, nx: int, ny: int, thickness: float):
    """Map the N membrane particle positions to the 2N slab points.

    ``membrane_pts`` is (N,3) in particle/``vid`` order (vid = j*(nx+1)+i). Returns
    (2N,3): the TOP layer (indices 0..N-1) then the BOTTOM layer (N..2N-1), each in
    the same vid order, offset ±thickness/2 along the per-vertex normal."""
    import numpy as np

    grid = np.asarray(membrane_pts, dtype=float).reshape(ny + 1, nx + 1, 3)
    n = _grid_vertex_normals(grid)
    h = thickness / 2.0
    top = (grid + n * h).reshape(-1, 3)
    bot = (grid - n * h).reshape(-1, 3)
    return np.concatenate([top, bot], axis=0)


def build_shell_mesh(
    stage,
    sheet: "Bedsheet",
    *,
    thickness: float = 0.06,
    color: Tuple[float, float, float] = (0.86, 0.86, 0.92),
    prim_path: str = "/World/SheetShell",
    hide_membrane: bool = True,
) -> ShellMesh:
    """Create the closed double-layer visual slab for ``sheet`` and (by default)
    hide the thin membrane. Topology is fixed; call :func:`sync_shell_fabric` (or
    :func:`sync_shell_mesh`) each frame to push the live deformed points."""
    nx, ny = sheet.nx, sheet.ny
    n_part = (nx + 1) * (ny + 1)

    def top(i: int, j: int) -> int:
        return j * (nx + 1) + i

    def bot(i: int, j: int) -> int:
        return n_part + j * (nx + 1) + i

    idx: List[int] = []
    counts: List[int] = []
    # TOP surface — same winding as the membrane (CCW from above → normal up).
    for j in range(ny):
        for i in range(nx):
            idx += [top(i, j), top(i + 1, j), top(i + 1, j + 1), top(i, j + 1)]
            counts.append(4)
    # BOTTOM surface — reversed winding so its normal points down.
    for j in range(ny):
        for i in range(nx):
            idx += [bot(i, j), bot(i, j + 1), bot(i + 1, j + 1), bot(i + 1, j)]
            counts.append(4)
    # PERIMETER side walls — march the boundary loop CCW (viewed from +z) and
    # stitch each top edge to the matching bottom edge.
    perim: List[Tuple[int, int]] = (
        [(i, 0) for i in range(nx + 1)]          # foot edge, +x
        + [(nx, j) for j in range(1, ny + 1)]    # right edge, +y
        + [(i, ny) for i in range(nx - 1, -1, -1)]  # head edge, −x
        + [(0, j) for j in range(ny - 1, 0, -1)]    # left edge, −y
    )
    for (ia, ja), (ib, jb) in zip(perim, perim[1:] + perim[:1]):
        idx += [top(ia, ja), top(ib, jb), bot(ib, jb), bot(ia, ja)]
        counts.append(4)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * (2 * n_part)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateDisplayColorAttr().Set([color])
    # doubleSided so the slab is opaque from every angle regardless of winding;
    # subdivision "none" so the renderer draws the actual extruded quads (a crisp
    # cover) rather than Catmull-Clark-rounding the slab into a pillow.
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    # Identity transform + world-space points (same contract as the membrane mesh:
    # under Fabric a non-identity xform double-transforms the world points).
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    if hide_membrane:
        UsdGeom.Imageable(sheet.mesh).MakeInvisible()

    return ShellMesh(prim_path, mesh, nx, ny, n_part, thickness)


def sync_shell_fabric(view, fabric_points, shell: ShellMesh, idx: int = 0):
    """Blit the live deformed slab points into the shell's Fabric ``points`` (for
    ``use_fabric=True``). ``fabric_points`` comes from :func:`make_fabric_points`
    on the shell prim. Returns the (N,3) membrane positions."""
    import numpy as np
    import usdrt

    p = view_positions(view, idx)
    sp = shell_points(p, shell.nx, shell.ny, shell.thickness)
    fabric_points.Set(usdrt.Vt.Vec3fArray(np.ascontiguousarray(sp, dtype=np.float32)))
    return p


def sync_shell_mesh(view, shell: ShellMesh, idx: int = 0):
    """USD-path equivalent of :func:`sync_shell_fabric` (for ``use_fabric=False``)."""
    import numpy as np
    from pxr import Vt

    p = view_positions(view, idx)
    sp = shell_points(p, shell.nx, shell.ny, shell.thickness)
    shell.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(sp.astype(np.float32)))
    return p
