"""PhysX particle-cloth bedsheet + grasp helpers for the Isaac Sim demo.

This is deliberately self-contained and uses only ``omni.physx`` + ``pxr`` so it
works under either the plain Isaac Sim Python or the Isaac Lab Python. Two facts
learned the hard way on the DGX Spark and baked in here:

* The cloth mesh is built **by hand** as a numpy/USD triangulated grid. The
  ``CreateMeshPrimWithDefaultXform`` kit *command* silently fails in a headless
  standalone app, so we never use it.
* Particle cloth only simulates when **GPU dynamics** is enabled on the physics
  scene (``enable_gpu_dynamics`` below). Without it the sheet is inert.

The grasp uses a PhysX auto-attachment between the cloth and a rigid body
(a robot hand): create the attachment to grab a corner, delete it to release.
PhysX writes the deformed cloth positions into Fabric, not the USD points attr,
so read live corner positions via :func:`particle_positions`, never
``UsdGeom.Mesh.GetPointsAttr``.
"""

from __future__ import annotations

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
    stretch_stiffness: float = 10000.0,
    bend_stiffness: float = 200.0,
    shear_stiffness: float = 100.0,
    damping: float = 0.2,
    particle_mass: float = 0.02,
) -> Bedsheet:
    """Create a draping particle-cloth sheet as a triangulated grid mesh.

    ``origin`` is the world translate of the (flat, +Z-up) sheet centre. The
    sheet falls and drapes under gravity once the sim steps.
    """
    from omni.physx.scripts import particleUtils, physicsUtils

    Wx, Wy = size
    nx, ny = resolution

    pts: List[Gf.Vec3f] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            pts.append(Gf.Vec3f(i / nx * Wx - Wx / 2.0, j / ny * Wy - Wy / 2.0, 0.0))
    idx: List[int] = []
    counts: List[int] = []

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            idx += [a, b, c, a, c, d]
            counts += [3, 3]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateDisplayColorAttr().Set([color])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*origin))

    # Particle system sized so particles just touch at rest.
    rest_offset = 0.5 * (Wy / ny)
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
        solver_position_iterations=16,
        simulation_owner=Sdf.Path(scene_path),
    )
    pmat = Sdf.Path(prim_path + "_material")
    particleUtils.add_pbd_particle_material(stage, pmat, drag=0.1, lift=0.0, friction=0.6)
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


def grasp(stage, cloth_path: str, body_path: str, attach_path: str) -> None:
    """Create a PhysX auto-attachment binding the cloth to a rigid body
    (a robot hand) where they overlap — i.e. grab whatever cloth is in the hand."""
    if stage.GetPrimAtPath(attach_path):
        return
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, Sdf.Path(attach_path))
    att.GetActor0Rel().SetTargets([Sdf.Path(cloth_path)])
    att.GetActor1Rel().SetTargets([Sdf.Path(body_path)])
    PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())


def release(stage, attach_path: str) -> None:
    """Remove a grasp attachment (let go of the cloth)."""
    if stage.GetPrimAtPath(attach_path):
        stage.RemovePrim(attach_path)


def live_positions(mesh_path: str):
    """Return live deformed cloth vertex positions (world frame) as an (N,3)
    numpy array, read from the Fabric/USDRT stage that PhysX writes into.

    Returns an empty array if the runtime stage isn't available. The sheet's
    local translate is baked into the deformed points by PhysX, so these are
    already world-space."""
    import numpy as np
    import omni.usd

    try:
        import usdrt
        from usdrt import Sdf as RtSdf  # noqa: F401

        stage_id = omni.usd.get_context().get_stage_id()
        rt_stage = usdrt.Usd.Stage.Attach(stage_id)
        prim = rt_stage.GetPrimAtPath(mesh_path)
        attr = prim.GetAttribute("points")
        if not attr:
            return np.zeros((0, 3))
        pts = attr.Get()
        return np.array([[p[0], p[1], p[2]] for p in pts], dtype=float)
    except Exception:
        return np.zeros((0, 3))


def corner_world_positions(sheet: "Bedsheet"):
    """Return {label: (x,y,z)} live world positions of the four sheet corners."""
    pts = live_positions(sheet.prim_path)
    out = {}
    if pts.shape[0] == 0:
        return out
    for label, vid in sheet.corner_vids.items():
        if vid < pts.shape[0]:
            out[label] = tuple(round(float(c), 4) for c in pts[vid])
    return out
