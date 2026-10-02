"""Rig a custom SOMA body template mesh and export as a UsdSkel USD file.

Upstream: ``tools/rig_soma_body_mesh.py`` (SOMA-X v0.3.3).

The input mesh MUST share the exact topology of the SOMA body template mesh
(``c_skin_mid``): 18 056 vertices with the same vertex ordering and
connectivity as ``SOMA_template_rig.usda``. The skinning weights are
transferred directly from the template rig, so any mismatch in vertex count or
ordering will produce incorrect deformation.

Supported input formats: OBJ, USD/USDA/USDC.

Upstream v0.3.3 raises before writing anything with its default procedural
layer: the 78-joint ``skeleton_transfer.fit`` bind is converted to local with
the 110-joint twist-rig ``joint_parent_ids``. The tool's logic predates the twist
rig (unchanged since v0.2.0), so the port writes the public 78-joint rig the
fitted bind belongs to — joint names, parents and folded skinning weights, the
rig upstream's own v0.3 ``export_soma_usd`` writes — with the template mesh's
quads and UV sets.

JAX port: ``--device`` is accepted and ignored.

Usage::

    python tools/rig_soma_body_mesh.py --input my_body.obj --output my_body_rig.usda
    python tools/rig_soma_body_mesh.py --input my_body.usda --output my_body_rig.usda
    python tools/rig_soma_body_mesh.py --input my_body.obj --output my_body_rig.usda --unit meters
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from logging_utils import add_logging_args, configure_logging  # noqa: E402

# Must match the vertex count of the SOMA body template mesh (c_skin_mid).
_SOMA_BODY_VERTEX_COUNT = 18_056
logger = logging.getLogger(__name__)


def load_mesh_vertices(path):
    """Load vertex positions from an OBJ or USD/USDA/USDC file.

    Returns:
        (V, 3) float32 numpy array of vertex positions.

    Raises:
        ValueError: If the format is unsupported or multiple meshes are found
            with ambiguous vertex counts.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        import trimesh
        mesh = trimesh.load(str(path), maintain_order=True, process=False)
        return np.asarray(mesh.vertices, dtype=np.float32)

    if suffix in (".usd", ".usda", ".usdc"):
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(path))
        mesh_prims = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
        if not mesh_prims:
            raise ValueError(f"No Mesh prim found in '{path}'")
        # A single mesh is used directly; otherwise pick the one whose vertex
        # count matches the SOMA body template.
        if len(mesh_prims) == 1:
            prim = mesh_prims[0]
        else:
            candidates = [
                p for p in mesh_prims
                if len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []) == _SOMA_BODY_VERTEX_COUNT
            ]
            if len(candidates) == 1:
                prim = candidates[0]
            elif len(candidates) == 0:
                counts = sorted(
                    set(len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []) for p in mesh_prims))
                raise ValueError(
                    f"No mesh with {_SOMA_BODY_VERTEX_COUNT} vertices found in '{path}'. "
                    f"Vertex counts present: {counts}")
            else:
                raise ValueError(
                    f"Multiple meshes with {_SOMA_BODY_VERTEX_COUNT} vertices found in '{path}'. "
                    "Please provide a file with a single body mesh.")
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        return np.array(pts, dtype=np.float32)

    raise ValueError(f"Unsupported mesh format '{suffix}'. Expected .obj, .usd, .usda, or .usdc.")


def main():
    from soma_jax.units import Unit

    parser = argparse.ArgumentParser(
        description=(
            "Rig a custom SOMA body template mesh and export as UsdSkel. "
            f"Input mesh must have exactly {_SOMA_BODY_VERTEX_COUNT} vertices "
            "matching the SOMA body template topology (c_skin_mid)."))
    parser.add_argument("--input", required=True,
                        help="Input mesh file (.obj, .usd, .usda, .usdc). Must match SOMA body "
                             "template topology.")
    parser.add_argument("--output", required=True, help="Output UsdSkel file (.usd, .usda, .usdc).")
    parser.add_argument("--data-root", default=None,
                        help="Path to SOMA assets directory (default: soma_jax.assets).")
    parser.add_argument("--unit", choices=[u.unit_name for u in Unit],
                        default=Unit.CENTIMETERS.unit_name,
                        help="Unit of the input mesh coordinates (default: centimeters, matching "
                             "SOMA template rig).")
    parser.add_argument("--device", default="cpu",
                        help="Accepted for upstream CLI compatibility; JAX uses its default "
                             "device (set JAX_PLATFORMS to choose).")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    import jax.numpy as jnp

    from soma_jax import SOMALayer
    from soma_jax.assets import data_root as default_data_root
    from soma_jax.geometry.rig_utils import joint_world_to_local
    from soma_jax.usd_io import load_lod_rig_from_usd, save_soma_usd

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    input_unit = Unit.from_name(args.unit)

    # --- Load input mesh ---
    logger.info(f"Loading mesh: {args.input} ...")
    verts = load_mesh_vertices(args.input)
    logger.info(f"  Vertices: {len(verts)}")
    if len(verts) != _SOMA_BODY_VERTEX_COUNT:
        raise ValueError(
            f"Input mesh has {len(verts)} vertices, expected {_SOMA_BODY_VERTEX_COUNT}. "
            "The mesh must match the SOMA body template topology (c_skin_mid) exactly.")

    # --- Initialize SOMA ---
    logger.info("\nInitializing SOMA...")
    soma = SOMALayer.from_upstream_assets(identity_model_type="mhr", output_unit=input_unit,
                                          data_root=str(data_root))

    # --- Fit skeleton to the input shape ---
    # The layer's skeleton transfer works in metres; its binds are reported in
    # the layer's output unit, as upstream's.
    logger.info("Fitting skeleton...")
    to_m = input_unit.meters_per_unit
    bind_world = soma.skeleton_transfer.fit(jnp.asarray(verts) * to_m)       # (J, 4, 4)
    bind_world = bind_world.at[..., :3, 3].divide(to_m)
    parents = np.asarray(soma._parents_np)
    bind_local = joint_world_to_local(bind_world, parents)                   # (J, 4, 4)

    # --- Export ---
    logger.info(f"\nExporting rig: {args.output}")
    mesh = load_lod_rig_from_usd(data_root / "SOMA_template_rig.usda", "mid")
    save_soma_usd(
        args.output,
        joint_names=list(soma.public_joint_names),
        joint_parent_ids=parents,
        bind_transforms_world=np.asarray(bind_world),
        bind_transforms_local=np.asarray(bind_local),
        rest_shape=verts,
        faces=np.asarray(soma.faces),
        face_vert_indices=mesh.get("face_vert_indices"),
        face_vert_counts=mesh.get("face_vert_counts"),
        uv_data=mesh.get("uv_data"),
        skinning_weights=np.asarray(soma.public_skinning_weights()),
        unit=args.unit,
    )


if __name__ == "__main__":
    main()
