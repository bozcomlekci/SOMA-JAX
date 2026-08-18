"""Drive every SOMA-JAX body model from a SINGLE SMPL-X motion sequence.

Source motion: an SMPL-X motion `.npz`, which stores per-frame SMPL-X
parameters.

Two retargeting paths, both in JAX:

  * SMPL / SMPL-H / SMPL-X  — share the SMPL kinematic tree, so the SMPL-X body
    pose drives them natively (exact). SMPL gets the 21 shared body joints;
    SMPL-H/SMPL-X additionally get the hand (and face) joints.

  * SOMA / MHR / Anny / Garment — use the SOMA 78-joint skeleton. We follow the
    NVlabs/SOMA-X approach: pose the SMPL-X mesh, transfer the posed vertices to
    SOMA topology (barycentric, via the SMPLX SOMA_wrap correspondence), then
    recover SOMA joint rotations by inverse-LBS (`PoseInversion`). The recovered
    rotations are pose data that is identity-agnostic, so the same SOMA poses are
    applied to every SOMA-family identity.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from build_identity_packs import load_obj_soma_x_style, compute_bary


# ----------------------------------------------------------------------------
# Body-model loading via SMPL-JAX (pure JAX — no PyTorch / vchoutas smplx)
# ----------------------------------------------------------------------------
# SMPL-X packs shape + expression in a single `shapedirs` block (300 shape +
# 100 expression). SMPL-JAX expects them split; we split on load. We also read
# the model directly from .npz/.pkl so we never depend on chumpy.
_SMPLX_SHAPE_SPLIT = 300


def _read_model_dict(path: str) -> dict:
    """Read a SMPL/SMPL-X model file (.npz numpy or .pkl pickle) into a dict.

    The .pkl branch uses SOMA-JAX's chumpy-free unpickler, so classic SMPL .pkl
    files load without pulling in chumpy."""
    if path.lower().endswith(".npz"):
        raw = np.load(path, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    from soma_jax.body_models.model_io import _ChumpyFreeUnpickler
    with open(path, "rb") as f:
        raw = _ChumpyFreeUnpickler(f, encoding="latin1").load()
    return raw if isinstance(raw, dict) else {k: getattr(raw, k) for k in dir(raw)}


def _normalize_arrays(d: dict):
    v_template = np.asarray(d["v_template"], dtype=np.float32)
    V = v_template.shape[0]
    shapedirs = np.asarray(d["shapedirs"], dtype=np.float32)
    if shapedirs.ndim == 2:
        shapedirs = shapedirs.reshape(V, 3, -1)
    posedirs = np.asarray(d["posedirs"], dtype=np.float32)
    if posedirs.ndim == 3:
        posedirs = posedirs.reshape(V * 3, -1)
    elif posedirs.shape[0] != V * 3:
        posedirs = posedirs.T
    jr = d["J_regressor"]
    if hasattr(jr, "todense"):
        jr = np.asarray(jr.todense())
    elif hasattr(jr, "toarray"):
        jr = np.asarray(jr.toarray())
    J_regressor = np.asarray(jr, dtype=np.float32)
    kintree = np.asarray(d["kintree_table"], dtype=np.int64)
    parents = kintree[0].copy()
    parents[0] = -1
    weights = np.asarray(d["weights"], dtype=np.float32)
    faces = np.asarray(d.get("f", d.get("faces")), dtype=np.int32)
    return v_template, shapedirs, posedirs, J_regressor, parents, weights, faces


def load_smpl_jax(path: str, num_betas: int = 10):
    """Load a plain SMPL model as a SMPL-JAX SMPLModel."""
    from smpl_jax import SMPLModel
    v_template, shapedirs, posedirs, J_regressor, parents, weights, faces = _normalize_arrays(_read_model_dict(path))
    return SMPLModel(
        v_template=v_template, shapedirs=shapedirs, posedirs=posedirs,
        J_regressor=J_regressor, parents=parents, weights=weights, faces=faces,
        num_betas=num_betas,
    )


def load_smplx_jax(path: str, num_betas: int = 10, num_expression_coeffs: int = 10):
    """Load an SMPL-X model as a SMPL-JAX SMPLXModel, splitting the packed basis."""
    from smpl_jax import SMPLXModel
    d = _read_model_dict(path)
    v_template, shapedirs, posedirs, J_regressor, parents, weights, faces = _normalize_arrays(d)
    expr = d.get("exprdirs", d.get("expr_dirs"))
    if expr is None:
        exprdirs = shapedirs[..., _SMPLX_SHAPE_SPLIT:_SMPLX_SHAPE_SPLIT + num_expression_coeffs]
    else:
        exprdirs = np.asarray(expr, dtype=np.float32)
        if exprdirs.ndim == 2:
            exprdirs = exprdirs.reshape(v_template.shape[0], 3, -1)
    return SMPLXModel(
        v_template=v_template, shapedirs=shapedirs[..., :num_betas], exprdirs=exprdirs,
        posedirs=posedirs, J_regressor=J_regressor, parents=parents, weights=weights,
        faces=faces, num_betas=num_betas, num_expression_coeffs=num_expression_coeffs,
    )


# ----------------------------------------------------------------------------
# Motion loading
# ----------------------------------------------------------------------------
def load_smplx_motion(path: str, num_frames: int = 48) -> dict:
    """Load + subsample an SMPL-X motion `.npz` file."""
    d = np.load(path, allow_pickle=True)
    T = d["poses"].shape[0]
    idx = np.linspace(0, T - 1, num_frames).astype(int)
    pose_eye = d["pose_eye"][idx].astype(np.float32)  # (N, 6) = [leye(3), reye(3)]
    fps = float(d["mocap_frame_rate"]) if "mocap_frame_rate" in d.files else 60.0
    return dict(
        root_orient=d["root_orient"][idx].astype(np.float32),   # (N, 3)
        pose_body=d["pose_body"][idx].astype(np.float32),       # (N, 63) = 21 joints
        pose_hand=d["pose_hand"][idx].astype(np.float32),       # (N, 90) = 30 joints
        pose_jaw=d["pose_jaw"][idx].astype(np.float32),         # (N, 3)
        leye_pose=pose_eye[:, :3],
        reye_pose=pose_eye[:, 3:],
        trans=d["trans"][idx].astype(np.float32),               # (N, 3)
        betas=d["betas"].astype(np.float32),                    # (16,)
        n_frames=len(idx),
        # Source timing — so the renderer can play the GIF at real-time speed.
        source_total_frames=T,
        source_fps=fps,
        source_duration_s=T / fps,
    )


# ----------------------------------------------------------------------------
# Native SMPL-family drivers (shared kinematic tree)
# ----------------------------------------------------------------------------
def smplx_params(model, m, use_betas=True):
    from smpl_jax import SMPLXParams
    N = m["n_frames"]
    nb = model.num_betas
    betas = np.tile(m["betas"][:nb], (N, 1)).astype(np.float32) if use_betas else np.zeros((N, nb), np.float32)
    return SMPLXParams(
        betas=jnp.asarray(betas),
        body_pose=jnp.asarray(m["pose_body"]),
        global_orient=jnp.asarray(m["root_orient"]),
        transl=jnp.asarray(m["trans"]),
        expression=jnp.zeros((N, model.num_expression_coeffs), np.float32),
        jaw_pose=jnp.asarray(m["pose_jaw"]),
        leye_pose=jnp.asarray(m["leye_pose"]),
        reye_pose=jnp.asarray(m["reye_pose"]),
        left_hand_pose=jnp.asarray(m["pose_hand"][:, :45]),
        right_hand_pose=jnp.asarray(m["pose_hand"][:, 45:]),
    )


def smplh_params(model, m):
    from soma_jax import SMPLHParams
    N = m["n_frames"]
    return SMPLHParams(
        betas=jnp.zeros((N, model.num_betas), np.float32),
        body_pose=jnp.asarray(m["pose_body"]),                  # (N, 63), 21 joints
        global_orient=jnp.asarray(m["root_orient"]),
        transl=jnp.asarray(m["trans"]),
        left_hand_pose=jnp.asarray(m["pose_hand"][:, :45]),
        right_hand_pose=jnp.asarray(m["pose_hand"][:, 45:]),
    )


def smpl_params(model, m):
    from smpl_jax import SMPLParams
    N = m["n_frames"]
    # SMPL body_pose = 23 joints. Joints 1..21 match SMPL-X body; the two SMPL
    # hand joints (22, 23) get the relaxed pose (zeros).
    body = np.zeros((N, 23 * 3), np.float32)
    body[:, : 21 * 3] = m["pose_body"]
    return SMPLParams(
        betas=jnp.zeros((N, model.num_betas), np.float32),
        body_pose=jnp.asarray(body),
        global_orient=jnp.asarray(m["root_orient"]),
        transl=jnp.asarray(m["trans"]),
    )


# ----------------------------------------------------------------------------
# SMPL-X motion -> SOMA poses (inverse-LBS retargeting)
# ----------------------------------------------------------------------------
def _world_to_local(R_world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Convert per-joint world rotations to parent-relative local rotations."""
    R_local = np.empty_like(R_world)
    for j in range(R_world.shape[0]):
        p = parents[j]
        R_local[j] = R_world[j] if p < 0 else R_world[p].T @ R_world[j]
    return R_local


def smplx_motion_to_soma_poses(smplx_model, soma_layer, motion, hf_dir=None,
                                pose_mode="analytical", refine_iters=0, return_extras=False):
    """Retarget an SMPL-X motion onto the SOMA 78-joint skeleton.

    Args:
        refine_iters: if >0, run autograd FK refinement after the analytical solve
            (SOMA-X's "Analytical + Autograd FK" mode — slower, higher accuracy).
        return_extras: if True, also return a dict with ``per_vertex_error`` (N,V),
            ``root_translation`` (N,3), and ``joint_names`` (for SOMA-X-format export).

    Returns:
        poses_local: (N, 78, 3) parent-relative axis-angle for SOMAParams.poses
        (and an extras dict if return_extras=True)
    """
    if hf_dir is None:
        from soma_jax.assets import data_root
        hf_dir = data_root()

    from soma_jax import PoseInversion
    from smpl_jax import SMPLXParams
    from soma_jax.geometry.transforms import rotmat_to_axis_angle
    if refine_iters > 0:
        pose_mode = "combined"

    # SMPL-X rest (single frame, zero pose, betas=0 so it matches the SOMA_wrap template).
    rest_single = SMPLXParams(
        betas=jnp.zeros((1, smplx_model.num_betas), np.float32),
        body_pose=jnp.zeros((1, 63), np.float32),
        global_orient=jnp.zeros((1, 3), np.float32),
        transl=jnp.zeros((1, 3), np.float32),
        expression=jnp.zeros((1, smplx_model.num_expression_coeffs), np.float32),
        jaw_pose=jnp.zeros((1, 3), np.float32),
        leye_pose=jnp.zeros((1, 3), np.float32),
        reye_pose=jnp.zeros((1, 3), np.float32),
        left_hand_pose=jnp.zeros((1, 45), np.float32),
        right_hand_pose=jnp.zeros((1, 45), np.float32),
    )
    smplx_rest = np.asarray(smplx_model(rest_single).vertices[0])      # (Vx, 3)
    smplx_faces = np.asarray(smplx_model.faces)
    smplx_posed = np.asarray(smplx_model(smplx_params(smplx_model, motion, use_betas=False)).vertices)  # (N, Vx, 3)
    # Remove the global root translation so the posed target lives in the SAME
    # root-at-origin frame as the zero-translation rest, the PoseInversion fit, and
    # the transl0=0 reconstruction below. Otherwise the per_vertex_error (and, in
    # autograd-refine mode, the recovered poses) are offset by ‖trans‖ (~1.5 m).
    # The translation is carried separately in extras["root_translation"] and
    # reapplied at render time.
    smplx_posed = smplx_posed - motion["trans"][:, None, :].astype(np.float32)

    # SMPL-X -> SOMA barycentric correspondence (computed on the rest template,
    # in canonical SOMA vertex order, via the HF SMPLX/SOMA_wrap.obj).
    wrap = load_obj_soma_x_style(Path(hf_dir) / "SMPLX" / "SOMA_wrap.obj")
    V_wrap = np.asarray(wrap.vertices, dtype=np.float32)              # (18056, 3), SOMA order
    face_ids, bary = compute_bary(smplx_rest, smplx_faces, V_wrap)

    def transfer(verts):  # (V, 3) -> (18056, 3)
        tri = verts[smplx_faces[face_ids]]                            # (18056, 3, 3)
        return (bary[..., None] * tri).sum(axis=1)

    soma_rest = transfer(smplx_rest)                                  # (18056, 3)
    soma_posed = np.stack([transfer(v) for v in smplx_posed], axis=0)  # (N, 18056, 3)

    # SOMA rest joints via the canonical J_regressor.
    J_reg = np.asarray(soma_layer.J_regressor)                       # (78, 18056)
    soma_rest_joints = J_reg @ soma_rest                            # (78, 3)
    parents = soma_layer._parents_np
    weights = np.asarray(soma_layer.weights)

    inv = PoseInversion(
        jnp.asarray(soma_rest), jnp.asarray(weights),
        jnp.asarray(soma_rest_joints), parents,
    )
    R_world = np.asarray(inv.fit(jnp.asarray(soma_posed), mode=pose_mode,
                                 num_refine_iters=refine_iters, lr=5e-3))  # (N, 78, 3, 3)

    # World -> local, then to axis-angle.
    N, J = R_world.shape[:2]
    poses_local = np.empty((N, J, 3), np.float32)
    for n in range(N):
        R_local = _world_to_local(R_world[n], parents)
        poses_local[n] = np.asarray(
            jnp.stack([rotmat_to_axis_angle(jnp.asarray(R_local[j])) for j in range(J)])
        )

    # Finger joints now have correct rest positions (fixed rig), so inverse-LBS
    # recovers usable finger curl from the SMPL-X hand pose baked into the target
    # mesh — keep them so the hands follow the motion. Only the "*End" leaf
    # markers carry no skin weight and have degenerate offsets; zero those (their
    # rotation affects nothing and they are not drawn).
    names = [str(n) for n in np.asarray(soma_layer.joint_names)]
    for j, nm in enumerate(names):
        if "End" in nm:
            poses_local[:, j] = 0.0

    # Direct SMPL-X -> SOMA hand retargeting. Mesh-based PoseInversion has too
    # little signal in the hand region (few verts, tiny skinning weight) and
    # under-recovers finger curl. We have the exact MANO axis-angle in the
    # motion's `pose_hand`, so for the fingers we copy SMPL-X values onto the
    # corresponding SOMA finger joints by name.
    #   MANO order per hand (15 joints): index1..3, middle1..3, pinky1..3,
    #                                    ring1..3, thumb1..3   (45 floats)
    #   SOMA per hand: Thumb1,2,3 + Index/Middle/Ring/Pinky 1..4. Index1
    #   (metacarpal) stays rigid; MANO j1/j2/j3 → SOMA finger 2/3/4. Thumb maps 1:1.
    name_to_idx = {n: i for i, n in enumerate(names)}
    _MANO_GROUPS = [("Index", 0), ("Middle", 9), ("Pinky", 18), ("Ring", 27), ("Thumb", 36)]
    for side, hand_pose in [("Left", motion["pose_hand"][:, :45]),
                            ("Right", motion["pose_hand"][:, 45:])]:
        for finger, start in _MANO_GROUPS:
            for k in range(3):                     # MANO joint k = 1..3
                src = hand_pose[:, start + 3 * k:start + 3 * k + 3]   # (N, 3)
                if finger == "Thumb":
                    dst = f"{side}HandThumb{k + 1}"           # Thumb maps 1:1 (3 joints)
                else:
                    dst = f"{side}Hand{finger}{k + 2}"        # MCP->2, PIP->3, DIP->4
                j = name_to_idx.get(dst)
                if j is not None:
                    poses_local[:, j] = src.astype(np.float32)

    if not return_extras:
        return poses_local

    # Per-vertex reconstruction error vs the SMPL-X target (SOMA-X export field).
    # Reconstruct with the SAME rest that PoseInversion fit (the SMPL-X-transferred
    # SOMA rest), not the canonical SOMA-shape, so the error measures pose fidelity.
    from soma_jax.geometry.transforms import axis_angle_to_rotmat
    import jax
    SR = jnp.asarray(soma_rest); RJ = jnp.asarray(soma_rest_joints)
    transl0 = jnp.zeros((1, 3), np.float32)
    errs = []
    for n in range(N):
        R = jax.vmap(axis_angle_to_rotmat)(jnp.asarray(poses_local[n]))
        out = soma_layer.pose(R[None], transl0, SR[None], RJ[None])
        errs.append(np.linalg.norm(np.asarray(out.vertices[0]) - soma_posed[n], axis=-1))
    extras = {
        "per_vertex_error": np.stack(errs, axis=0).astype(np.float32),   # (N, V) meters
        "root_translation": motion["trans"].astype(np.float32),          # (N, 3)
        "joint_names": np.asarray(names),
        # Export the SOMA-topology rest shape + joints used during PoseInversion
        # so callers can render the retargeted poses on the SAME bind that the
        # rotations were fit to. The canonical SOMA bind is Y-up centered at
        # the feet (Y∈[0,1.68]), while the SMPL-X-transferred bind is centered
        # at the pelvis (Y∈[-1.3,0.42]); mixing the two causes a ~1 m global
        # offset + a stale skeleton in --motion-mode renders.
        "soma_rest_verts": np.asarray(soma_rest, dtype=np.float32),       # (V, 3)
        "soma_rest_joints": np.asarray(soma_rest_joints, dtype=np.float32),  # (J, 3)
    }
    return poses_local, extras
