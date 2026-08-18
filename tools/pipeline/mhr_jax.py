"""Pure-JAX port of the MHR (Momentum Human Rig) parametric body model.

SOMA-X's MHR identity runs a TorchScript character model whose rest shape depends
on BOTH identity (45 shape coeffs) AND `scale_params` (68 body-part scales) — the
`scale_params` are "required for MHR" in SOMA-X. Our earlier MHR pack used only the
45 blend-shape directions (exact at neutral, but no body-part scaling). This module
ports the full rig so scale_params work end-to-end.

MHR forward (rest, zero pose), reverse-engineered from `mhr_model_lod1.pt`:

    v          = base_shape + shape_vectors·identity (+ face_vectors·face)
    joint_par  = parameter_transform (889,249) @ [pose=0 | scale(68) | id=0]   # (127,7)
    local_t    = joint_translation_offsets + joint_par[:, :3]
    local_q    = prerotation ⊗ euler_xyz(joint_par[:, 3:6])
    local_s    = 2 ** joint_par[:, 6]
    global     = FK over parents (scaled-rigid skel-state composition)
    skin_xf    = global ∘ inverse_bind_pose         # per joint
    verts      = Σ_j W[v,j] · apply(skin_xf_j, v)   # linear blend skinning

Validated to <0.01 mm against the TorchScript model. Reading the .pt uses torch
once (offline export); evaluation is pure JAX.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Buffer export / load
# ---------------------------------------------------------------------------
_KEYS = ("base", "shape_vectors", "face_vectors", "parameter_transform",
         "joint_offsets", "joint_prerotations", "joint_parents",
         "inverse_bind_pose", "skin_weights", "faces")


def export_mhr_rig(pt_path: str, out_npz: str):
    """One-off: read the MHR TorchScript checkpoint and save rig buffers as .npz."""
    import torch
    m = torch.jit.load(pt_path, map_location="cpu")
    sd = m.state_dict()
    g = lambda k: sd[k].numpy()
    base = g("character_torch.blend_shape.base_shape")               # (V,3)
    V = base.shape[0]
    J = g("character_torch.skeleton.joint_parents").shape[0]
    si = g("character_torch.linear_blend_skinning.skin_indices_flattened").astype(np.int64)
    sw = g("character_torch.linear_blend_skinning.skin_weights_flattened").astype(np.float32)
    vi = g("character_torch.linear_blend_skinning.vert_indices_flattened").astype(np.int64)
    W = np.zeros((V, J), np.float32)
    W[vi, si] = sw
    faces = g("character_torch.mesh.faces").astype(np.int32)
    np.savez(
        out_npz,
        base=base.astype(np.float32),
        shape_vectors=g("character_torch.blend_shape.shape_vectors").astype(np.float32),   # (45,V,3)
        face_vectors=g("face_expressions_model.shape_vectors").astype(np.float32),          # (72,V,3)
        parameter_transform=g("character_torch.parameter_transform.parameter_transform").astype(np.float32),  # (889,249)
        joint_offsets=g("character_torch.skeleton.joint_translation_offsets").astype(np.float32),  # (J,3)
        joint_prerotations=g("character_torch.skeleton.joint_prerotations").astype(np.float32),    # (J,4) xyzw
        joint_parents=g("character_torch.skeleton.joint_parents").astype(np.int64),                # (J,)
        inverse_bind_pose=g("character_torch.linear_blend_skinning.inverse_bind_pose").astype(np.float32),  # (J,8)
        skin_weights=W,
        faces=faces,
    )
    return out_npz


def load_mhr_rig(npz_path: str) -> dict:
    """Load rig buffers (npz) into JAX arrays — no torch needed."""
    d = np.load(npz_path, allow_pickle=False)
    rig = {k: jnp.asarray(d[k]) for k in _KEYS if k in d}
    rig["joint_parents"] = np.asarray(d["joint_parents"]).astype(int)  # static (host)
    rig["faces"] = np.asarray(d["faces"]).astype(np.int32)
    rig["num_identity"] = int(d["shape_vectors"].shape[0])
    rig["num_scale"] = 68
    return rig


# ---------------------------------------------------------------------------
# Quaternion / skel-state math (Momentum convention: quaternions are [x,y,z,w])
# ---------------------------------------------------------------------------
def _qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _qrot(q, v):
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * jnp.cross(qv, v)
    return v + qw * t + jnp.cross(qv, t)


def _axis_quat(axis: int, ang):
    c = jnp.cos(ang / 2.0)
    s = jnp.sin(ang / 2.0)
    comps = [jnp.zeros_like(ang), jnp.zeros_like(ang), jnp.zeros_like(ang), c]
    comps[axis] = s
    return jnp.stack(comps, axis=-1)


def _euler_xyz_to_quat(e):
    qx = _axis_quat(0, e[..., 0])
    qy = _axis_quat(1, e[..., 1])
    qz = _axis_quat(2, e[..., 2])
    return _qmul(_qmul(qx, qy), qz)


def _fk(local_t, local_q, local_s, parents):
    """Scaled-rigid skel-state FK over the (static) parents list."""
    J = local_t.shape[0]
    gt = [None] * J
    gq = [None] * J
    gs = [None] * J
    for j in range(J):
        p = int(parents[j])
        if p < 0 or p == j:
            gt[j], gq[j], gs[j] = local_t[j], local_q[j], local_s[j]
        else:
            gq[j] = _qmul(gq[p], local_q[j])
            gs[j] = gs[p] * local_s[j]
            gt[j] = gt[p] + _qrot(gq[p], gs[p] * local_t[j])
    return jnp.stack(gt), jnp.stack(gq), jnp.stack(gs)


# ---------------------------------------------------------------------------
# Forward (rest shape)
# ---------------------------------------------------------------------------
def mhr_rest_vertices(rig: dict, identity_coeffs=None, scale_params=None, face_coeffs=None):
    """MHR rest-pose vertices (V, 3) in centimeters, as a function of identity + scale.

    identity_coeffs: (45,) shape coeffs (default zeros)
    scale_params:    (68,) body-part log2 scales (default zeros = neutral)
    """
    V = rig["base"].shape[0]
    nid = rig["num_identity"]
    idc = jnp.zeros(nid) if identity_coeffs is None else jnp.asarray(identity_coeffs)
    sc = jnp.zeros(rig["num_scale"]) if scale_params is None else jnp.asarray(scale_params)

    v = rig["base"] + jnp.einsum("nvd,n->vd", rig["shape_vectors"], idc)
    if face_coeffs is not None:
        v = v + jnp.einsum("nvd,n->vd", rig["face_vectors"], jnp.asarray(face_coeffs))

    # model_parameters (249) = [pose(136)=0 | scale(68) | identity(45)=0]
    mp = jnp.concatenate([jnp.zeros(136), sc, jnp.zeros(nid)])
    jp = (rig["parameter_transform"] @ mp).reshape(-1, 7)        # (J, 7)

    local_t = rig["joint_offsets"] + jp[:, :3]
    local_q = _qmul(rig["joint_prerotations"], _euler_xyz_to_quat(jp[:, 3:6]))
    local_s = jnp.exp(jp[:, 6] * np.log(2.0))
    gt, gq, gs = _fk(local_t, local_q, local_s, rig["joint_parents"])

    inv = rig["inverse_bind_pose"]
    it, iq, iss = inv[:, :3], inv[:, 3:7], inv[:, 7]
    st = gt + _qrot(gq, gs[:, None] * it)        # compose(global, inv_bind)
    sq = _qmul(gq, iq)
    ss = gs * iss

    def apply_joint(j):
        return _qrot(sq[j], ss[j] * v) + st[j]   # (V, 3)
    deformed = jax.vmap(apply_joint)(jnp.arange(st.shape[0]))   # (J, V, 3)
    return jnp.einsum("vj,jvd->vd", rig["skin_weights"], deformed)


def mhr_soma_rest(rig: dict, identity_pack, identity_coeffs=None, scale_params=None):
    """MHR rest shape transferred to SOMA topology, in meters.

    identity_pack: dict-like from np.load('identity_mhr.npz') providing the MHR
        faces + barycentric correspondence (bary_face_ids, bary_coords).
    Returns (V_soma, 3) meters, ready to use as SOMALayer rest vertices.
    """
    mhr_cm = np.asarray(mhr_rest_vertices(rig, identity_coeffs, scale_params))  # (V_mhr,3) cm
    faces = np.asarray(identity_pack["faces"])
    fid = np.asarray(identity_pack["bary_face_ids"])
    bary = np.asarray(identity_pack["bary_coords"])
    tri = mhr_cm[faces[fid]]                          # (V_soma, 3, 3)
    soma_cm = (bary[..., None] * tri).sum(axis=1)     # (V_soma, 3)
    return (soma_cm * 0.01).astype(np.float32)        # cm -> m (MHR native_unit)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export MHR rig buffers to .npz")
    # Resolved through soma_jax.assets: mhr_model_lod1.pt ships inside the
    # vendored SOMA-X submodule, not under assets/, so a literal path rots.
    from soma_jax.assets import resolve
    p.add_argument("--pt",
                   default=str(resolve("MHR/mhr_model_lod1.pt", required=False) or ""))
    p.add_argument("--out", default="assets/mhr_rig.npz")
    a = p.parse_args()
    export_mhr_rig(a.pt, a.out)
    print(f"Wrote {a.out}")
