"""Geometry kernels on a synthetic rig — no model assets needed."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from soma_jax.geometry import (
    RadialBasisFunction, SkeletonTransfer, align_vectors, axis_angle_to_rotmat,
    chamfer_distance, compute_skeleton_levels, fk_levelorder, forward_kinematics, kabsch,
    lbs, lbs_sparse, lbs_transforms, newton_schulz, rotation_6d_to_rotmat,
    rotation_matrices_are_valid, rotmat_to_6d, rotmat_to_axis_angle, topk_skinning,
)
from soma_jax.geometry.interpolate import rbf_basis_weights_np

from ._helpers import need_module

RNG = np.random.default_rng(42)


def _random_rotmats(n, scale=0.8):
    return jax.vmap(axis_angle_to_rotmat)(
        jnp.asarray(RNG.normal(size=(n, 3)) * scale, jnp.float32))


# ------------------------------------------------------------------ rotations
def test_axis_angle_roundtrip():
    R = _random_rotmats(64)
    R2 = jax.vmap(axis_angle_to_rotmat)(jax.vmap(rotmat_to_axis_angle)(R))
    np.testing.assert_allclose(np.asarray(R2), np.asarray(R), atol=2e-5)


def test_6d_roundtrip():
    R = _random_rotmats(64)
    R2 = jax.vmap(rotation_6d_to_rotmat)(jax.vmap(rotmat_to_6d)(R))
    np.testing.assert_allclose(np.asarray(R2), np.asarray(R), atol=2e-5)


def test_rotations_are_valid():
    assert bool(jnp.all(rotation_matrices_are_valid(_random_rotmats(32))))


def test_kabsch_and_align_vectors_recover_rotation():
    R = axis_angle_to_rotmat(jnp.array([0.3, -0.7, 0.4]))
    src = jnp.asarray(RNG.normal(size=(80, 3)), jnp.float32)
    np.testing.assert_allclose(np.asarray(kabsch(src, src @ R.T)), np.asarray(R), atol=1e-4)
    for method in ("auto", "kabsch", "newton-schulz"):
        got = align_vectors(src @ R.T, src, method=method)
        np.testing.assert_allclose(np.asarray(got), np.asarray(R), atol=2e-3, err_msg=method)


def test_newton_schulz_orthonormalises():
    A = _random_rotmats(16) + 0.05 * jnp.asarray(RNG.normal(size=(16, 3, 3)), jnp.float32)
    Q = newton_schulz(A)
    eye = jnp.broadcast_to(jnp.eye(3), Q.shape)
    np.testing.assert_allclose(np.asarray(Q @ jnp.swapaxes(Q, -1, -2)), np.asarray(eye), atol=1e-3)


# ------------------------------------------------------------------ FK + LBS
def test_fk_scan_matches_levelorder(synthetic_rig):
    r = synthetic_rig
    R = _random_rotmats(len(r["parents"]), 0.5)
    G1 = forward_kinematics(R, jnp.asarray(r["joints"]), r["parents"])
    G2 = fk_levelorder(R, jnp.asarray(r["joints"]), jnp.asarray(r["parents"]),
                       compute_skeleton_levels(r["parents"]))
    np.testing.assert_allclose(np.asarray(G1), np.asarray(G2), atol=1e-5)


def test_identity_pose_is_rest(synthetic_rig):
    r = synthetic_rig
    J = len(r["parents"])
    G = forward_kinematics(jnp.broadcast_to(jnp.eye(3), (J, 3, 3)), jnp.asarray(r["joints"]),
                           r["parents"])
    np.testing.assert_allclose(np.asarray(G[:, :3, 3]), r["joints"], atol=1e-6)
    T = lbs_transforms(G, jnp.asarray(r["joints"]))[None]
    v = lbs(jnp.asarray(r["verts"])[None], jnp.zeros((1,) + r["verts"].shape), T,
            jnp.asarray(r["weights"]))
    np.testing.assert_allclose(np.asarray(v[0]), r["verts"], atol=1e-5)


def test_sparse_lbs_matches_dense(synthetic_rig):
    r = synthetic_rig
    J = len(r["parents"])
    G = forward_kinematics(_random_rotmats(J, 0.6), jnp.asarray(r["joints"]), r["parents"])
    T = lbs_transforms(G, jnp.asarray(r["joints"]))[None]
    v0 = jnp.asarray(r["verts"])[None]
    dense = lbs(v0, jnp.zeros_like(v0), T, jnp.asarray(r["weights"]))
    idx, val = topk_skinning(r["weights"], k=J)
    sparse = lbs_sparse(v0, jnp.zeros_like(v0), T, jnp.asarray(val), jnp.asarray(idx))
    np.testing.assert_allclose(np.asarray(sparse), np.asarray(dense), atol=1e-5)
    np.testing.assert_allclose(np.asarray(val).sum(1), 1.0, atol=1e-5)


def test_fk_lbs_jit_vmap_grad(synthetic_rig):
    r = synthetic_rig
    J = len(r["parents"])
    rest_j, v0, W = jnp.asarray(r["joints"]), jnp.asarray(r["verts"]), jnp.asarray(r["weights"])

    def energy(aa):
        G = forward_kinematics(jax.vmap(axis_angle_to_rotmat)(aa), rest_j, r["parents"])
        v = lbs(v0[None], jnp.zeros_like(v0)[None], lbs_transforms(G, rest_j)[None], W)
        return jnp.sum(v[0, :, 1] ** 2)

    aa = jnp.asarray(RNG.normal(size=(J, 3)) * 0.3, jnp.float32)
    batched = jax.jit(jax.vmap(energy))(jnp.stack([aa, -aa]))
    np.testing.assert_allclose(float(batched[0]), float(energy(aa)), rtol=1e-5)
    g = jax.grad(energy)(aa)
    assert np.isfinite(np.asarray(g)).all() and float(jnp.abs(g).sum()) > 0
    # Central difference on one coordinate (float32 -> loose tolerance).
    eps, k = 1e-2, (3, 1)
    d = jnp.zeros_like(aa).at[k].set(eps)
    fd = (energy(aa + d) - energy(aa - d)) / (2 * eps)
    np.testing.assert_allclose(float(g[k]), float(fd), rtol=5e-2, atol=1e-3)


# ------------------------------------------------------------------ RBF + skeleton fit
@pytest.mark.parametrize("kernel", ["linear", "thin_plate_spline", "cubic", "gaussian"])
def test_host_rbf_weights_match_jax(kernel, synthetic_rig):
    pts = synthetic_rig["verts"][:60].astype(np.float32)
    q = synthetic_rig["joints"][3].astype(np.float32)
    w_np = rbf_basis_weights_np(pts, q, kernel=kernel)
    w_jx = np.asarray(RadialBasisFunction(jnp.asarray(pts), kernel=kernel)
                      .get_basis_weights(jnp.asarray(q)))
    assert w_np.dtype == np.float32 and w_np.shape == (60,)
    scale = max(1.0, float(np.abs(w_jx).max()))
    assert float(np.abs(w_np - w_jx).max()) < 1e-3 * scale


def test_rbf_reproduces_linear_maps(synthetic_rig):
    """With polynomial augmentation the interpolant is exact for affine motion."""
    pts = synthetic_rig["verts"][:80]
    q = synthetic_rig["joints"][:5]
    A = np.asarray(axis_angle_to_rotmat(jnp.array([0.2, 0.1, -0.3]))) * 1.1
    moved = pts @ A.T + np.array([0.1, -0.2, 0.05], np.float32)
    rbf = RadialBasisFunction(jnp.asarray(pts), kernel="linear")
    got = np.asarray(rbf.interpolate(jnp.asarray(moved, jnp.float32), jnp.asarray(q)))
    np.testing.assert_allclose(got, q @ A.T + np.array([0.1, -0.2, 0.05]), atol=2e-3)


def test_skeleton_transfer_refits_its_own_bind(synthetic_rig):
    r = synthetic_rig
    st = SkeletonTransfer(r["parents"].copy(), r["bind_world"], r["verts"], r["weights"])
    fit = np.asarray(st.fit(jnp.asarray(r["verts"])[None]))[0]
    np.testing.assert_allclose(fit[:, :3, 3], r["joints"], atol=2e-3)
    np.testing.assert_allclose(fit[:, :3, :3], r["bind_world"][:, :3, :3], atol=2e-3)


def test_skeleton_transfer_matrix_and_per_joint_paths_agree(synthetic_rig):
    r = synthetic_rig
    rng = np.random.default_rng(1)
    target = r["verts"] * 1.07 + rng.normal(size=r["verts"].shape).astype(np.float32) * 0.003
    fast = SkeletonTransfer(r["parents"].copy(), r["bind_world"], r["verts"], r["weights"])
    slow = SkeletonTransfer(r["parents"].copy(), r["bind_world"], r["verts"], r["weights"],
                            use_sparse_rbf_matrix=False)
    assert fast._joint_pos_regressors is None          # built lazily, never needed here
    a = np.asarray(fast.fit_joint_positions(jnp.asarray(target)))
    # The lazy regressors are first built under jit here; they must come out
    # concrete, or the eager call after it raises UnexpectedTracerError.
    b_jit = np.asarray(jax.jit(slow.fit_joint_positions)(jnp.asarray(target)))
    b = np.asarray(slow.fit_joint_positions(jnp.asarray(target)))
    np.testing.assert_allclose(a, b, atol=5e-4)
    np.testing.assert_allclose(b_jit, b, atol=1e-5)
    assert slow._joint_pos_regressors is not None


@pytest.mark.parametrize("method", ["auto", "newton-schulz", "kabsch"])
@pytest.mark.parametrize("case", ["zero", "rank1", "rank2", "generic"])
def test_rotation_from_covariance_gradient_is_finite(method, case):
    """jnp.where evaluates both branches: a degenerate covariance must not
    poison the gradient (the skeleton fit builds all-zero ones for joints
    without support)."""
    from soma_jax.geometry.transforms import rotation_from_covariance
    rng = np.random.default_rng(7)
    u, v = rng.normal(size=3), rng.normal(size=3)
    H = {"zero": np.zeros((3, 3)), "rank1": np.outer(u, v),
         "rank2": np.outer(u, v) + np.outer(v, rng.normal(size=3)),
         "generic": rng.normal(size=(3, 3))}[case]
    W = jnp.arange(9.0).reshape(3, 3)
    g = jax.grad(lambda h: jnp.sum(rotation_from_covariance(h, method=method) * W))(
        jnp.asarray(H, jnp.float32))
    assert np.isfinite(np.asarray(g)).all()


def test_skeleton_fit_is_differentiable(synthetic_rig):
    r = synthetic_rig
    st = SkeletonTransfer(r["parents"].copy(), r["bind_world"], r["verts"], r["weights"])
    target = jnp.asarray(r["verts"] * 1.05)[None]
    g = jax.grad(lambda v: jnp.sum(st.fit(v)[..., :3, :] ** 2))(target)
    assert np.isfinite(np.asarray(g)).all()


def test_warp_backend_reports_honestly():
    from soma_jax.geometry import warp_kabsch
    if jax.default_backend() != "gpu":
        assert not warp_kabsch.is_available()
        assert warp_kabsch.unavailable_reason()
    if not warp_kabsch.is_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            SkeletonTransfer(np.array([-1, 0]), np.tile(np.eye(4), (2, 1, 1)),
                             np.zeros((4, 3), np.float32), np.ones((4, 2), np.float32) / 2,
                             rotation_backend="warp")


# ------------------------------------------------------------------ misc
def test_chamfer_distance():
    x = jnp.asarray(RNG.normal(size=(50, 3)), jnp.float32)
    assert float(chamfer_distance(x, x)) < 1e-10
    d1 = float(chamfer_distance(x, x + 0.1))
    d2 = float(chamfer_distance(x + 0.1, x))
    assert d1 > 0 and abs(d1 - d2) < 1e-6


def test_barycentric_transfer_reproduces_surface_points():
    trimesh = need_module("trimesh")
    from soma_jax.geometry import barycentric_interpolate, compute_barycentric_coords
    mesh = trimesh.creation.icosphere(subdivisions=2)
    verts, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces)
    query = verts[::5] * 0.999
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        face_ids, coords = compute_barycentric_coords(query, verts, faces)
    fallback = [w for w in caught if "closest-face search unavailable" in str(w.message)]
    if fallback:   # trimesh is here but its proximity query is not (no rtree)
        pytest.fail(f"{fallback[0].message} — the 'vis' extra installs rtree")
    got = np.asarray(barycentric_interpolate(jnp.asarray(verts), jnp.asarray(faces),
                                             jnp.asarray(face_ids), jnp.asarray(coords)))
    np.testing.assert_allclose(got, query, atol=2e-2)


# ------------------------------------------------------------- pose inversion
def _pose_inversion_case(synthetic_rig, hinge=None):
    from soma_jax import PoseInversion
    from soma_jax.pose_inversion import _build_fk_lbs_fn
    r = synthetic_rig
    parents = r["parents"]
    inv = PoseInversion(rest_verts=jnp.asarray(r["verts"]), weights=jnp.asarray(r["weights"]),
                        rest_joints=jnp.asarray(r["joints"]), parents=parents,
                        dof_constraints=hinge)
    fk_lbs = _build_fk_lbs_fn(inv.rest_verts, inv.weights, inv.rest_joints, inv.parents,
                              inv.skeleton_levels)
    aa = np.zeros((len(parents), 3), np.float32)
    aa[2] = [0.3, 0.2, 0.5]          # the hinge joint's parent is rotated off-axis
    aa[3] = [-0.8, 0.0, 0.0]         # a pure X hinge relative to that parent
    aa[6] = [0.1, -0.4, 0.2]
    R_local = jax.vmap(axis_angle_to_rotmat)(jnp.asarray(aa))
    return inv, fk_lbs, R_local, fk_lbs(R_local)


def _world_to_local(Rw, parents):
    return jnp.stack([Rw[j] if parents[j] < 0 else Rw[parents[j]].T @ Rw[j]
                      for j in range(len(parents))])


@pytest.mark.parametrize("mode", ["analytical", "combined"])
def test_pose_inversion_hinge_is_parent_relative(synthetic_rig, mode):
    """A 1-DOF hinge constrains the joint relative to its parent: on a pose
    that satisfies it, enabling it must not pull the joint off the pose."""
    X = jnp.array([1.0, 0.0, 0.0])
    inv, _, R_gt, posed = _pose_inversion_case(synthetic_rig, hinge={3: X})
    R = inv.fit(posed, mode=mode, num_refine_iters=300, lr=5e-3)
    local = _world_to_local(R, synthetic_rig["parents"])
    # The constrained joint is an exact X rotation relative to its parent ...
    np.testing.assert_allclose(np.asarray(local[3] @ X), np.asarray(X), atol=1e-5)
    if mode == "combined":
        # ... and, refined, the right one (projecting the world rotation
        # instead left it ~0.4 off here).
        assert float(jnp.abs(local[3] - R_gt[3]).max()) < 0.1


def test_pose_inversion_hierarchical_improves_on_its_init(synthetic_rig):
    inv, fk_lbs, _, posed = _pose_inversion_case(synthetic_rig)
    parents = synthetic_rig["parents"]

    def err(Rw):
        return float(jnp.linalg.norm(fk_lbs(_world_to_local(Rw, parents)) - posed, -1).mean())
    init = err(inv.fit(posed, mode="analytical"))
    hier = inv.fit_hierarchical(posed, num_refine_iters=60, lr=5e-3)
    assert hier.shape == (len(parents), 3, 3)
    assert err(hier) < 0.5 * init, (err(hier), init)


def test_pose_mirror_pairs_names_and_takes_numpy():
    from soma_jax import PoseMirror
    names = ["root", "left_leg", "right_leg", "LeftHand", "RightHand",
             "l_hand", "r_hand", "arm_l", "arm_r", "spine"]
    pm = PoseMirror(names)
    pairs = {(names[i], names[j]) for i, j in enumerate(pm.mirror_indices) if i < j}
    assert pairs == {("left_leg", "right_leg"), ("LeftHand", "RightHand"),
                     ("l_hand", "r_hand"), ("arm_l", "arm_r")}
    R = np.asarray(_random_rotmats(2 * len(names))).reshape(2, len(names), 3, 3)
    m = pm.mirror_rotmats(R)                            # NumPy in, NumPy out
    assert isinstance(m, np.ndarray) and m.shape == R.shape
    np.testing.assert_allclose(pm.mirror_rotmats(m), R, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pm.mirror_rotmats_jax(jnp.asarray(R))), m, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-5)   # still rotations
