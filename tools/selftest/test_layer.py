"""The full SOMA layer on the real assets: construction routes, forward, transforms."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from soma_jax import SOMALayer, SOMAParams

from ._helpers import UPSTREAM_ASSETS, need_assets, need_module

pytestmark = [pytest.mark.assets, pytest.mark.slow]


def _params(layer, B=2, sigma=0.3, seed=0, id_scale=0.5):
    rng = np.random.default_rng(seed)
    J = len(layer.public_joint_names)
    poses = np.zeros((B, J, 3), np.float32)
    poses[:, 1:] = rng.standard_normal((B, J - 1, 3)) * sigma
    ids = (rng.standard_normal((B, 128)) * id_scale).astype(np.float32)
    return SOMAParams(poses=jnp.asarray(poses), transl=jnp.zeros((B, 3)),
                      identity_coeffs=jnp.asarray(ids))


def test_readme_quickstart(layer_procedural):
    """The README "Usage" snippet, verbatim apart from the layer fixture."""
    import equinox as eqx
    layer = layer_procedural
    B, J = 4, len(layer.public_joint_names)
    assert J == 78
    out = eqx.filter_jit(layer)(SOMAParams(
        poses=jnp.zeros((B, J, 3)), transl=jnp.zeros((B, 3)),
        identity_coeffs=jnp.zeros((B, 128))))
    assert out.vertices.shape == (B, 18056, 3)
    assert out.joints.shape == (B, 78, 3)
    assert np.isfinite(np.asarray(out.vertices)).all()
    height = float(out.vertices[0, :, 1].max() - out.vertices[0, :, 1].min())
    assert 1.4 < height < 2.1, f"implausible body height {height:.2f} m"


def test_procedural_rig_is_active(layer_procedural, layer_plain):
    """Equal at rest; the twist joints move the posed surface."""
    rest = _params(layer_plain, sigma=0.0)
    a = np.asarray(layer_procedural(rest).vertices)
    b = np.asarray(layer_plain(rest).vertices)
    assert float(np.abs(a - b).max()) < 1e-4
    posed = _params(layer_plain, sigma=0.45)
    gap = np.linalg.norm(np.asarray(layer_procedural(posed).vertices)
                         - np.asarray(layer_plain(posed).vertices), axis=-1)
    assert float(gap.max()) > 5e-3, "procedural twist joints had no effect"


def test_jit_matches_eager_and_batches_are_independent(layer_plain):
    import equinox as eqx
    p = _params(layer_plain, B=3)
    eager = np.asarray(layer_plain(p).vertices)
    jitted = np.asarray(eqx.filter_jit(layer_plain)(p).vertices)
    np.testing.assert_allclose(jitted, eager, atol=1e-5)
    one = SOMAParams(poses=p.poses[1:2], transl=p.transl[1:2],
                     identity_coeffs=p.identity_coeffs[1:2])
    np.testing.assert_allclose(np.asarray(layer_plain(one).vertices[0]), eager[1], atol=1e-5)


@pytest.mark.parametrize("rig", ["layer_plain", "layer_procedural"])
def test_77_posable_joints_and_joint_count_errors(request, rig):
    """Upstream poses the 77 posable joints (Root never moves): every rig
    takes that as well as the 78 public joints, and says so on a mismatch."""
    layer = request.getfixturevalue(rig)
    p = _params(layer, B=1)                  # Root (joint 0) is left at zero
    full = np.asarray(layer(p).vertices)
    posable = np.asarray(layer(p._replace(poses=p.poses[:, 1:])).vertices)
    np.testing.assert_allclose(posable, full, atol=1e-6)
    with pytest.raises(ValueError, match="public joints"):
        layer(p._replace(poses=p.poses[:, :50]))


def test_rebind_keeps_the_skeleton_fit_consistent(layer_plain):
    """Rebinding to the current bind shape changes nothing; a new shape takes
    its own joints and leaves the original layer alone."""
    p = _params(layer_plain)
    ref = np.asarray(layer_plain(p).vertices)
    bind_shape = layer_plain.skeleton_transfer.bind_shape
    same = layer_plain.rebind(bind_shape)
    np.testing.assert_allclose(np.asarray(same(p).vertices), ref, atol=1e-5)
    # A different template takes the joints the skeleton fit gives it: the
    # identity whose rest shape it is (here the zero identity, whose rest is
    # v_template) poses exactly as before — keeping the old joints moved it
    # by ~10 cm — and other identities shift a little.
    moved = layer_plain.rebind(layer_plain.v_template)
    zero = p._replace(identity_coeffs=jnp.zeros_like(p.identity_coeffs))
    np.testing.assert_allclose(np.asarray(moved(zero).vertices),
                               np.asarray(layer_plain(zero).vertices), atol=1e-5)
    assert float(np.abs(np.asarray(moved(p).vertices) - ref).max()) > 1e-3
    np.testing.assert_array_equal(np.asarray(layer_plain(p).vertices), ref)


def test_gradients_flow_end_to_end(layer_plain):
    p = _params(layer_plain, B=1)

    def loss(poses, ids):
        out = layer_plain(SOMAParams(poses=poses, transl=p.transl, identity_coeffs=ids))
        return jnp.mean(out.vertices ** 2)

    g_pose, g_id = jax.grad(loss, argnums=(0, 1))(p.poses, p.identity_coeffs)
    for g in (g_pose, g_id):
        g = np.asarray(g)
        assert np.isfinite(g).all() and np.abs(g).sum() > 0


def test_low_lod(layer_plain):
    low = SOMALayer.from_upstream_assets(lod="low")
    out = low(_params(low))
    assert out.vertices.shape[1] == 4505
    assert np.isfinite(np.asarray(out.vertices)).all()


def test_fit_joint_regressor_false_still_builds():
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    layer = SOMALayer.from_upstream_assets(procedural=False, fit_joint_regressor=False)
    assert layer.J_regressor.shape == (78, 18056)


def test_explicit_npz_path_is_used(tmp_path):
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    with pytest.raises(FileNotFoundError):
        SOMALayer.from_upstream_assets(npz_path=str(tmp_path / "missing" / "SOMA_neutral.npz"))


def test_load_rejects_upstream_archive_with_guidance():
    need_assets("SOMA_neutral.npz")
    from soma_jax.assets import resolve
    with pytest.raises(ValueError, match="from_upstream_assets"):
        SOMALayer.load(str(resolve("SOMA_neutral.npz")))


def test_prebuilt_archive_route_agrees(layer_plain):
    """If a §4.2 archive was baked, it and the torch-free route give one model."""
    from soma_jax.assets import resolve
    fixed = resolve("SOMA_neutral_fixed.npz", required=False)
    if fixed is None:
        pytest.skip("no SOMA_neutral_fixed.npz (optional, docs/INSTALL.md §4.2)")
    p = _params(layer_plain, sigma=0.3)
    a = np.asarray(SOMALayer.load(str(fixed))(p).vertices)
    b = np.asarray(layer_plain(p).vertices)
    assert float(np.abs(a - b).max()) < 2e-5


def test_bone_scales_change_the_skeleton(layer_plain):
    rest, joints, binds = layer_plain.prepare_identity(jnp.zeros((1, 128)),
                                                       return_bind_transforms=True)
    R = jnp.broadcast_to(jnp.eye(3), (1, len(layer_plain.joint_names), 3, 3))
    scales = jnp.ones((1, layer_plain.num_bone_scale_params))
    base = layer_plain.pose(R, jnp.zeros((1, 3)), rest, joints, bind_transforms=binds,
                            bone_scales=scales, fk_only=True)
    i = layer_plain.scale_param_names.index("LeftForeArm")
    longer = layer_plain.pose(R, jnp.zeros((1, 3)), rest, joints, bind_transforms=binds,
                              bone_scales=scales.at[0, i].set(1.5), fk_only=True)
    assert float(jnp.abs(longer.joints - base.joints).max()) > 1e-3


def test_animation_npz_roundtrip(tmp_path, layer_plain):
    from soma_jax import load_soma_npz, save_soma_npz
    T, J = 5, len(layer_plain.joint_names)
    poses = np.random.default_rng(0).standard_normal((T, J, 3)).astype(np.float32) * 0.1
    path = tmp_path / "clip.soma.npz"
    save_soma_npz(str(path), poses, np.zeros((T, 3), np.float32), layer_plain.joint_names,
                  "soma", np.zeros(128, np.float32))
    back = load_soma_npz(str(path))
    assert back["poses"].shape[0] == T
    assert list(map(str, back["joint_names"]))[-J + 1:] == layer_plain.joint_names[-J + 1:]


def test_pose_inversion_roundtrip(layer_procedural):
    """README "Recover pose from a posed mesh"."""
    from soma_jax import SOMAPoseInversion
    p = _params(layer_procedural, B=1, sigma=0.2, seed=3)
    posed = layer_procedural(p).vertices
    inv = SOMAPoseInversion(layer_procedural)
    inv.prepare_identity(p.identity_coeffs)
    result = inv.fit(posed)
    err = np.asarray(result.per_vertex_error)
    assert result.rotations.shape == (1, 78, 3, 3)
    assert float(err.mean()) < 5e-3 and float(err.max()) < 6e-2, (err.mean(), err.max())


def test_usd_export_roundtrip(tmp_path, layer_plain):
    need_module("pxr")
    from soma_jax import export_soma_usd, load_usd_skeleton
    rest, _, binds = layer_plain.prepare_identity(jnp.zeros((1, 128)),
                                                  return_bind_transforms=True)
    T, J = 3, len(layer_plain.joint_names)
    rot = np.broadcast_to(np.eye(3, dtype=np.float32), (T, J, 3, 3)).copy()
    out = tmp_path / "anim.usda"
    export_soma_usd(str(out), layer_plain, rot, np.zeros((T, 3), np.float32),
                    bind_transforms_world=binds[0], rest_shape=rest[0], fps=30.0)
    joint_paths, bind_tf, parents = load_usd_skeleton(str(out))
    assert len(joint_paths) == J and np.asarray(bind_tf).shape[0] == J


@pytest.mark.parametrize("backend,builder", [("anny", "build_anny_pack"),
                                             ("garment_measurement", "build_garment_pack")])
def test_identity_backend(tmp_path, backend, builder, layer_plain):
    """Topology transfer + a non-SOMA identity backend (packs need no torch)."""
    need_module("trimesh")
    need_assets("Anny/base_body.obj", "Anny/SOMA_wrap.obj", "GarmentMeasurements/mean.obj",
                "GarmentMeasurements/SOMA_wrap.obj", "GarmentMeasurements/point.npz")
    import importlib.util
    from soma_jax.assets import REPO_ROOT, data_root
    spec = importlib.util.spec_from_file_location(
        "_bip", REPO_ROOT / "tools" / "pipeline" / "build_identity_packs.py")
    bip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bip)
    pack = tmp_path / f"identity_{backend}.npz"
    try:
        getattr(bip, builder)(data_root(), pack)
    except ModuleNotFoundError as e:      # trimesh < 5 needs rtree
        pytest.fail(f"{e} — the 'vis' extra installs rtree")
    layer = SOMALayer.from_upstream_assets(procedural=False, identity_model_type=backend,
                                           identity_model_path=str(pack))
    # First use under jit: the identity model builds lazy caches (the
    # Laplacian blend) on first call, and a cache built inside a trace would
    # hold tracers and break every later call with UnexpectedTracerError.
    ids = jnp.asarray(np.random.default_rng(0).standard_normal((1, 16)) * 0.5, jnp.float32)
    jitted = np.asarray(jax.jit(lambda c: layer.prepare_identity(c)[0])(ids))
    out = layer(SOMAParams(poses=jnp.zeros((1, 78, 3)), transl=jnp.zeros((1, 3)),
                           identity_coeffs=ids))
    v = np.asarray(out.vertices)
    assert v.shape == (1, 18056, 3) and np.isfinite(v).all()
    assert 0.9 < float(v[0, :, 1].max() - v[0, :, 1].min()) < 2.3
    np.testing.assert_allclose(np.asarray(layer.prepare_identity(ids)[0]), jitted, atol=1e-4)
    if backend == "garment_measurement":
        # Upstream's body: pca_mean + (coeffs * sqrt(eigenvalues)) @ pca_matrix.T
        # (a pack without the eigenvalues was 36 mm off it on average).
        d = np.load(data_root() / "GarmentMeasurements" / "point.npz")
        c = np.asarray(ids[:, :15], np.float64)
        want = (d["pca_mean"] + (c * np.sqrt(d["eigenvalues"])) @ d["pca_matrix"].T)
        got = np.asarray(layer.identity_model.get_rest_shape(ids[:, :15]))
        np.testing.assert_allclose(got.reshape(want.shape), want, atol=2e-5)
