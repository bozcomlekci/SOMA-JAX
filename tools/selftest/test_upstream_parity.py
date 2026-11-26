"""Against upstream SOMA-X itself — the pinned ``third_party/SOMA-X`` (v0.3.6).

Runs only where torch, warp-lang, usd-core and the submodule are available,
and skips otherwise (CI's ``parity`` job installs them). Both sides run on CPU,
upstream in its pure-torch ``mode="dense"`` path, on identical inputs. Each
test asserts a bound a few times above the agreement measured when it was
written, so it holds across BLAS and driver versions; ``docs/FAITHFULNESS.md``
quotes the measured figures.
"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from soma_jax import SOMALayer

from ._helpers import UPSTREAM_ASSETS, need_assets, need_module

REPO = Path(__file__).resolve().parents[2]
SUBMODULE = REPO / "third_party" / "SOMA-X"
# Memory-heavy (each upstream layer retains ~0.5 GB of RBF systems; ~7 GB peak
# in one process): under pytest-xdist pass --dist loadgroup to keep the module
# on one worker.
pytestmark = [pytest.mark.assets, pytest.mark.slow, pytest.mark.xdist_group("upstream_parity")]
B = 2
HAND_ASSETS = ("SOMAHand.npz", "MANO/base_hand_left.obj", "MANO/base_hand_right.obj",
               "MANO/SOMA_wrap_left.obj", "MANO/SOMA_wrap_right.obj")


# ---------------------------------------------------------------- fixtures
def _required(check):
    """Run a skip-or-pass check; with SOMA_JAX_REQUIRE_PARITY=1 (CI's parity
    job) a skip is a failure, so the job cannot pass without running."""
    try:
        return check()
    except pytest.skip.Exception as skip:
        if os.environ.get("SOMA_JAX_REQUIRE_PARITY") == "1":
            pytest.fail(f"parity required but skipped: {skip}")
        raise


@pytest.fixture(scope="module")
def up():
    """Upstream's package from the submodule, plus layer caches for both sides."""
    torch = _required(lambda: need_module("torch"))
    _required(lambda: need_module("warp"))
    _required(lambda: need_module("pxr"))
    _required(lambda: need_assets(*UPSTREAM_ASSETS, "correctives_model.pt"))
    if str(SUBMODULE) not in sys.path:
        sys.path.append(str(SUBMODULE))

    def _import():
        try:
            import soma
        except ImportError as e:
            pytest.skip(f"upstream SOMA-X not importable: {e}")
        if SUBMODULE not in Path(soma.__file__).resolve().parents:
            pytest.skip(f"another soma package is installed ({soma.__file__}); "
                        "the parity tests need the pinned submodule's")
        return soma
    soma = _required(_import)
    from soma_jax.assets import data_root

    class Upstream:
        pass

    u = Upstream()
    u.torch, u.soma, u.root = torch, soma, str(data_root())
    # A few layers per side at a time: every LOD x rig x correctives at once
    # would hold eighteen layers in memory.
    u.layers, u.ours = collections.OrderedDict(), collections.OrderedDict()

    def _cached(cache, key, build, size=2):
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = build()
            while len(cache) > size:
                cache.popitem(last=False)
        return cache[key]

    def layer(procedural=True, lod="mid", correctives=True):
        kw = {} if correctives else {"correctives_model_path": None}
        return _cached(u.layers, (procedural, lod, correctives), lambda: soma.SOMALayer(
            data_root=u.root, identity_model_type="soma", device="cpu", mode="dense",
            enable_procedural_transforms=procedural, lod=lod, **kw))

    def ours(procedural=True, lod="mid", correctives=True):
        kw = {} if correctives else {"correctives_model_path": None}
        return _cached(u.ours, (procedural, lod, correctives), lambda: (
            SOMALayer.from_upstream_assets(procedural=procedural, lod=lod, **kw)))

    u.layer, u.ours_layer = layer, ours
    yield u
    u.layers.clear()
    u.ours.clear()


def _inputs(sigma, seed=0, batch=B):
    rng = np.random.default_rng(seed)
    poses = (rng.standard_normal((batch, 77, 3)) * sigma).astype(np.float32)
    ids = (rng.standard_normal((batch, 128)) * 0.5).astype(np.float32)
    transl = (rng.standard_normal((batch, 3)) * 0.1).astype(np.float32)
    return poses, ids, transl


def _report(name, **values):
    """One PARITY line per comparison (shown with ``pytest -s``)."""
    print(f"PARITY {name}: " + ", ".join(f"{k} {v:.2e}" for k, v in values.items()))


def _maxnorm(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=-1).max())


# ---------------------------------------------------------------- body layer
@pytest.mark.parametrize("lod", ["mid", "low", "xlo"])
@pytest.mark.parametrize("procedural,correctives",
                         [(True, True), (True, False), (False, False)],
                         ids=["procedural+correctives", "procedural", "plain"])
def test_forward_matches_upstream(up, lod, procedural, correctives):
    """forward(): every LOD, both rigs, with and without the v0.3.6 correctives."""
    poses, ids, transl = _inputs(0.45)
    with up.torch.no_grad():
        ref = up.layer(procedural, lod, correctives)(
            up.torch.from_numpy(poses), up.torch.from_numpy(ids),
            transl=up.torch.from_numpy(transl), apply_correctives=correctives)
    got = up.ours_layer(procedural, lod, correctives).forward(
        jnp.asarray(poses), jnp.asarray(ids), transl=jnp.asarray(transl),
        apply_correctives=correctives)
    tol = 1e-4 if correctives else 5e-5
    dv = _maxnorm(got["vertices"], ref["vertices"].numpy())
    dj = _maxnorm(got["joints"], ref["joints"].numpy())
    _report(f"forward lod={lod} procedural={procedural} correctives={correctives}",
            vertices=dv, joints=dj)
    assert dv < tol and dj < tol
    assert got["transforms"].shape == tuple(ref["transforms"].shape)


def test_correctives_have_an_effect(up):
    """The comparison above is not vacuous: the checkpoint moves the surface."""
    poses, ids, _ = _inputs(0.45, seed=1)
    with up.torch.no_grad():
        on = up.layer(True, "mid", True)(up.torch.from_numpy(poses), up.torch.from_numpy(ids),
                                         apply_correctives=True)["vertices"].numpy()
        off = up.layer(True, "mid", True)(up.torch.from_numpy(poses), up.torch.from_numpy(ids),
                                          apply_correctives=False)["vertices"].numpy()
    assert _maxnorm(on, off) > 5e-3


@pytest.mark.parametrize("case", ["default", "absolute", "fk_only", "reference", "bone_scales",
                                  "global_scale"])
def test_prepare_identity_and_pose_match_upstream(up, case):
    """Upstream's prepare_identity() + pose() workflow, call by call."""
    torch = up.torch
    poses, ids, transl = _inputs(0.3, seed=2)
    t_layer, j_layer = up.layer(True, "mid", True), up.ours_layer(True, "mid", True)
    prep = {}
    kw = {}
    if case == "absolute":
        kw = {"absolute_pose": True}
    elif case == "fk_only":
        kw = {"fk_only": True}
    elif case == "reference":
        kw = {"reference_pose": {"version": "v0.1.0"}}
    elif case == "bone_scales":
        scales = np.ones((B, j_layer.num_bone_scale_params), np.float32)
        scales[:, j_layer.scale_param_names.index("LeftForeArm")] = 1.4
        scales[:, j_layer.scale_param_names.index("RightShin")] = 0.8
        prep = {"scale_params": scales}
    elif case == "global_scale":
        prep = {"global_scale": 1.25}
    with torch.no_grad():
        t_layer.prepare_identity(
            torch.from_numpy(ids),
            **{k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in prep.items()})
        ref = t_layer.pose(torch.from_numpy(poses), transl=torch.from_numpy(transl), **kw)
    identity = j_layer.prepare_identity(
        jnp.asarray(ids), **{k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                             for k, v in prep.items()})
    got = j_layer.pose(jnp.asarray(poses), transl=jnp.asarray(transl), identity=identity, **kw)
    assert set(got) == set(ref)
    diffs = {key: _maxnorm(np.asarray(got[key]).reshape(-1, 3) if key != "transforms"
                           else np.asarray(got[key])[..., :3, 3],
                           ref[key].numpy().reshape(-1, 3) if key != "transforms"
                           else ref[key].numpy()[..., :3, 3]) for key in got}
    _report(f"prepare_identity + pose [{case}]", **diffs)
    assert all(d < 1e-4 for d in diffs.values()), diffs
    np.testing.assert_allclose(np.asarray(got["transforms"])[..., :3, :3],
                               ref["transforms"].numpy()[..., :3, :3], atol=1e-4)


def test_public_joint_orient_matches_upstream(up):
    for procedural in (True, False):
        a = up.layer(procedural, "mid", False).public_joint_orient()
        b = up.ours_layer(procedural, "mid", False).public_joint_orient()
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(y), x.numpy(), atol=1e-6)


# ---------------------------------------------------------------- USD motion
def _example_usd():
    _required(lambda: need_assets("example_animation.usd"))
    from soma_jax.assets import resolve
    return str(resolve("example_animation.usd"))


def test_load_soma_animation_matches_upstream(up):
    from soma_jax import load_soma_animation
    usd = _example_usd()
    names = up.ours_layer(True, "mid", False).public_joint_names
    for world_space in (False, True):
        ref, ref_fps = up.soma.io.load_soma_animation(usd, names, world_space=world_space)
        got, fps = load_soma_animation(usd, names, world_space=world_space)
        assert fps == ref_fps
        np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("procedural", [True, False])
def test_body_load_motion_matches_upstream(up, procedural):
    usd = _example_usd()
    t_layer, j_layer = up.layer(procedural, "mid", False), up.ours_layer(procedural, "mid", False)
    for kw in ({}, {"absolute_pose": True}, {"reference_pose": {"version": "v0.1.0"}}):
        tp, tt, tfps = t_layer.load_motion(usd, **kw)
        jp, jt, jfps = j_layer.load_motion(usd, **kw)
        assert jfps == tfps and jp.shape == tuple(tp.shape)
        np.testing.assert_allclose(np.asarray(jp), tp.numpy(), atol=1e-5)
        np.testing.assert_allclose(np.asarray(jt), tt.numpy(), atol=1e-6)


# ---------------------------------------------------------------- hand layer
@pytest.fixture(scope="module")
def hands(up):
    _required(lambda: need_assets(*HAND_ASSETS))
    from soma_jax.hand import SOMAHandLayer
    return {side: (up.soma.SOMAHandLayer(data_root=up.root, hand_type=side, device="cpu"),
                   SOMAHandLayer(data_root=up.root, hand_type=side))
            for side in ("left", "right")}


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("correctives", [True, False])
def test_hand_forward_matches_upstream(up, hands, side, correctives):
    """Includes v0.3.6's identity wrist for the hand correctives."""
    t_hand, j_hand = hands[side]
    rng = np.random.default_rng(3)
    poses = (rng.standard_normal((B, 25, 3)) * 0.4).astype(np.float32)
    ids = (rng.standard_normal((B, j_hand.num_shape_components)) * 0.5).astype(np.float32)
    with up.torch.no_grad():
        ref = t_hand(up.torch.from_numpy(poses), up.torch.from_numpy(ids),
                     apply_correctives=correctives)
    got = j_hand(jnp.asarray(poses), jnp.asarray(ids), apply_correctives=correctives)
    dv = _maxnorm(got["vertices"], ref["vertices"].numpy())
    dj = _maxnorm(got["joints"], ref["joints"].numpy())
    _report(f"hand {side} correctives={correctives}", vertices=dv, joints=dj)
    assert dv < 5e-5 and dj < 5e-5


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_load_motion_matches_upstream(up, hands, side):
    usd = _example_usd()
    t_hand, j_hand = hands[side]
    for kw in ({}, {"absolute_pose": True}):
        tp, tt, tfps = t_hand.load_motion(usd, **kw)
        jp, jt, jfps = j_hand.load_motion(usd, **kw)
        assert jfps == tfps
        np.testing.assert_allclose(np.asarray(jp), tp.numpy(), atol=1e-5)
        np.testing.assert_allclose(np.asarray(jt), tt.numpy(), atol=1e-6)


# ---------------------------------------------------------------- pose inversion
def test_pose_inversion_matches_upstream(up):
    """Analytical + Lie-GN inversion of the same posed meshes, both sides."""
    torch = up.torch
    from soma_jax import SOMAPoseInversion
    poses, ids, transl = _inputs(0.25, seed=4)
    t_layer, j_layer = up.layer(True, "mid", False), up.ours_layer(True, "mid", False)
    with torch.no_grad():
        posed = t_layer(torch.from_numpy(poses), torch.from_numpy(ids),
                        transl=torch.from_numpy(transl), apply_correctives=False)["vertices"]
    from soma.fitting import PoseInversion
    t_inv = PoseInversion(t_layer)
    t_inv.prepare_identity(torch.from_numpy(ids))
    ref = t_inv.fit(posed)
    j_inv = SOMAPoseInversion(j_layer)
    j_inv.prepare_identity(jnp.asarray(ids))
    got = j_inv.fit(jnp.asarray(posed.numpy()))
    # Iterative float32 solves on both sides (upstream's refit runs a fused Warp
    # kernel). Neither returns exactly orthonormal matrices (upstream's
    # determinants reach 0.995), so the rotations are compared entry by entry;
    # measured 6.4e-3 max, the fits 2.416 vs 2.409 mm mean error.
    diff = np.abs(np.asarray(got.rotations) - ref.rotations.numpy())
    _report("pose inversion", rotation_entries_max=float(diff.max()),
            rotation_entries_mean=float(diff.mean()),
            root=float(np.abs(np.asarray(got.root_translation)
                              - ref.root_translation.numpy()).max()),
            mean_error_gap=abs(float(np.asarray(got.per_vertex_error).mean())
                               - float(ref.per_vertex_error.mean())))
    assert diff.max() < 1.5e-2 and diff.mean() < 1e-3, (diff.max(), diff.mean())
    np.testing.assert_allclose(np.asarray(got.root_translation), ref.root_translation.numpy(),
                               atol=5e-4)
    assert abs(float(np.asarray(got.per_vertex_error).mean())
               - float(ref.per_vertex_error.mean())) < 2e-4


# ---------------------------------------------------------------- geometry
def test_transforms_match_upstream(up):
    torch = up.torch
    from soma.geometry import transforms as T
    from soma_jax.geometry import transforms as J
    rng = np.random.default_rng(5)
    vec = np.concatenate([rng.normal(size=(256, 3)) * 1.2, rng.normal(size=(64, 3)) * 1e-4,
                          np.zeros((1, 3))]).astype(np.float32)
    R = T.rotvec_to_matrix(torch.from_numpy(vec))
    np.testing.assert_allclose(np.asarray(J.rotvec_to_matrix(jnp.asarray(vec))), R.numpy(),
                               atol=2e-6)
    np.testing.assert_allclose(np.asarray(J.matrix_to_rotvec(jnp.asarray(R.numpy()))),
                               T.matrix_to_rotvec(R).numpy(), atol=2e-6)
    a = rng.normal(size=(128, 6, 3)).astype(np.float32)
    rot = T.rotvec_to_matrix(torch.from_numpy(rng.normal(size=(128, 3)).astype(np.float32)))
    b = np.einsum("nij,nkj->nki", rot.numpy(), a).astype(np.float32)
    for method in ("auto", "kabsch", "newton-schulz"):
        ref = T.align_vectors(torch.from_numpy(b), torch.from_numpy(a), method=method)
        got = J.align_vectors(jnp.asarray(b), jnp.asarray(a), method=method)
        np.testing.assert_allclose(np.asarray(got), ref.numpy(), atol=1e-5, err_msg=method)
    q = rng.normal(size=(64, 4)).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(J.quaternion_xyzw_to_rotmat(jnp.asarray(q))),
        T.quaternion_xyzw_to_matrix(torch.from_numpy(q)).numpy(), atol=2e-6)


# ---------------------------------------------------------------- files
def test_correctives_checkpoint_reads_without_torch(up):
    """read_torch_checkpoint() reproduces torch.load on the shipped checkpoint."""
    from soma_jax.assets import resolve
    from soma_jax.correctives_model import read_torch_checkpoint
    path = resolve("correctives_model.pt")
    ref = up.torch.load(path, map_location="cpu", weights_only=True)
    got = read_torch_checkpoint(path)
    assert set(got) == set(ref)
    for key, value in ref.items():
        if up.torch.is_tensor(value):
            dense = value.to_dense() if value.is_sparse else value
            np.testing.assert_array_equal(got[key], dense.numpy(), err_msg=key)
        else:
            assert got[key] == value, key


def test_npz_export_is_upstreams(up, tmp_path):
    """tools/conversion_utils.export_soma_npz (v0.3.6's resolve_joint_orient
    path) writes what upstream's writes, and each side reads the other's file."""
    import importlib.util
    torch = up.torch
    spec = importlib.util.spec_from_file_location("jax_cu", REPO / "tools" / "conversion_utils.py")
    jax_cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jax_cu)
    spec = importlib.util.spec_from_file_location("up_cu", SUBMODULE / "tools" / "conversion_utils.py")
    up_cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up_cu)
    from soma_jax import load_soma_npz
    for procedural in (True, False):
        t_layer = up.layer(procedural, "mid", False)
        j_layer = up.ours_layer(procedural, "mid", False)
        poses, _, transl = _inputs(0.3, seed=6, batch=6)
        R = up.soma.geometry.transforms.rotvec_to_matrix(torch.from_numpy(poses))
        R = torch.cat([torch.eye(3).expand(6, 1, 3, 3), R], dim=1)       # absolute, Root first
        a, b = tmp_path / f"up_{procedural}.npz", tmp_path / f"jax_{procedural}.npz"
        up_cu.export_soma_npz(str(a), R, torch.from_numpy(transl), t_layer, output_unit="meters",
                              identity_coeffs=torch.zeros(1, 128))
        jax_cu.export_soma_npz(str(b), R.numpy(), transl, j_layer, output_unit="meters",
                               identity_coeffs=np.zeros((1, 128), np.float32))
        ua, jb = dict(np.load(a, allow_pickle=True)), dict(np.load(b, allow_pickle=True))
        assert set(ua) == set(jb)
        for key in ua:
            if ua[key].dtype.kind in "fc":
                np.testing.assert_allclose(jb[key], ua[key], atol=1e-5, err_msg=key)
            else:
                np.testing.assert_array_equal(jb[key], ua[key], err_msg=key)
        np.testing.assert_allclose(load_soma_npz(str(a))["poses"],
                                   up.soma.io.load_soma_npz(str(b))["poses"], atol=1e-5)
