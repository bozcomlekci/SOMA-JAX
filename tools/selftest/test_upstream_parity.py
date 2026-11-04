"""Against upstream SOMA-X itself — only where torch, warp-lang and the
``third_party/SOMA-X`` submodule are available (it skips otherwise).

Both sides on CPU, upstream in its pure-torch ``mode="dense"``. Tolerances are
the developer suite's 5e-5 m guard; measured agreement is a few micrometres.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from soma_jax import SOMALayer, SOMAParams

from ._helpers import UPSTREAM_ASSETS, need_assets, need_module

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.assets, pytest.mark.slow]
B = 2


@pytest.fixture(scope="module")
def upstream():
    torch = need_module("torch")
    need_module("warp")
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS, "correctives_model.pt")
    sys.path.append(str(REPO / "third_party" / "SOMA-X"))
    try:
        from soma.soma import SOMALayer as TorchSOMA
    except ImportError as e:
        pytest.skip(f"upstream SOMA-X not importable: {e}")
    from soma_jax.assets import data_root
    root = str(data_root())
    cache = {}

    def build(procedural: bool, correctives: bool):
        key = (procedural, correctives)
        if key not in cache:
            kw = {} if correctives else {"correctives_model_path": None}
            cache[key] = TorchSOMA(data_root=root, identity_model_type="soma", device="cpu",
                                   mode="dense", enable_procedural_transforms=procedural, **kw)
        return cache[key]

    def run(procedural, poses77, ids, correctives=False):
        with torch.no_grad():
            out = build(procedural, correctives).forward(
                poses=torch.from_numpy(poses77), identity_coeffs=torch.from_numpy(ids),
                transl=torch.zeros(B, 3), apply_correctives=correctives)
        return out["vertices"].numpy()
    return run


def _inputs(sigma, seed=0):
    rng = np.random.default_rng(seed)
    poses = (rng.standard_normal((B, 77, 3)) * sigma).astype(np.float32)
    ids = (rng.standard_normal((B, 128)) * 0.5).astype(np.float32)
    return poses, ids


def _ours(layer, poses77, ids, correctives=False):
    poses = np.concatenate([np.zeros((B, 1, 3), np.float32), poses77], axis=1)
    out = layer(SOMAParams(poses=jnp.asarray(poses), transl=jnp.zeros((B, 3)),
                           identity_coeffs=jnp.asarray(ids)), apply_correctives=correctives)
    return np.asarray(out.vertices)


@pytest.mark.parametrize("sigma", [0.0, 0.45])
def test_procedural_forward_matches_upstream(upstream, layer_procedural, sigma):
    poses, ids = _inputs(sigma)
    err = np.linalg.norm(_ours(layer_procedural, poses, ids) - upstream(True, poses, ids), axis=-1)
    assert float(err.max()) < 5e-5, f"max {err.max() * 1e3:.4f} mm"


@pytest.mark.parametrize("sigma", [0.0, 0.45])
def test_plain_forward_matches_upstream(upstream, layer_plain, sigma):
    poses, ids = _inputs(sigma)
    err = np.linalg.norm(_ours(layer_plain, poses, ids) - upstream(False, poses, ids), axis=-1)
    assert float(err.max()) < 5e-5, f"max {err.max() * 1e3:.4f} mm"


def test_correctives_match_upstream(upstream, tmp_path):
    """The shipped corrective checkpoint, converted, on upstream's default rig."""
    from soma_jax.assets import resolve
    npz = tmp_path / "correctives.npz"
    subprocess.run([sys.executable, str(REPO / "tools" / "convert" /
                                        "convert_correctives_pt_to_npz.py"),
                    str(resolve("correctives_model.pt")), str(npz)],
                   check=True, capture_output=True, timeout=900)
    layer = SOMALayer.from_upstream_assets(correctives_path=str(npz))
    poses, ids = _inputs(0.45, seed=1)
    ref = upstream(True, poses, ids, correctives=True)
    effect = np.linalg.norm(ref - upstream(True, poses, ids), axis=-1)
    assert float(effect.max()) > 5e-3, "upstream correctives had no effect"
    err = np.linalg.norm(_ours(layer, poses, ids, correctives=True) - ref, axis=-1)
    assert float(err.max()) < 1e-4, f"max {err.max() * 1e3:.4f} mm"
