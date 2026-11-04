"""Portable self-test for a SOMA-JAX install.

Unlike the developer suite in ``tests/`` (kept local, see
``docs/DESCRIPTION.md#testing``), this one ships with the repository and checks
that *this* machine can install and run SOMA-JAX: package imports, asset
resolution (including Git LFS stubs), the geometry kernels on synthetic data,
the full layer on the real assets, and the command-line tools::

    python -m pytest tools/selftest -q                 # everything available
    python -m pytest tools/selftest -q -m "not slow"   # ~1 minute

Tests that need the model assets skip when they are missing (fetch them with
``python tools/download_assets.py --minimal``); tests that need optional
packages (usd-core, trimesh, pyrender, torch) skip without them. Set
``SOMA_JAX_SELFTEST_ASSETS=0`` to skip every asset-backed test.

JAX runs on its default backend so the check covers the accelerator a user
actually has; ``SOMA_JAX_TEST_PLATFORM=cpu`` (or ``gpu``) pins it. Matmul
precision is pinned to ``highest`` so tolerances hold on TF32-capable GPUs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"

_platform = os.environ.get("SOMA_JAX_TEST_PLATFORM", "").strip().lower()
if _platform:
    os.environ["JAX_PLATFORMS"] = {"gpu": "cuda"}.get(_platform, _platform)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

for _p in (str(REPO), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: builds the full layer or runs a tool")
    config.addinivalue_line("markers", "assets: needs the SOMA model assets")


def pytest_report_header(config):
    import numpy as np
    lines = [f"soma-jax selftest: jax {jax.__version__} on {jax.default_backend()} "
             f"({', '.join(str(d) for d in jax.devices())}), numpy {np.__version__}"]
    from soma_jax import assets
    missing = assets.missing_assets()
    lines.append("assets: all required present" if not missing
                 else f"assets: {len(missing)} required file(s) missing — asset tests may skip")
    return lines


# --------------------------------------------------------------------------- assets
from ._helpers import UPSTREAM_ASSETS, need_assets, need_module  # noqa: E402


@pytest.fixture(scope="session")
def layer_procedural():
    """Upstream's default 122-joint procedural rig, built from the upstream assets."""
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    from soma_jax import SOMALayer
    return SOMALayer.from_upstream_assets()


@pytest.fixture(scope="session")
def layer_plain():
    """The 78-joint non-procedural rig, built from the upstream assets."""
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    from soma_jax import SOMALayer
    return SOMALayer.from_upstream_assets(procedural=False)


# ------------------------------------------------------------------ synthetic rig
@pytest.fixture(scope="session")
def synthetic_rig():
    """A small branching skeleton with a point cloud skinned to it."""
    import numpy as np
    rng = np.random.default_rng(0)
    parents = np.array([-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9], dtype=np.int64)
    J = len(parents)
    offsets = rng.normal(size=(J, 3)) * 0.12
    offsets[0] = 0.0
    joints = np.zeros((J, 3))
    for j in range(J):
        joints[j] = offsets[j] if parents[j] < 0 else joints[parents[j]] + offsets[j]
    V = 600
    owner = rng.integers(1, J, V)
    verts = joints[owner] + rng.normal(size=(V, 3)) * 0.03
    W = np.zeros((V, J))
    W[np.arange(V), owner] = 1.0
    W[np.arange(V), parents[owner]] += 0.6
    W += rng.random((V, J)) ** 8 * 0.05
    W /= W.sum(1, keepdims=True)
    bind_world = np.tile(np.eye(4), (J, 1, 1))
    bind_world[:, :3, 3] = joints
    return {
        "parents": parents, "joints": joints.astype(np.float32),
        "verts": verts.astype(np.float32), "weights": W.astype(np.float32),
        "bind_world": bind_world.astype(np.float32),
    }
