"""Skip helpers shared by the self-test modules."""
from __future__ import annotations

import os

import pytest

#: What ``SOMALayer.from_upstream_assets()`` reads.
UPSTREAM_ASSETS = ("SOMA_neutral.npz", "SOMA_template_rig.usda",
                   "SOMA_procedural_transforms.json")


def assets_enabled() -> bool:
    return os.environ.get("SOMA_JAX_SELFTEST_ASSETS", "1").lower() not in ("0", "false", "no")


def need_assets(*names: str) -> None:
    """Skip unless every named asset resolves (and asset tests are enabled)."""
    if not assets_enabled():
        pytest.skip("SOMA_JAX_SELFTEST_ASSETS=0")
    from soma_jax.assets import resolve
    missing = [n for n in names if resolve(n, required=False) is None]
    if missing:
        fetch = ("--minimal" if set(missing) <= set(UPSTREAM_ASSETS)
                 else "--only " + " ".join(missing))
        pytest.skip(f"assets missing: {', '.join(missing)} "
                    f"(python tools/download_assets.py {fetch})")


def need_module(name: str):
    """Import an optional package or skip."""
    return pytest.importorskip(name, reason=f"optional package {name!r} not installed")
