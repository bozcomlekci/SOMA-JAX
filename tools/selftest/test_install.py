"""The package installs, imports and declares itself correctly on this machine."""
from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Modules that import an optional third-party package at import time.
_OPTIONAL_AT_IMPORT = {"soma_jax.body_models.anny_native": "anny"}


def _all_modules():
    import soma_jax
    for info in pkgutil.walk_packages(soma_jax.__path__, prefix="soma_jax."):
        yield info.name


@pytest.mark.parametrize("name", sorted(_all_modules()))
def test_every_module_imports(name):
    needs = _OPTIONAL_AT_IMPORT.get(name)
    if needs:
        pytest.importorskip(needs)
    importlib.import_module(name)


@pytest.mark.parametrize("module", ["soma_jax", "soma_jax.geometry", "soma_jax.body_models"])
def test_public_names_resolve(module):
    mod = importlib.import_module(module)
    missing = [n for n in getattr(mod, "__all__", []) if not hasattr(mod, n)]
    assert not missing, f"{module}.__all__ lists undefined names: {missing}"


def test_installed_version_matches_package():
    from importlib.metadata import version
    import soma_jax
    assert version("soma-jax") == soma_jax.__version__


def test_numpy_and_jaxlib_are_abi_compatible():
    # An old jaxlib built against NumPy 1.x fails here with `_ARRAY_API not
    # found` under NumPy 2 — the reason the jax floor is 0.4.26.
    import jax.numpy as jnp
    import numpy as np
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_allclose(np.asarray(jnp.asarray(x) @ jnp.ones((3,))), x.sum(1))


def test_sources_compile_without_syntax_warnings():
    """Invalid escape sequences are SyntaxWarnings on Python 3.12+ (errors later)."""
    bad = []
    for root in ("soma_jax", "tools", "benchmarks"):
        for path in sorted((REPO / root).rglob("*.py")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            bad += [f"{path.relative_to(REPO)}:{w.lineno}: {w.message}" for w in caught]
    assert not bad, "\n".join(bad)


def test_download_assets_check_runs_without_jax(tmp_path):
    """`tools/download_assets.py --check` must work before JAX is installed."""
    blocker = tmp_path / "jax"
    blocker.mkdir()
    (blocker / "__init__.py").write_text("raise ImportError('jax must not be imported')\n")
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "download_assets.py"), "--check", "--minimal"],
        capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)},
    )
    assert "jax must not be imported" not in proc.stdout + proc.stderr
    assert "required assets:" in proc.stdout
    assert proc.returncode in (0, 1), proc.stderr


def test_soma_jax_assets_module_is_stdlib_only(tmp_path):
    """soma_jax/assets.py is loaded standalone by the downloader."""
    code = textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("a", {str(REPO / "soma_jax" / "assets.py")!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        heavy = sorted(n for n in ("jax", "numpy", "scipy", "equinox") if n in sys.modules)
        print("HEAVY", heavy)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300).stdout
    assert "HEAVY []" in out, out
