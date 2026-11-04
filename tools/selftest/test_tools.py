"""The command-line tools run end to end on this machine (subprocesses)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ._helpers import UPSTREAM_ASSETS, need_assets, need_module

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.slow]


def _run(*args, timeout=1800, env=None):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          timeout=timeout, cwd=REPO, env={**os.environ, **(env or {})})
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return proc.stdout


def _gl_works() -> str | None:
    """Platform a pyrender offscreen context can be made with here, or skip."""
    need_module("pyrender")
    probe = (
        "import sys; sys.path.insert(0, %r)\n"
        "from _common import ensure_headless_gl\n"
        "p = ensure_headless_gl()\n"
        "import pyrender\n"
        "r = pyrender.OffscreenRenderer(16, 16); r.delete(); print('GL-OK', p)\n"
    ) % str(REPO / "tools")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          timeout=300)
    if "GL-OK" not in proc.stdout:
        pytest.skip("no usable OpenGL context for offscreen rendering here: "
                    + (proc.stderr.strip().splitlines() or ["?"])[-1])
    return proc.stdout.split("GL-OK", 1)[1].strip()


def test_download_assets_check_minimal():
    need_assets(*UPSTREAM_ASSETS)
    out = _run(REPO / "tools" / "download_assets.py", "--check", "--minimal")
    assert "all required assets present and verified" in out


def test_vis_mesh_export(tmp_path):
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    obj = tmp_path / "rest.obj"
    _run(REPO / "tools" / "vis" / "vis_mesh_export.py", "--output", obj)
    lines = obj.read_text().splitlines()
    assert sum(ln.startswith("v ") for ln in lines) == 18056
    assert sum(ln.startswith("f ") for ln in lines) == 36108


def test_vis_pyrender_offscreen_is_lit(tmp_path):
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    _gl_works()
    PIL = need_module("PIL.Image")
    png = tmp_path / "rest.png"
    _run(REPO / "tools" / "vis" / "vis_pyrender.py", "--output", png,
         "--width", "256", "--height", "256")
    img = np.asarray(PIL.open(png).convert("RGB"), np.float32)
    body = img.mean(axis=2) < 250                 # white background
    assert body.mean() > 0.02, "nothing rendered"
    lum = img[body].mean(axis=1)
    # A lit, smooth-shaded body spans a range of intensities; an unlit one
    # (e.g. integer light colours, which pyrender reads as 0-255) is flat.
    assert np.percentile(lum, 95) - np.percentile(lum, 5) > 20, "render is unshaded"


def test_demo_compute_only(tmp_path):
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    out = _run(REPO / "tools" / "pipeline" / "demo_soma_vis.py", "--no-render",
               "--num-frames", "3", "--output-dir", tmp_path)
    assert "Computed 3-frame pose sequence" in out


def test_demo_renders(tmp_path):
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    _gl_works()
    _run(REPO / "tools" / "pipeline" / "demo_soma_vis.py", "--num-frames", "4",
         "--width", "128", "--height", "128", "--output-dir", tmp_path,
         "--gif", tmp_path / "demo.gif")
    for name in ("01_rest.png", "02_custom_shape.png", "03_posed.png", "demo.gif"):
        assert (tmp_path / name).stat().st_size > 1000, name
