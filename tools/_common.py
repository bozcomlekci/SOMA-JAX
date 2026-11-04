"""Helpers shared by the command-line tools (not part of the ``soma_jax`` package).

* :func:`ensure_headless_gl` picks an offscreen OpenGL backend for pyrender
  before it is imported, so the render scripts work on a machine with no
  display (servers, CI, containers) without hand-setting ``PYOPENGL_PLATFORM``.
* :func:`load_layer` builds a :class:`soma_jax.SOMALayer` from whichever SOMA
  asset is at hand — nothing (resolve the upstream assets), NVIDIA's own
  ``SOMA_neutral.npz``, or a prebuilt ``SOMA_neutral_fixed.npz``.

The scripts put ``tools/`` on ``sys.path`` and ``import _common``.
"""
from __future__ import annotations

import ctypes.util
import os
import sys
import zipfile

__all__ = ["ensure_headless_gl", "load_layer", "is_upstream_archive"]


def _pyopengl_version() -> tuple[int, ...]:
    try:
        from importlib.metadata import version
        return tuple(int(p) for p in version("PyOpenGL").split(".")[:3] if p.isdigit())
    except Exception:  # noqa: BLE001 — not installed / odd version string
        return ()


def ensure_headless_gl() -> str | None:
    """Choose PyOpenGL's platform for offscreen rendering; call before importing pyrender.

    PyOpenGL reads ``PYOPENGL_PLATFORM`` once, at import, so this has to run
    first. An explicit ``PYOPENGL_PLATFORM`` always wins. Otherwise, on Linux
    without an X display, EGL is used when ``libEGL`` is installed (Mesa's
    software EGL works without a GPU), then OSMesa. With a display — and on
    macOS / Windows — the platform's native GL is left in charge.

    Returns:
        The platform in effect, or ``None`` for PyOpenGL's default.

    Raises:
        RuntimeError: headless Linux with neither EGL nor OSMesa installed.
    """
    platform = os.environ.get("PYOPENGL_PLATFORM")
    if platform:
        return platform
    if not sys.platform.startswith("linux") or os.environ.get("DISPLAY"):
        return None
    if ctypes.util.find_library("EGL"):
        platform = "egl"
    elif ctypes.util.find_library("OSMesa"):
        platform = "osmesa"
        if _pyopengl_version() and _pyopengl_version() < (3, 1, 5):
            print("WARNING: rendering through OSMesa, but PyOpenGL "
                  f"{'.'.join(map(str, _pyopengl_version()))} (pyrender's pin) cannot "
                  "create an OSMesa context. Install EGL instead (apt install libegl1), "
                  "or upgrade PyOpenGL: pip install 'PyOpenGL>=3.1.5' "
                  "(pip reports pyrender's pin as a conflict; the newer version works).",
                  file=sys.stderr)
    else:
        raise RuntimeError(
            "No display and no headless OpenGL library. Install EGL "
            "(Debian/Ubuntu: apt install libegl1; Fedora: dnf install mesa-libEGL) "
            "or OSMesa (apt install libosmesa6), or run under a virtual display: "
            "xvfb-run -a python ...")
    os.environ["PYOPENGL_PLATFORM"] = platform
    return platform


def is_upstream_archive(path) -> bool:
    """True for NVIDIA's ``SOMA_neutral.npz`` schema (``mean``, no ``v_template``)."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = {n[:-4] if n.endswith(".npy") else n for n in zf.namelist()}
    except (OSError, zipfile.BadZipFile):
        return False
    return "v_template" not in names and "mean" in names


def load_layer(path=None, identity_model_type: str = "soma", **kwargs):
    """A ``SOMALayer`` from whichever SOMA asset is at hand.

    Args:
        path: ``None`` to build from the resolved upstream assets
            (:meth:`SOMALayer.from_upstream_assets`, upstream's default
            procedural rig); NVIDIA's full-schema ``SOMA_neutral.npz`` (same
            route, from that file); or a SOMA-JAX archive such as
            ``SOMA_neutral_fixed.npz`` (:meth:`SOMALayer.load`).
        identity_model_type: identity backend.
        **kwargs: forwarded to the constructor that is used.
    """
    from soma_jax import SOMALayer

    if path is None or is_upstream_archive(path):
        return SOMALayer.from_upstream_assets(
            npz_path=None if path is None else str(path),
            identity_model_type=identity_model_type, **kwargs)
    return SOMALayer.load(str(path), identity_model_type=identity_model_type, **kwargs)
