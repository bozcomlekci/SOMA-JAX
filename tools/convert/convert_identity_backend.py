"""Convert the identity backend of a full-body SOMA NPZ animation.

Upstream: ``tools/convert_identity_backend.py`` (SOMA-X v0.3.0). Same arguments
and defaults; the layers are :meth:`soma_jax.SOMALayer.from_upstream_assets`
(procedural, no correctives), whose non-SOMA identity backends are built from
the asset directory exactly as upstream's.

Usage::

    python tools/convert/convert_identity_backend.py in.npz out.npz --target-backend mhr
    python tools/convert/convert_identity_backend.py in.npz out.npz --target-backend smpl \\
        --target-model-path data/smpl/SMPL_NEUTRAL.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools", REPO / "tools" / "convert"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from identity_conversion import convert_soma_npz, model_kwargs  # noqa: E402
from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

BODY_BACKENDS = ("soma", "mhr", "anny", "smpl", "smplh", "smplx", "garment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert identity parameters in a full-body SOMA NPZ via bind-pose fitting.")
    parser.add_argument("input", type=Path, help="Input SOMA NPZ from soma_jax.io.save_soma_npz.")
    parser.add_argument("output", type=Path, help="Output SOMA NPZ with converted identity.")
    parser.add_argument("--target-backend", required=True, choices=BODY_BACKENDS)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Upstream-layout asset directory (default: soma_jax.assets).")
    parser.add_argument("--device", default=None,
                        help="Accepted for upstream CLI compatibility; JAX places arrays on "
                             "its default device (set JAX_PLATFORMS to choose).")
    parser.add_argument("--lod", choices=("mid", "low", "xlo"), default="low")
    parser.add_argument("--source-model-path", default=None)
    parser.add_argument("--target-model-path", default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument(
        "--no-optimize-scale-params", action="store_true",
        help="Keep target scale parameters neutral. Native SOMA bone scales are optimized by default.")
    parser.add_argument(
        "--optimize-global-scale", action="store_true",
        help="Optimize a target global scale. By default the input global scale is fixed.")
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(args)
    from soma_jax import SOMALayer

    def make_layer(backend: str, unit: str, role: str):
        selected_path = args.source_model_path if role == "source" else args.target_model_path
        return SOMALayer.from_upstream_assets(
            identity_model_type=backend,
            identity_model_kwargs=model_kwargs(selected_path),
            lod=args.lod,
            data_root=args.data_root,
        )

    result = convert_soma_npz(
        args.input, args.output,
        target_backend=args.target_backend,
        layer_factory=make_layer,
        optimize_scale_params=not args.no_optimize_scale_params,
        optimize_global_scale=args.optimize_global_scale,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
    )
    logger.info("Output: %s", args.output)
    logger.info("Mean bind-pose vertex error: %.6f", float(result.vertex_error.mean()))


if __name__ == "__main__":
    main()
