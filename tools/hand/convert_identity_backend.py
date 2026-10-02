"""Convert the identity backend of a SOMA Hand NPZ animation.

Upstream: ``tools/hand/convert_identity_backend.py`` (SOMA-X v0.3.0). Same
arguments and defaults, over :class:`soma_jax.hand.SOMAHandLayer` and the shared
optimizer in ``tools/convert/identity_conversion.py``.

Usage::

    python tools/hand/convert_identity_backend.py hand.npz out.npz --target-backend mhr
    python tools/hand/convert_identity_backend.py hand.npz out.npz --target-backend mano \\
        --hand-type left --target-model-path /path/to/MANO_LEFT.pkl
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

HAND_BACKENDS = ("soma", "mhr", "mano")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert identity parameters in a SOMA Hand NPZ via bind-pose fitting.")
    parser.add_argument("input", type=Path, help="Input SOMA NPZ from soma_jax.io.save_soma_npz.")
    parser.add_argument("output", type=Path, help="Output SOMA NPZ with converted identity.")
    parser.add_argument("--target-backend", required=True, choices=HAND_BACKENDS)
    parser.add_argument("--hand-type", choices=("left", "right"), default=None)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Upstream-layout asset directory (default: soma_jax.assets.data_root()).")
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
    from soma_jax.assets import data_root as default_data_root
    from soma_jax.hand import SOMAHandLayer
    from soma_jax.io import load_soma_npz
    from soma_jax.units import Unit

    input_data = load_soma_npz(str(args.input))
    hand_type = args.hand_type or input_data.get("hand_type")
    if hand_type is None:
        raise ValueError("Hand SOMA NPZ must contain hand_type or pass --hand-type")
    hand_type = str(hand_type)
    data_root = args.data_root if args.data_root is not None else default_data_root()

    def make_layer(backend: str, unit: str, role: str):
        selected_path = args.source_model_path if role == "source" else args.target_model_path
        return SOMAHandLayer(
            data_root=data_root,
            hand_type=hand_type,
            identity_model_type=backend,
            identity_model_kwargs=model_kwargs(selected_path),
            lod=args.lod,
            output_unit=Unit.from_name(unit),
            correctives_model_path=None,
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
        expected_hand_type=hand_type,
    )
    logger.info("Output: %s", args.output)
    logger.info("Mean bind-pose vertex error: %.6f", float(result.vertex_error.mean()))


if __name__ == "__main__":
    main()
