"""Minimal script to convert GarmentMeasurements point.pca to point.npz format.

Upstream: ``tools/convert_gm_pca_to_npz.py`` (SOMA-X v0.3.3), unchanged apart from
the import path. The ``point.npz`` it writes is what the ``garment`` identity
backend reads (``<data-root>/GarmentMeasurements/point.npz``).

Usage::

    python tools/convert/convert_gm_pca_to_npz.py <input.pca> <output.npz>

Example::

    python tools/convert/convert_gm_pca_to_npz.py \\
        assets/GarmentMeasurements/point.pca assets/GarmentMeasurements/point.npz
"""
from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def convert_pca_to_npz(input_file: str, output_file: str):
    """Convert binary PCA file to NPZ format."""
    with open(input_file, "rb") as f:
        # Read dimensions
        m = struct.unpack("I", f.read(4))[0]
        n = struct.unpack("I", f.read(4))[0]

        # Read PCA matrix (column-major order)
        pca_matrix = np.frombuffer(f.read(m * n * 8), dtype=np.float64).reshape(n, m).T

        # Read mean vector
        pca_mean = np.frombuffer(f.read(m * 8), dtype=np.float64)

        # Read eigenvalues
        eigenvalues = np.frombuffer(f.read(n * 8), dtype=np.float64)

    # Save as NPZ
    np.savez_compressed(
        output_file,
        pca_matrix=pca_matrix,
        pca_mean=pca_mean,
        eigenvalues=eigenvalues,
        dimensions=np.array([m, n], dtype=np.int32),
    )

    logger.info(f"Converted {input_file} to {output_file}")
    logger.info(f"Dimensions: m={m}, n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input GarmentMeasurements .pca file.")
    parser.add_argument("output_file", help="Output .npz file.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    convert_pca_to_npz(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
