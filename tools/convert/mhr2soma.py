"""MHR to SOMA pose converter.

Upstream: ``tools/mhr2soma.py`` (SOMA-X v0.3.3). Reads SAM 3D Body parquet files
holding MHR parameters (``shape_params``, ``model_params``) and converts them to
SOMA skeleton parameters with pose inversion, with upstream's arguments,
diagnostics and outputs — including v0.3's ``--smooth`` SO(3) RTS smoothing.

SAM 3D Body ``model_params`` layout (204 floats)::

    [0:3]     global translation (cm)
    [3:136]   pose parameters (axis-angle)
    [136:204] body-part scale parameters (68)

JAX port: the MHR forward is :class:`~soma_jax.body_models.mhr_native.MHRNativeModel`
(the weights lifted out of the same ``mhr_model_lod1.pt``), SOMA layers are
built in centimetres like upstream's (``output_unit=CENTIMETERS``), and the
refit runs on a low-LOD layer SOMA-JAX builds itself (upstream's
``PoseInversion(low_lod=True)`` builds it internally).

Upstream v0.3.3 fails on several of these paths with its own procedural
(twist-joint) layer; the port implements what the code intends instead:

* the internal low-LOD layer import (``from .body import SOMALayer`` inside
  ``soma/fitting``) raises ``ModuleNotFoundError``;
* ``--output-npz`` raises in ``export_soma_npz``, which removes the 110-joint
  twist-rig orient from the 78 public rotations — the port uses the public one;
* rendering raises in ``batched_skinning.pose``, which takes the 110-joint rig's
  rotations — the port renders the layer's own pose (public FK, twist
  expansion, LBS, no correctives);
* ``--autograd-iters`` >= 2 raises "backward through the graph a second time":
  the MHR TorchScript parameters require grad, so the prepared identity carries
  graph history — the port's identity is a constant, as intended;
* the drift summary labels joints from the 110-joint ``rig_data`` list, which
  mislabels every public joint from ``RightShoulder`` on — the port uses the
  public names.

The first chunk's (and a ragged last chunk's) timing includes XLA compilation.

Usage::

    python tools/convert/mhr2soma.py --input /path/to/sam_3d_body/coco_train
    python tools/convert/mhr2soma.py --input shard.parquet --max-samples 100 --output-npz out.npz
    python tools/convert/mhr2soma.py --input shard.parquet --smooth --no-render
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from conversion_utils import add_inversion_args, export_soma_npz  # noqa: E402
from logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_sam_parquet(path, max_samples=None):
    """Load MHR parameters from SAM 3D Body parquet file(s).

    Returns:
        dict with ``shape_params`` (N, 45), ``model_params`` (N, 204), and metadata.
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {path}")
        dfs, n = [], 0
        for f in files:
            df = pd.read_parquet(f)
            df = df[df["mhr_valid"]]
            dfs.append(df)
            n += len(df)
            if max_samples is not None and n >= max_samples:
                break
        df = pd.concat(dfs, ignore_index=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df[df["mhr_valid"]]
    else:
        raise ValueError(f"Expected .parquet file or directory, got: {path}")
    if max_samples is not None:
        df = df.iloc[:max_samples]

    shape_params = np.stack(df["shape_params"].values).astype(np.float32)
    model_params = np.stack(df["model_params"].values).astype(np.float32)
    logger.info(f"Loaded {len(df)} MHR samples from {path}")
    return {
        "shape_params": shape_params,
        "model_params": model_params,
        "datasets": df["dataset"].values if "dataset" in df.columns else None,
        "images": df["image"].values if "image" in df.columns else None,
    }


def parse_mhr_model_params(model_params):
    """Split ``model_params`` (N, 204) into translation (cm), pose (133) and scale (68)."""
    return model_params[:, :3], model_params[:, 3:136], model_params[:, 136:]


def _vertex_ids_for_roots(cache, root_names):
    from soma_jax.geometry.rig_utils import body_part_vertex_ids
    name_to_idx = {name: idx for idx, name in enumerate(cache["joint_names"])}
    ids = []
    for root_name in root_names:
        root_idx = name_to_idx.get(root_name)
        if root_idx is None:
            continue
        ids.append(np.asarray(body_part_vertex_ids(
            np.asarray(cache["skinning_weights"]), np.asarray(cache["parents"]), root_idx,
            include_root=True), np.int64))
    if not ids:
        return None
    return np.unique(np.concatenate(ids))


def compute_region_error_metrics(err, inv):
    """Coarse body-region error summaries for fit diagnostics."""
    from soma_jax.fitting.pose_inversion import _heel_vertex_ids
    cache = inv._cache
    if cache is None:
        return {}
    err = np.asarray(err)
    feet = _vertex_ids_for_roots(cache, ["LeftFoot", "RightFoot"])
    bind_shape = np.asarray(inv._rest_shape)
    heel_vids = _heel_vertex_ids(cache["joint_names"], np.asarray(cache["parents"]),
                                 np.asarray(cache["skinning_weights"]), bind_shape,
                                 np.asarray(inv._bind_joint_positions(cache)))
    heels = np.asarray(heel_vids, np.int64) if heel_vids else None
    regions = {
        "all": np.arange(err.shape[1]),
        "heels": heels,
        "feet": feet,
        "hands": _vertex_ids_for_roots(cache, ["LeftHand", "RightHand"]),
        "head": _vertex_ids_for_roots(cache, ["Head"]),
    }
    metrics = {}
    for name, vids in regions.items():
        if vids is None or len(vids) == 0:
            continue
        vals = err[:, vids].reshape(-1)
        metrics[name] = {"n": len(vids), "mean": float(vals.mean()),
                         "p95": float(np.quantile(vals, 0.95)), "max": float(vals.max())}
    return metrics


def print_region_error_summary(metrics, unit_label):
    """Log contact-focused region errors for fit diagnostics."""
    if "feet" in metrics and "hands" in metrics:
        contact_mean = 0.5 * (metrics["feet"]["mean"] + metrics["hands"]["mean"])
        logger.info(f"  Feet/hands equal-region mean: {contact_mean:.6f} {unit_label}")
    logger.info("  Region vertex error:")
    for name in ("heels", "feet", "hands", "head", "all"):
        if name not in metrics:
            continue
        vals = metrics[name]
        logger.info(f"    {name:6s} n={vals['n']:5d} "
                    f"mean={vals['mean']:.6f} {unit_label}, "
                    f"p95={vals['p95']:.6f} {unit_label}, "
                    f"max={vals['max']:.6f} {unit_label}")


def print_pose_drift_summary(local_rotation_drift, root_translation_drift, joint_names, unit_label):
    """Log drift from the warm-start pose used by regularized refinement."""
    if local_rotation_drift is None:
        return
    drift_deg = np.rad2deg(np.asarray(local_rotation_drift))
    if len(joint_names) + 1 == drift_deg.shape[1]:
        drift_eval, eval_names = drift_deg[:, 1:], joint_names
    elif joint_names and joint_names[0] == "Root" and drift_deg.shape[1] > 1:
        drift_eval, eval_names = drift_deg[:, 1:], joint_names[1:]
    else:
        drift_eval, eval_names = drift_deg, joint_names
    vals = drift_eval.reshape(-1)
    logger.info("  Local-rotation drift from refinement warm start:")
    logger.info(f"    mean={vals.mean():.4f} deg, p95={np.quantile(vals, 0.95):.4f} deg, "
                f"max={vals.max():.4f} deg")
    joint_means = drift_eval.mean(axis=0)
    topk = min(5, joint_means.size)
    if topk:
        top_ids = np.argsort(-joint_means, kind="stable")[:topk]
        top_parts = [f"{eval_names[int(i)]}={float(joint_means[i]):.3f} deg" for i in top_ids]
        logger.info(f"    largest mean joint drift: {', '.join(top_parts)}")
    if root_translation_drift is not None:
        rt = np.asarray(root_translation_drift)
        logger.info(f"    root translation drift mean={rt.mean():.6f} {unit_label}, "
                    f"p95={np.quantile(rt, 0.95):.6f} {unit_label}, max={rt.max():.6f} {unit_label}")


def parse_pose_prior_weights(spec):
    """Parse named autograd pose-prior profiles or comma-separated weights."""
    if spec is None or spec == "uniform":
        return None
    if spec == "heel_contact":
        weights = {}
        for joint in ("Hips", "Spine1", "Spine2", "Chest"):
            weights[joint] = 0.35
        for side in ("Left", "Right"):
            weights[f"{side}Leg"] = 0.35
            weights[f"{side}Shin"] = 6.0
            weights[f"{side}Foot"] = 8.0
            weights[f"{side}ToeBase"] = 10.0
            weights[f"{side}ToeEnd"] = 10.0
        return weights
    weights = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                "Expected --autograd-pose-prior-weights entries as "
                "'JointName=value' or the named profile 'heel_contact'.")
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights or None


def get_mhr_posed_vertices(mhr_model, identity_coeffs, model_params, batch_size=64):
    """MHR forward (with pose correctives): (N, V, 3) posed vertices in centimetres."""
    N = identity_coeffs.shape[0]
    out = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        verts, _ = mhr_model(identity_coeffs[start:end], model_params[start:end])
        out.append(np.asarray(verts))
    return np.concatenate(out, axis=0)


def convert_mhr_to_soma(posed_vertices, inv, body_iters=2, finger_iters=0, full_iters=1,
                        lie_iters=3, lie_lambda=1e-1, autograd_iters=0, autograd_lr=5e-3,
                        autograd_translation_lr_scale=1.0, autograd_pose_prior=0.0,
                        autograd_pose_prior_weights=None, autograd_leaf_weight=None,
                        leaf_weight=1.0, batch_size=64):
    """Invert MHR posed vertices (in the inversion layer's unit) to SOMA rotations."""
    return inv.fit(
        posed_vertices, body_iters=body_iters, finger_iters=finger_iters,
        full_iters=full_iters, lie_iters=lie_iters, lie_lambda=lie_lambda,
        autograd_iters=autograd_iters, autograd_lr=autograd_lr,
        autograd_translation_lr_scale=autograd_translation_lr_scale,
        autograd_pose_prior=autograd_pose_prior,
        autograd_pose_prior_weights=autograd_pose_prior_weights,
        autograd_leaf_weight=autograd_leaf_weight, leaf_weight=leaf_weight,
        batch_size=batch_size)


def main():
    from soma_jax.io import add_npz_args
    from soma_jax.fitting.rts_smoothing import RTS_SMOOTHING_PRESETS

    parser = argparse.ArgumentParser(description="MHR to SOMA pose converter.")
    parser.add_argument("--input", required=True,
                        help="Path to SAM 3D Body .parquet file or directory of parquet files. "
                             "Download parquet data from "
                             "https://huggingface.co/datasets/facebook/sam-3d-body-dataset")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    parser.add_argument("--output-usd", default=None, help="Output .usd/.usda/.usdc with UsdSkel.")
    add_npz_args(parser)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process.")
    add_inversion_args(parser, autograd=True)
    parser.add_argument("--leaf-weight", type=float, default=1.0,
                        help="Uniform extremity vertex weight (default: 1.0 = no upweight).")
    parser.add_argument("--hand-weight", type=float, default=None,
                        help="Override whole-hand vertex weight (default: same as --leaf-weight).")
    parser.add_argument("--foot-weight", type=float, default=None,
                        help="Override foot vertex weight (default: same as --leaf-weight).")
    parser.add_argument("--heel-weight", type=float, default=None,
                        help="Override rear heel vertex weight without weighting the whole foot.")
    parser.add_argument("--autograd-pose-prior", type=float, default=0.0,
                        help="Local-rotation prior weight for autograd FK refinement.")
    parser.add_argument("--autograd-pose-prior-weights", default=None,
                        help="Optional autograd pose-prior joint weights. Use 'heel_contact' "
                             "or comma-separated JointName=value entries. Values >1 stiffen a "
                             "joint; values <1 let it move more.")
    parser.add_argument("--autograd-hand-weight", type=float, default=None,
                        help="Whole-hand vertex weight used only by autograd FK.")
    parser.add_argument("--autograd-foot-weight", type=float, default=None,
                        help="Foot vertex weight used only by autograd FK.")
    parser.add_argument("--autograd-heel-weight", type=float, default=None,
                        help="Rear heel vertex weight used only by autograd FK.")
    parser.add_argument("--fps", type=int, default=4, help="Video frame rate (default: 4).")
    parser.add_argument("--smooth", action="store_true",
                        help="Apply reusable SO(3) RTS smoothing to SOMA rotations and root "
                             "translation.")
    parser.add_argument("--smooth-preset", choices=tuple(RTS_SMOOTHING_PRESETS), default="default",
                        help="RTS smoothing preset (default: default).")
    parser.add_argument("--smooth-fps", type=float, default=None,
                        help="Source frame rate for RTS smoothing dynamics (default: --fps).")
    parser.add_argument("--smooth-include-limbs", action="store_true",
                        help="Apply faster hand gains to limb joints as well as hand/finger joints.")
    parser.add_argument("--smooth-no-hand-gains", action="store_true",
                        help="Use body RTS gains for every joint.")
    parser.add_argument("--no-smooth-root", action="store_true",
                        help="Keep root translation unchanged when --smooth is enabled.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    import jax.numpy as jnp
    import trimesh

    from soma_jax import SOMALayer
    from soma_jax.assets import data_root as default_data_root
    from soma_jax.body_models.mhr_native import MHRNativeModel
    from soma_jax.fitting.pose_inversion import PoseInversion
    from soma_jax.fitting.rts_smoothing import smooth_pose
    from soma_jax.units import Unit

    data_root = Path(args.data_root) if args.data_root else default_data_root()

    # --- Load data ---
    sam_data = load_sam_parquet(args.input, max_samples=args.max_samples)
    shape_params = sam_data["shape_params"]
    model_params_raw = sam_data["model_params"]
    N = shape_params.shape[0]
    # MHR's 6 flexible bone-length parameters (model_params 130-135): identity-
    # like skeleton proportions that SAM 3D Body lets vary per frame; passed to
    # SOMA's identity model through kwargs so the rest shape matches.
    bone_length_flexibles = model_params_raw[:, 130:136].copy()
    translation_cm, pose_params, scale_params = parse_mhr_model_params(model_params_raw)
    logger.info(f"  Samples: {N}")
    logger.info(f"  Translation range (cm): [{translation_cm.min():.1f}, {translation_cm.max():.1f}]")
    logger.info(f"  Scale params range: [{scale_params.min():.3f}, {scale_params.max():.3f}]")

    # --- MHR model ---
    mhr_faces = trimesh.load(data_root / "MHR" / "base_body_lod1.obj", maintain_order=True,
                             process=False).faces
    mhr_model = MHRNativeModel.from_torchscript(data_root / "MHR" / "mhr_model_lod1.pt")

    # --- SOMA + pose inversion (centimetres, as upstream) ---
    logger.info("\nInitializing SOMA layer...")
    soma = SOMALayer.from_upstream_assets(
        identity_model_type="mhr", output_unit=Unit.CENTIMETERS,
        identity_model_kwargs={"mhr_model": mhr_model}, data_root=args.data_root)
    # Use low LOD for inversion (faster, negligible accuracy loss),
    # high LOD (soma) for rendering/evaluation.
    inv = PoseInversion(soma, low_lod=True)

    all_ic = jnp.asarray(shape_params)
    all_sp = jnp.asarray(scale_params)
    all_bl = jnp.asarray(bone_length_flexibles)

    if any(w is not None for w in (args.hand_weight, args.foot_weight, args.heel_weight)):
        leaf_weight = {
            "head": args.leaf_weight,
            "hands": args.leaf_weight if args.hand_weight is None else args.hand_weight,
            "feet": args.leaf_weight if args.foot_weight is None else args.foot_weight,
        }
        if args.heel_weight is not None:
            leaf_weight["heels"] = args.heel_weight
    else:
        leaf_weight = args.leaf_weight
    autograd_leaf_weight = None
    if any(w is not None for w in (args.autograd_hand_weight, args.autograd_foot_weight,
                                   args.autograd_heel_weight)):
        autograd_leaf_weight = {
            "head": 1.0,
            "hands": 1.0 if args.autograd_hand_weight is None else args.autograd_hand_weight,
            "feet": 1.0 if args.autograd_foot_weight is None else args.autograd_foot_weight,
        }
        if args.autograd_heel_weight is not None:
            autograd_leaf_weight["heels"] = args.autograd_heel_weight
    autograd_pose_prior_weights = parse_pose_prior_weights(args.autograd_pose_prior_weights)

    parts = []
    if args.body_iters > 0 or args.finger_iters > 0 or args.full_iters > 0:
        parts.append(f"analytical (body={args.body_iters}, finger={args.finger_iters}, "
                     f"full={args.full_iters})")
    if args.lie_iters > 0:
        parts.append(f"lie-gn ({args.lie_iters} iters, lambda={args.lie_lambda})")
    if args.autograd_iters > 0:
        parts.append(f"autograd FK ({args.autograd_iters} iters, lr={args.autograd_lr}, "
                     f"translation_lr_scale={args.autograd_translation_lr_scale}, "
                     f"pose_prior={args.autograd_pose_prior})")
    method_desc = " + ".join(parts) if parts else "none"
    if leaf_weight != 1.0:
        method_desc += f", leaf_weight={leaf_weight}"
    if autograd_leaf_weight is not None:
        method_desc += f", autograd_leaf_weight={autograd_leaf_weight}"
    if autograd_pose_prior_weights is not None:
        method_desc += f", autograd_pose_prior_weights={autograd_pose_prior_weights}"

    batch_size = args.batch_size
    logger.info(f"\nInverting {N} samples with {method_desc}...")
    t0 = time.perf_counter()
    all_rotations, all_root_transl, all_errors = [], [], []
    all_rotation_drift, all_root_translation_drift = [], []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # Per-chunk identity, including the per-frame bone-length flexibles.
        inv.prepare_identity(all_ic[start:end], all_sp[start:end],
                             kwargs={"bone_length_flexibles": all_bl[start:end]})
        verts_cm, _ = mhr_model(all_ic[start:end], jnp.asarray(model_params_raw[start:end]))
        result = convert_mhr_to_soma(
            verts_cm, inv, body_iters=args.body_iters, finger_iters=args.finger_iters,
            full_iters=args.full_iters, lie_iters=args.lie_iters, lie_lambda=args.lie_lambda,
            autograd_iters=args.autograd_iters, autograd_lr=args.autograd_lr,
            autograd_translation_lr_scale=args.autograd_translation_lr_scale,
            autograd_pose_prior=args.autograd_pose_prior,
            autograd_pose_prior_weights=autograd_pose_prior_weights,
            autograd_leaf_weight=autograd_leaf_weight, leaf_weight=leaf_weight,
            batch_size=None)
        # Host copies per chunk, as upstream's `.cpu()`: device memory stays
        # bounded by one chunk however many samples the shard holds.
        all_rotations.append(np.asarray(result["rotations"]))
        all_root_transl.append(np.asarray(result["root_translation"]))
        all_errors.append(np.asarray(result["per_vertex_error"]))
        if "local_rotation_drift" in result:
            all_rotation_drift.append(np.asarray(result["local_rotation_drift"]))
        if "root_translation_drift" in result:
            all_root_translation_drift.append(np.asarray(result["root_translation_drift"]))
    dt = time.perf_counter() - t0

    rotations = jnp.asarray(np.concatenate(all_rotations, axis=0))
    root_transl = jnp.asarray(np.concatenate(all_root_transl, axis=0))
    err = np.concatenate(all_errors, axis=0)
    rotation_drift = np.concatenate(all_rotation_drift, axis=0) if all_rotation_drift else None
    root_translation_drift = (np.concatenate(all_root_translation_drift, axis=0)
                              if all_root_translation_drift else None)

    unit_label = "cm" if soma.output_unit == Unit.CENTIMETERS else "m"
    logger.info(f"  Inversion time: {dt:.2f}s ({N / dt:.0f} FPS)")
    logger.info(f"  Mean vertex error: {err.mean():.6f} {unit_label}")
    logger.info(f"  Max vertex error:  {err.max():.6f} {unit_label}")
    # torch.median of an even count returns the lower middle value.
    logger.info(f"  Median vertex error: {np.sort(err.ravel())[(err.size - 1) // 2]:.6f} "
                f"{unit_label}")
    print_region_error_summary(compute_region_error_metrics(err, inv), unit_label)
    print_pose_drift_summary(rotation_drift, root_translation_drift,
                             list(soma.public_joint_names), unit_label)

    if args.smooth:
        smooth_config = replace(
            RTS_SMOOTHING_PRESETS[args.smooth_preset],
            fps=float(args.smooth_fps if args.smooth_fps is not None else args.fps),
            smooth_root_translation=not args.no_smooth_root)
        logger.info("\nApplying SO(3) RTS smoothing "
                    f"(preset={args.smooth_preset}, fps={smooth_config.fps:g})...")
        rotations, root_transl = smooth_pose(
            rotations, root_transl, soma_layer=soma, config=smooth_config,
            rotation_convention="absolute", output_rotation_convention="absolute",
            use_hand_gains=not args.smooth_no_hand_gains,
            include_limb_gains=args.smooth_include_limbs)

    if args.output_npz:
        extra_arrays = {"bone_length_flexibles": bone_length_flexibles}
        if args.smooth:
            extra_arrays.update({
                "rts_smoothing": np.array("so3_error_state"),
                "rts_smoothing_preset": np.array(args.smooth_preset),
                "rts_smoothing_fps": np.float32(smooth_config.fps),
                "rts_smoothing_root_enabled": np.bool_(not args.no_smooth_root),
            })
        if rotation_drift is not None:
            extra_arrays["local_rotation_drift_deg"] = np.rad2deg(rotation_drift)
        export_soma_npz(args.output_npz, rotations, root_transl, soma,
                        output_unit=args.output_unit, keep_root=args.keep_root,
                        identity_coeffs=shape_params, scale_params=scale_params,
                        extra_arrays=extra_arrays)

    if args.output_usd:
        # USD has one bind pose, but SAM 3D Body identities vary per sample.
        identity_varies = (np.std(shape_params, axis=0).max() > 1e-4
                           or np.std(scale_params, axis=0).max() > 1e-4
                           or np.std(bone_length_flexibles, axis=0).max() > 1e-4)
        if identity_varies:
            import warnings
            warnings.warn(
                "Identity parameters vary across samples. USD export uses the first sample's "
                "shape as a fixed bind pose; skinning will be approximate for samples with "
                "different body shapes.", stacklevel=1)
        from soma_jax.usd_io import export_soma_usd
        rest, _, binds = soma.prepare_identity(
            all_ic[:1], all_sp[:1], return_bind_transforms=True,
            kwargs={"bone_length_flexibles": all_bl[:1]})
        export_soma_usd(args.output_usd, soma, rotations, root_transl,
                        bind_transforms_world=binds, rest_shape=rest, fps=float(args.fps),
                        unit="centimeters")

    if args.no_render:
        return

    from vis_pyrender import default_pyopengl_platform, render_comparison_video, set_pyopengl_platform
    set_pyopengl_platform(default_pyopengl_platform())
    cm_to_m = Unit.CENTIMETERS.meters_per_unit
    mhr_verts_all, eval_verts_all = [], []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        rest, joints, binds = soma.prepare_identity(
            all_ic[start:end], all_sp[start:end], return_bind_transforms=True,
            kwargs={"bone_length_flexibles": all_bl[start:end]})
        verts_cm, _ = mhr_model(all_ic[start:end], jnp.asarray(model_params_raw[start:end]))
        mhr_verts_all.append(np.asarray(verts_cm) * cm_to_m)
        eval_v = soma.pose(rotations[start:end], root_transl[start:end], rest, joints,
                           bind_transforms=binds, absolute_pose=True,
                           apply_correctives=False).vertices
        eval_verts_all.append(np.asarray(eval_v) * cm_to_m)

    parts_tag = []
    if args.body_iters > 0 or args.finger_iters > 0 or args.full_iters > 0:
        parts_tag.append("analytical")
    if args.lie_iters > 0:
        parts_tag.append(f"lie{args.lie_iters}")
    if args.autograd_iters > 0:
        parts_tag.append(f"autograd{args.autograd_iters}")
    method_tag = "_".join(parts_tag) if parts_tag else "none"
    out_name = f"out/mhr2soma_eval_{method_tag}.mp4"
    Path(out_name).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Rendering MHR (with correctives) vs SOMA (no correctives) -> {out_name}")
    render_comparison_video(out_name, np.concatenate(mhr_verts_all, axis=0), mhr_faces,
                            np.concatenate(eval_verts_all, axis=0), np.asarray(soma.faces),
                            center=True, cam_dist_scale=5.0, fps=args.fps, label_source="MHR",
                            label_soma=f"SOMA {method_tag}")


if __name__ == "__main__":
    main()
