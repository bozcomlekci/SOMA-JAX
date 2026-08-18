"""Interactive demo for SOMA-JAX visualizing different identity models and poses.

This demo generates a visualization comparing:
    - Rest pose with default shape (zero coefficients)
    - Rest pose with custom shape coefficients
    - Posed mesh at a sample pose
    - Optionally cycles through identity models if their data is available

Requires: pyrender, trimesh, pillow (for image output)

Usage::

    # Quick demo with default settings
    python tools/demo_soma_vis.py --soma-model path/to/SOMA_neutral.npz

    # Demo with multiple identity models
    python tools/demo_soma_vis.py \\
        --soma-model path/to/SOMA_neutral.npz \\
        --smpl-model path/to/SMPL_NEUTRAL.pkl \\
        --output-dir demo_renders/

    # Animated GIF from sample poses
    python tools/demo_soma_vis.py \\
        --soma-model path/to/SOMA_neutral.npz \\
        --gif demo.gif --num-frames 30
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="SOMA-JAX visualization demo")
    p.add_argument("--soma-model", required=True, help="SOMA_neutral.npz")
    p.add_argument("--smpl-model", default=None, help="Optional SMPL .pkl file")
    p.add_argument("--smplx-model", default=None, help="Optional SMPL-X .pkl/.npz file")
    p.add_argument("--smplh-model", default=None, help="Optional SMPL-H .pkl/.npz file")
    p.add_argument("--mhr-identity", default=None,
                   help="Optional MHR identity .npz (v_template, shapedirs, bary_*). "
                        "Used by SOMALayer with identity_model_type='mhr'.")
    p.add_argument("--anny-identity", default=None,
                   help="Optional Anny identity .npz. Used by SOMALayer with identity_model_type='anny'.")
    p.add_argument("--garment-identity", default=None,
                   help="Optional garment-measurement identity .npz. "
                        "Used by SOMALayer with identity_model_type='garment_measurement'.")
    p.add_argument("--output-dir", default="demo_output", help="Output directory")
    p.add_argument("--gif", default=None, help="Output animated GIF path")
    p.add_argument("--num-frames", type=int, default=30, help="Frames for animation")
    p.add_argument("--width", type=int, default=512, help="Render width")
    p.add_argument("--height", type=int, default=512, help="Render height")
    p.add_argument("--no-render", action="store_true", help="Compute meshes but skip rendering")
    p.add_argument("--wireframe", action="store_true", help="Render wireframe view")
    p.add_argument("--side-by-side", action="store_true",
                   help="Render all loaded body models side-by-side over the same motion")
    p.add_argument("--soma-skeleton-overlay", action="store_true",
                   help="Overlay the SOMA 78-joint skeleton on every column (including "
                        "SMPL/SMPL-X) instead of each model's native skeleton. Demonstrates "
                        "that the unified SOMA rig fits any body model shape.")
    p.add_argument("--motion", default=None,
                   help="SMPL-X motion clip (.npz). "
                        "When set, ALL models obey this single sequence: SMPL/-H/-X natively, "
                        "SOMA/MHR/Anny/Garment via inverse-LBS retargeting onto the SOMA skeleton.")
    p.add_argument("--bvh-motion", default=None,
                   help="SOMA-skeleton BVH file. Bypasses SMPL-X retargeting since "
                        "a BVH is already in SOMA format. SMPL/SMPL-X columns are skipped.")
    p.add_argument("--mhr-subject", default=None,
                   help="MHR subject .npz (identity_params + scale_params). When set, the MHR "
                        "column uses the subject's real identity_params + scale_params via the "
                        "full JAX MHR rig.")
    p.add_argument("--hf-dir", default=None,
                   help="Directory with the nvidia/SOMA-X HF assets (for SMPL-X→SOMA wrap "
                        "correspondence). Default: soma_jax.assets.data_root().")
    p.add_argument("--random-shape", action="store_true",
                   help="Sample random identity coefficients per model (SOMA-X demo feature) "
                        "instead of the neutral shape")
    p.add_argument("--refine-iters", type=int, default=0,
                   help="Autograd FK refinement iterations for SMPL-X→SOMA retargeting "
                        "(0 = analytical only; >0 = SOMA-X 'best accuracy' mode)")
    p.add_argument("--export-soma-npz", default=None,
                   help="Export retargeted SOMA motion to this .npz (SOMA-X format: poses, "
                        "root_translation, joint_names, per_vertex_error, identity_coeffs)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --random-shape")
    p.add_argument("--mhr-rig", default="assets/mhr_rig.npz",
                   help="MHR rig buffers (.npz from tools/mhr_jax.py) for the full parametric "
                        "MHR model with scale_params")
    p.add_argument("--mhr-scale", type=float, default=None,
                   help="Apply this log2 body-part scale to all 68 MHR scale_params "
                        "(demonstrates MHR's scale_params via the full JAX rig)")
    p.add_argument("--low-lod", action="store_true",
                   help="Use low-LOD SOMA assets (SOMA-X low_lod) from --lowlod-dir "
                        "for the SOMA-family columns")
    p.add_argument("--lowlod-dir", default="assets/lowlod",
                   help="Directory of low-LOD assets (from tools/build_lowlod.py)")
    p.add_argument("--target-fps", type=float, default=None,
                   help="When driving from a BVH, pick num_frames so the GIF plays at this rate "
                        "(default: auto-cap source FPS to 30). Overrides --num-frames.")
    p.add_argument("--ground-lock", action="store_true",
                   help="Per-frame, shift each identity's vertices so its lowest Y sits on y=0. "
                        "Removes the global up/down translation flicker between mismatched body "
                        "sizes so all side-by-side characters share a common ground plane.")
    args = p.parse_args()
    if args.hf_dir is None:
        # The NVIDIA source assets live in assets/third_party/ or the vendored
        # submodule depending on how they were obtained; data_root() resolves
        # both and materialises a single view over them.
        from soma_jax.assets import data_root
        args.hf_dir = str(data_root(materialise=True))
    return args


# Z-up (SMPL-X world) -> Y-up (render camera) rotation.
_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


# Per-model color palette for side-by-side renders (deeper, saturated tones so
# the body reads clearly even when rendered semi-transparent over the skeleton).
_MODEL_COLORS = {
    "SOMA":    (0.42, 0.62, 0.45),   # green
    "SMPL":    (0.52, 0.60, 0.40),   # olive
    "SMPL-H":  (0.62, 0.50, 0.38),   # tan
    "SMPL-X":  (0.68, 0.40, 0.58),   # magenta/pink
    "MHR":     (0.38, 0.52, 0.70),   # blue
    "Anny":    (0.70, 0.62, 0.30),   # gold
    "Garment": (0.72, 0.42, 0.40),   # salmon
}

# Skeleton overlay colors.
_BONE_COLOR = (0.18, 0.18, 0.22)
_JOINT_COLOR = (0.95, 0.85, 0.25)


def _pose_smpl(model, poses_seq, transl_seq):
    """SMPL: 1 root + 23 body joints; body_pose flat (T, 69)."""
    import jax.numpy as jnp
    T = poses_seq.shape[0]
    from smpl_jax import SMPLParams
    return SMPLParams(
        betas=jnp.zeros((T, model.num_betas), dtype=jnp.float32),
        body_pose=jnp.asarray(poses_seq[:, 1:24].reshape(T, -1)),
        global_orient=jnp.asarray(poses_seq[:, 0]),
        transl=jnp.asarray(transl_seq),
    )


def _pose_smplh(model, poses_seq, transl_seq):
    """SMPL-H: 1 root + 21 body + 15+15 hand joints."""
    import jax.numpy as jnp
    T = poses_seq.shape[0]
    from soma_jax import SMPLHParams
    return SMPLHParams(
        betas=jnp.zeros((T, model.num_betas), dtype=jnp.float32),
        body_pose=jnp.asarray(poses_seq[:, 1:22].reshape(T, -1)),
        global_orient=jnp.asarray(poses_seq[:, 0]),
        transl=jnp.asarray(transl_seq),
        left_hand_pose=jnp.zeros((T, 15 * 3), dtype=jnp.float32),
        right_hand_pose=jnp.zeros((T, 15 * 3), dtype=jnp.float32),
    )


def _pose_smplx(model, poses_seq, transl_seq):
    """SMPL-X: 1 root + 21 body + 3 face + 15+15 hand joints."""
    import jax.numpy as jnp
    T = poses_seq.shape[0]
    from smpl_jax import SMPLXParams
    return SMPLXParams(
        betas=jnp.zeros((T, model.num_betas), dtype=jnp.float32),
        body_pose=jnp.asarray(poses_seq[:, 1:22].reshape(T, -1)),
        global_orient=jnp.asarray(poses_seq[:, 0]),
        transl=jnp.asarray(transl_seq),
        expression=jnp.zeros((T, model.num_expression_coeffs), dtype=jnp.float32),
        jaw_pose=jnp.zeros((T, 3), dtype=jnp.float32),
        leye_pose=jnp.zeros((T, 3), dtype=jnp.float32),
        reye_pose=jnp.zeros((T, 3), dtype=jnp.float32),
        left_hand_pose=jnp.zeros((T, 15 * 3), dtype=jnp.float32),
        right_hand_pose=jnp.zeros((T, 15 * 3), dtype=jnp.float32),
    )


def run_soma_identity_sequence(soma_path, identity_type, identity_path, poses_seq, transl_seq, n_id_coeffs=10):
    """Pose a SOMALayer with the given identity model over a (T, J, 3) sequence.

    SOMA-X reuses the SOMA topology and skeleton for every identity variant
    (smpl / smplx / mhr / anny / soma / garment_measurement). The identity model
    only chooses the rest-pose vertices; the LBS and joints are shared.
    Returns (T, V, 3) posed vertices and the SOMA faces.
    """
    import jax.numpy as jnp
    import numpy as np
    from soma_jax import SOMALayer, SOMAParams
    layer = SOMALayer.load(
        soma_path,
        identity_model_type=identity_type,
        identity_model_path=identity_path,
    )
    T = poses_seq.shape[0]
    # Match this variant's joint count (SOMA topology = 78, but slice to be safe).
    n_joints = len(layer.joint_names)
    seq = poses_seq[:, :n_joints]
    if seq.shape[1] < n_joints:
        # Pad with zeros if the unified sequence is shorter than the SOMA skeleton.
        pad = np.zeros((T, n_joints - seq.shape[1], 3), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=1)
    verts = []
    for i in range(T):
        params = SOMAParams(
            poses=jnp.asarray(seq[i]),
            transl=jnp.asarray(transl_seq[i]),
            identity_coeffs=jnp.zeros(n_id_coeffs, dtype=jnp.float32),
        )
        out = layer(params)
        verts.append(np.array(out.vertices))
    return np.stack(verts, axis=0), np.array(layer.faces)


def pose_soma_identity_sequence(soma_path, identity_type, identity_path, soma_poses,
                                n_id_coeffs=10, correctives=None, identity_coeffs=None,
                                rest_verts_override=None, transl_seq=None):
    """Apply a precomputed SOMA pose sequence to a SOMA-family identity.

    identity_coeffs: optional (n_id_coeffs,) shape coefficients (e.g. for
        --random-shape). Defaults to the neutral (zero) shape.
    rest_verts_override: optional (V_soma, 3) SOMA-topology rest vertices to use
        instead of the identity model's (e.g. the full MHR rig with scale_params,
        or the SMPL-X-transferred SOMA rest from the --motion retarget — required
        when the rotations were fit relative to that rest so the FK chain
        evaluates against the same skeleton).
    transl_seq: optional (T, 3) per-frame root translation. When provided the
        SOMA character translates with the motion (matching what SMPL/SMPL-X do
        when fed the same source `trans`).

    soma_poses: (T, 78, 3) parent-relative axis-angle (e.g. from motion retargeting).
    correctives: optional dict from correctives_jax.load_correctives. When given,
        the unified pose-corrective offsets are added to the rest shape before LBS,
        matching SOMA-X's "Unified Pose Correctives" (fixes LBS volume artifacts).
    Returns (verts_seq (T,V,3), faces, joints_seq (T,J,3), parents (J,)).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from soma_jax import SOMALayer, SOMAParams
    from soma_jax.geometry.transforms import axis_angle_to_rotmat
    layer = SOMALayer.load(soma_path, identity_model_type=identity_type,
                           identity_model_path=identity_path)
    T = soma_poses.shape[0]
    J = soma_poses.shape[1]
    verts, joints = [], []
    if identity_coeffs is None:
        identity_coeffs = np.zeros(n_id_coeffs, dtype=np.float32)
    id_coeffs = jnp.asarray(identity_coeffs, dtype=jnp.float32)
    transl_arr = (np.asarray(transl_seq, dtype=np.float32) if transl_seq is not None
                  else np.zeros((T, 3), dtype=np.float32))

    if rest_verts_override is not None:
        rest_verts = jnp.asarray(rest_verts_override, dtype=jnp.float32)
        rest_joints = jnp.einsum("jv,vd->jd", jnp.asarray(layer.J_regressor), rest_verts)
    else:
        rest_verts = rest_joints = None

    if correctives is not None:
        import correctives_jax as cj
        if rest_verts is None:
            rest_verts, rest_joints = layer.prepare_identity(id_coeffs[None], repose_to_bind_pose=False, skeleton_fit="linear")
            rest_verts, rest_joints = rest_verts[0], rest_joints[0]
        for i in range(T):
            R = jax.vmap(axis_angle_to_rotmat)(jnp.asarray(soma_poses[i]))   # (J,3,3)
            offset = cj.corrective_offsets(R, correctives)                   # (V,3)
            out = layer.pose(R[None], jnp.asarray(transl_arr[i:i+1]),
                             (rest_verts + offset)[None], rest_joints[None])
            verts.append(np.array(out.vertices[0]))
            joints.append(np.array(out.joints[0]))
        verts_arr = np.stack(verts, axis=0)
        joints_seq = np.stack(joints, axis=0)
        joints_seq = _mask_nonanatomical_joints(joints_seq, layer.joint_names)
        return verts_arr, np.array(layer.faces), joints_seq, np.asarray(layer._parents_np)

    for i in range(T):
        if rest_verts is not None:
            R = jax.vmap(axis_angle_to_rotmat)(jnp.asarray(soma_poses[i]))
            out = layer.pose(R[None], jnp.asarray(transl_arr[i:i+1]),
                             rest_verts[None], rest_joints[None])
            verts.append(np.array(out.vertices[0]))
            joints.append(np.array(out.joints[0]))
        else:
            out = layer(SOMAParams(
                poses=jnp.asarray(soma_poses[i]),
                transl=jnp.asarray(transl_arr[i]),
                identity_coeffs=id_coeffs,
            ))
            verts.append(np.array(out.vertices))
            joints.append(np.array(out.joints))
    joints_seq = _mask_nonanatomical_joints(np.stack(joints, axis=0), layer.joint_names)
    return (np.stack(verts, axis=0), np.array(layer.faces),
            joints_seq, np.asarray(layer._parents_np))


def load_smplx_with_split_basis(path, num_betas=10, num_expression_coeffs=10):
    """Load SMPL-X tolerating the official packed-basis layout.

    Official SMPL-X files store shape + expression blend shapes concatenated in
    `shapedirs` (300 shape + 100 expression). `soma_jax.body_models.load_smpl_data`
    now performs that split itself, so this is a thin alias kept for callers that
    still import it.
    """
    from soma_jax.body_models.smplx import SMPLXModel
    return SMPLXModel.load(
        path,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
    )


def compute_model_sequence(name, model, poses_seq, transl_seq):
    """Run a SMPL-family model over a (T, J, 3) pose sequence; return (T, V, 3) verts, faces."""
    import numpy as np
    builders = {"SMPL": _pose_smpl, "SMPL-H": _pose_smplh, "SMPL-X": _pose_smplx}
    params = builders[name](model, poses_seq, transl_seq)
    out = model(params)
    return np.array(out.vertices), np.array(model.faces)


def _annotate(img_arr, label):
    """Draw the model name onto a rendered frame."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(img_arr).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    # Black outline for legibility against any background.
    x, y = 12, 10
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), label, fill=(0, 0, 0), font=font)
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    import numpy as np
    return np.array(img)


def sample_pose_sequence(n_joints: int, n_frames: int, seed: int = 42) -> np.ndarray:
    """Generate a smooth pose sequence using sinusoidal joint motion.

    Returns:
        (n_frames, n_joints, 3) axis-angle pose sequence.
    """
    rng = np.random.default_rng(seed)
    # Per-joint amplitude and phase (zero for root joint)
    amps = rng.uniform(0.0, 0.4, (n_joints, 3))
    amps[0] = 0  # root stays at identity
    phases = rng.uniform(0, 2 * np.pi, (n_joints, 3))

    t = np.linspace(0, 2 * np.pi, n_frames)[:, None, None]
    poses = amps[None] * np.sin(t + phases[None])
    return poses.astype(np.float32)


# Which SOMA joints to hold back from the skeleton overlay. Upstream SOMA-X
# draws the FULL 78-joint tree — eyes, jaw, and every fingertip/toe tip included
# (see third_party/SOMA-X/tools/vis_pyrender.py::build_skeleton_mesh, which
# spheres every joint and bones every valid parent, with no face masking). To
# match it we now draw them all; the ONLY joint held back is "Root", the
# Mixamo-style floor-reference control node — it sits at the world origin
# (self-parented, zero skin weight), so drawing it adds a stray bone from the
# floor up to the pelvis that is not part of the body skeleton. Everything
# anatomical (eyes, jaw, fingertips, toes) is shown.
_NONANATOMICAL_JOINT_TOKENS = ("Root",)
_KEEP_JOINTS_ALWAYS = frozenset()   # nothing needs force-keeping now


def _joint_depth(j, parents):
    """Depth of joint ``j`` in the tree (root = 0); used to fill parents first."""
    d = 0
    while 0 <= parents[j] != j and d < 200:
        j = parents[j]; d += 1
    return d


def _soma_joints_for_model(rest_verts, faces, soma_wrap_path, J_regressor, posed_seq,
                           parents=None, canonical_joints=None):
    """Compute SOMA joint positions in this model's coordinate frame.

    Pipeline:
      1. Load the SOMA-topology wrap of the model's rest mesh
         (rest model verts reordered + barycentrically interpolated to SOMA
         topology; ships with the nvidia/SOMA-X HF assets as
         ``<MODEL>/SOMA_wrap.obj``).
      2. Solve barycentric coordinates of each SOMA-topology vertex on the
         model's rest triangles.
      3. For each posed frame, apply those barycentric weights to the model's
         posed triangles, yielding a SOMA-topology posed mesh in the model's
         coordinate frame.
      4. Apply the canonical SOMA J_regressor.

    This lets us draw the unified SOMA 78-joint skeleton anchored to *each*
    body model's actual posed anatomy, demonstrating that the SOMA rig fits
    any identity-model shape (SMPL, SMPL-X, MHR, Anny, ...).

    Args:
        rest_verts: (V_model, 3) model rest verts (zero pose, zero shape).
        faces:      (F, 3) model faces.
        soma_wrap_path: path to ``<MODEL>/SOMA_wrap.obj``.
        J_regressor: (J=78, V_soma=18056) SOMA joint regressor.
        posed_seq:  (T, V_model, 3) posed mesh sequence in this model's frame.

    Returns:
        (T, J, 3) SOMA joint positions in the model's coordinate frame.
    """
    from build_identity_packs import load_obj_soma_x_style, compute_bary
    wrap = load_obj_soma_x_style(soma_wrap_path)
    V_wrap = np.asarray(wrap.vertices, dtype=np.float32)
    face_ids, bary = compute_bary(rest_verts.astype(np.float32),
                                    faces.astype(np.int32), V_wrap)
    tris_seq = posed_seq[:, faces[face_ids]]                              # (T, V_soma, 3, 3)
    soma_topo_seq = (bary[None, :, :, None] * tris_seq).sum(axis=2)        # (T, V_soma, 3)
    soma_joints = np.einsum("jv,tvd->tjd", J_regressor, soma_topo_seq)       # (T, J, 3)
    # Zero-skin-weight joints (eyes and the fingertip/toe tips) have empty
    # regressor rows, so the einsum places them at (0,0,0). Put them where they
    # belong: their canonical parent-relative offset added to the (regressed)
    # parent, resolved parent-before-child so chained tips inherit a filled
    # parent. Without this they'd collapse to the world origin and web the body.
    if parents is not None and canonical_joints is not None:
        parents = np.asarray(parents, dtype=int)
        can = np.asarray(canonical_joints, dtype=np.float32)
        empty = np.abs(J_regressor).sum(axis=1) < 1e-6                        # (J,)
        for j in sorted(range(len(parents)), key=lambda k: _joint_depth(k, parents)):
            if not empty[j]:
                continue
            p = int(parents[j])
            if p < 0 or p == j:
                soma_joints[:, j] = can[j]
            else:
                soma_joints[:, j] = soma_joints[:, p] + (can[j] - can[p])
    return soma_joints.astype(np.float32)


def _mask_nonanatomical_joints(joints_seq, joint_names):
    """NaN-mask non-anatomical joints so the skeleton overlay draws cleanly."""
    names = [str(n) for n in np.asarray(joint_names)]
    for j, nm in enumerate(names):
        if nm in _KEEP_JOINTS_ALWAYS:
            continue
        if any(tok in nm for tok in _NONANATOMICAL_JOINT_TOKENS):
            joints_seq[:, j] = np.nan
    return joints_seq


def _build_skeleton_mesh(joints, parents, radius):
    """Build a single trimesh combining joint spheres + bone cylinders.

    ``radius`` is the maximum sphere/bone radius (sized from body extent).
    Per-joint radius scales with the bone-to-parent length so finger and face
    chains aren't drawn at body-bone thickness; we'd otherwise lose individual
    finger joints inside a single fat sphere.
    """
    import trimesh
    geoms = []
    valid = np.isfinite(joints).all(axis=1)  # joints set to NaN are skipped
    min_r = max(radius * 0.25, 0.002)
    for j in range(len(joints)):
        if not valid[j]:
            continue
        p = int(parents[j])
        if 0 <= p < len(joints) and valid[p]:
            bone_len = float(np.linalg.norm(joints[j] - joints[p]))
            r = float(np.clip(bone_len * 0.18, min_r, radius))
        else:
            r = radius
        s = trimesh.creation.uv_sphere(radius=r, count=[8, 8])
        s.apply_translation(joints[j])
        s.visual.vertex_colors = np.tile(
            (np.array([*_JOINT_COLOR, 1.0]) * 255).astype(np.uint8), (len(s.vertices), 1))
        geoms.append(s)
        if p < 0 or not (0 <= p < len(joints)) or not valid[p]:
            continue
        a, b = joints[j], joints[p]
        if np.linalg.norm(a - b) < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=r * 0.5, segment=np.array([a, b]), sections=8)
        cyl.visual.vertex_colors = np.tile(
            (np.array([*_BONE_COLOR, 1.0]) * 255).astype(np.uint8), (len(cyl.vertices), 1))
        geoms.append(cyl)
    return trimesh.util.concatenate(geoms)


def _make_ground_and_shadow(vertices, faces, ground_y=0.0, ground_extent=20.0,
                            shadow_eps=0.003):
    """Build a ground-plane quad + a projection-shadow mesh in one shot.

    The shadow is the body mesh with `y` collapsed onto the ground plane (a
    parallel-light projection from directly overhead). Drawn just above the
    plane with low alpha so it reads as a contact shadow without needing a
    real shadow-map pass.
    """
    import trimesh
    import pyrender
    g = ground_extent
    ground_verts = np.array([
        [-g, ground_y, -g], [g, ground_y, -g],
        [ g, ground_y,  g], [-g, ground_y,  g],
    ], dtype=np.float32)
    # CCW from above so the surface normal points +Y. Default back-face culling
    # would hide a plane whose normal points -Y from a camera looking down at it.
    ground_faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    ground_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.88, 0.88, 0.92, 1.0],
        metallicFactor=0.0, roughnessFactor=0.95, alphaMode="OPAQUE",
    )
    ground_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=ground_verts, faces=ground_faces, process=False),
        material=ground_mat, smooth=False,
    )
    # Shadow: project all body verts to y ≈ ground. We keep it OPAQUE rather
    # than translucent — many coplanar body triangles project on top of each
    # other, and alpha-blending those varies frame-to-frame with mesh
    # deformation (different occlusion order ⇒ visibly flickering color). One
    # solid grey silhouette is stable. To avoid Z-fighting between the shadow
    # polys and the ground plane we put the shadow just above the plane.
    shadow_verts = vertices.copy()
    shadow_verts[:, 1] = ground_y + shadow_eps
    shadow_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.32, 0.32, 0.35, 1.0],
        metallicFactor=0.0, roughnessFactor=1.0, alphaMode="OPAQUE",
        doubleSided=True,
    )
    shadow_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=shadow_verts, faces=faces, process=False),
        material=shadow_mat, smooth=False,
    )
    return ground_mesh, shadow_mesh


def render_mesh_png(vertices, faces, output_path, width=512, height=512, color=(0.7, 0.7, 0.85),
                    wireframe=False, joints=None, parents=None, body_alpha=1.0,
                    camera_pose=None, ground=False):
    """Render a mesh to PNG using pyrender; returns numpy image array.

    If `joints` and `parents` are provided, the skeleton (bone cylinders + joint
    spheres) is drawn inside the body, which is rendered semi-transparent
    (`body_alpha`) so the rig shows through.

    `camera_pose`: (4, 4) world transform of the camera. If None, the camera is
        framed per-mesh (the legacy single-PNG path). Pass a shared pose across
        side-by-side columns to get a consistent floor line.
    `ground`: when True, add a ground-plane quad at y=0 plus a projection-shadow
        copy of the body to ground the character visually.
    """
    import trimesh
    import pyrender
    show_skeleton = joints is not None and parents is not None
    alpha = body_alpha if show_skeleton else 1.0

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color, alpha],
        metallicFactor=0.1, roughnessFactor=0.6,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True, wireframe=wireframe)
    scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35])

    if ground:
        ground_mesh, shadow_mesh = _make_ground_and_shadow(vertices, faces)
        scene.add(ground_mesh)
        scene.add(shadow_mesh)

    # Add skeleton first (opaque) so the translucent body blends over it.
    if show_skeleton:
        extent = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        skel = _build_skeleton_mesh(np.asarray(joints), np.asarray(parents), radius=extent * 0.005)
        scene.add(pyrender.Mesh.from_trimesh(skel, smooth=False))
    scene.add(mesh)

    if camera_pose is not None:
        cam_pose = np.asarray(camera_pose, dtype=np.float32)
    else:
        # Legacy per-mesh framing: distance scales with vertical extent.
        center = vertices.mean(axis=0)
        body_height = float(vertices[:, 1].max() - vertices[:, 1].min())
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = center + np.array([0.0, 0.0, body_height * 1.35 + 0.3])
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(cam, pose=cam_pose)

    # Multi-directional lighting
    light_main = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light_main, pose=cam_pose)

    light_fill = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=2.0)
    lp_fill = np.eye(4)
    lp_fill[:3, :3] = np.array([[0.707, 0, 0.707], [0, 1, 0], [-0.707, 0, 0.707]])
    scene.add(light_fill, pose=lp_fill)

    light_back = pyrender.DirectionalLight(color=[1.0, 0.9, 0.9], intensity=2.0)
    lp_back = np.eye(4)
    lp_back[:3, :3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    scene.add(light_back, pose=lp_back)

    # Cache the OffscreenRenderer per (w, h): creating + deleting an EGL
    # context every frame dominates side-by-side render time at scale (a
    # 200-frame × 6-column pass was hitting ~15+ min before this).
    cache = globals().setdefault("_OFFSCREEN_RENDERER_CACHE", {})
    key = (width, height)
    renderer = cache.get(key)
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(width, height)
        cache[key] = renderer
    color, _ = renderer.render(scene)

    if output_path is not None:
        try:
            from PIL import Image
            Image.fromarray(color).save(output_path)
        except ImportError:
            import imageio.v2 as imageio
            imageio.imwrite(output_path, color)
    return color


def main():
    args = parse_args()
    import jax.numpy as jnp
    from soma_jax import SOMALayer, SOMAParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SOMA model...")
    layer = SOMALayer.load(args.soma_model, identity_model_type="soma")
    n_joints = len(layer.joint_names)
    faces = np.array(layer.faces)
    print(f"  Joints: {n_joints}, Vertices: {layer.v_template.shape[0]}, Faces: {faces.shape[0]}")

    # 1. Rest mesh (zero shape)
    rest_verts, _ = layer.prepare_identity(jnp.zeros((1, 10)), repose_to_bind_pose=False, skeleton_fit="linear")
    verts_rest = np.array(rest_verts[0])
    print("\n[1/3] Computed rest mesh (zero shape)")

    # 2. Custom shape
    rng = np.random.default_rng(0)
    coeffs = jnp.array(rng.standard_normal((1, 10)).astype(np.float32) * 0.5)
    custom_verts, _ = layer.prepare_identity(coeffs, repose_to_bind_pose=False, skeleton_fit="linear")
    verts_custom = np.array(custom_verts[0])
    print("[2/3] Computed mesh with custom shape coefficients")

    # 3. Animation
    poses_seq = sample_pose_sequence(n_joints, args.num_frames)
    transl = np.zeros((args.num_frames, 3), dtype=np.float32)
    posed_verts_seq = []
    for i in range(args.num_frames):
        params = SOMAParams(
            poses=jnp.asarray(poses_seq[i]),
            transl=jnp.asarray(transl[i]),
            identity_coeffs=jnp.zeros(10),
        )
        out = layer(params)
        posed_verts_seq.append(np.array(out.vertices))
    print(f"[3/3] Computed {args.num_frames}-frame pose sequence")

    if args.no_render:
        print(f"\nSkipping render (--no-render). Meshes computed.")
        return

    try:
        import pyrender
        import trimesh  # noqa: F401
    except ImportError:
        print("\nNOTE: Install pyrender + trimesh for rendering: pip install pyrender trimesh")
        return

    wireframe_mode = getattr(args, "wireframe", False)
    # In --side-by-side mode the 01–05 single-model previews + demo.gif are
    # redundant noise: the side-by-side artifact already shows every model
    # under the same motion. Skip them.
    if not args.side_by_side:
        print("\nRendering still images...")
        render_mesh_png(verts_rest, faces, output_dir / "01_rest.png", args.width, args.height, wireframe=wireframe_mode)
        render_mesh_png(verts_custom, faces, output_dir / "02_custom_shape.png", args.width, args.height, color=(0.85, 0.7, 0.7), wireframe=wireframe_mode)
        render_mesh_png(posed_verts_seq[args.num_frames // 2], faces, output_dir / "03_posed.png", args.width, args.height, color=(0.7, 0.85, 0.7), wireframe=wireframe_mode)
        print(f"  Wrote 3 PNG images to {output_dir}")

        if args.gif:
            print(f"\nRendering animation → {args.gif}...")
            try:
                from PIL import Image
                frames = []
                for i, v in enumerate(posed_verts_seq):
                    img = render_mesh_png(v, faces, None, args.width, args.height, color=(0.7, 0.7, 0.85), wireframe=wireframe_mode)
                    frames.append(Image.fromarray(img))
                frames[0].save(
                    args.gif, save_all=True, append_images=frames[1:],
                    duration=80, loop=0, optimize=True,
                )
                print(f"  Saved {args.gif}")
            except ImportError:
                try:
                    import imageio.v2 as imageio
                    frames = []
                    for v in posed_verts_seq:
                        img = render_mesh_png(v, faces, None, args.width, args.height)
                        frames.append(img)
                    imageio.mimsave(args.gif, frames, duration=0.08)
                    print(f"  Saved {args.gif} (via imageio)")
                except ImportError:
                    print("  Need Pillow or imageio for GIF output: pip install Pillow")

        # Demonstrate SMPL forward pass if model provided
        if args.smpl_model:
            print(f"\n[Bonus] Loading SMPL (SMPL-JAX) from {args.smpl_model}...")
            try:
                import jax.numpy as _jnp
                import motion_pipeline as mp
                from smpl_jax import SMPLParams
                smpl = mp.load_smpl_jax(args.smpl_model)
                out = smpl(SMPLParams(betas=_jnp.zeros((1, smpl.num_betas)), body_pose=_jnp.zeros((1, 69)),
                                      global_orient=_jnp.zeros((1, 3)), transl=_jnp.zeros((1, 3))))
                print(f"  SMPL output: vertices {out.vertices.shape}, joints {out.joints.shape}")
                render_mesh_png(np.array(out.vertices[0]), np.array(smpl.faces), output_dir / "04_smpl.png",
                                args.width, args.height, color=(0.85, 0.85, 0.7))
                print(f"  Wrote {output_dir}/04_smpl.png")
            except Exception as e:
                print(f"  SMPL render failed: {e}")

        if args.smplx_model:
            print(f"\n[Bonus] Loading SMPL-X (SMPL-JAX) from {args.smplx_model}...")
            try:
                import jax.numpy as _jnp
                import motion_pipeline as mp
                from smpl_jax import SMPLXParams
                smplx = mp.load_smplx_jax(args.smplx_model)
                out = smplx(SMPLXParams(
                    betas=_jnp.zeros((1, smplx.num_betas)), body_pose=_jnp.zeros((1, 63)),
                    global_orient=_jnp.zeros((1, 3)), transl=_jnp.zeros((1, 3)),
                    expression=_jnp.zeros((1, smplx.num_expression_coeffs)), jaw_pose=_jnp.zeros((1, 3)),
                    leye_pose=_jnp.zeros((1, 3)), reye_pose=_jnp.zeros((1, 3)),
                    left_hand_pose=_jnp.zeros((1, 45)), right_hand_pose=_jnp.zeros((1, 45))))
                print(f"  SMPL-X output: vertices {out.vertices.shape}, joints {out.joints.shape}")
                render_mesh_png(np.array(out.vertices[0]), np.array(smplx.faces), output_dir / "05_smplx.png",
                                args.width, args.height, color=(0.7, 0.85, 0.85))
                print(f"  Wrote {output_dir}/05_smplx.png")
            except Exception as e:
                print(f"  SMPL-X render failed: {e}")

    if args.side_by_side:
        print("\nRendering side-by-side comparison across body models...")
        # SMPL/SMPL-X come from SMPL-JAX; SMPL-H falls back to soma_jax (SMPL-JAX
        # has no SMPL-H), and is only used in the legacy synthetic path if data exists.
        from soma_jax.body_models import SMPLHModel

        columns = []  # (label, verts_seq, faces, color, joints_seq|None, parents|None)
        view_rot = None  # optional global rotation applied at render time

        if args.motion or args.bvh_motion:
            import motion_pipeline as mp
            # Both BVH and SMPL-X motions live in a Z-up world frame
            # (the Hips channels carry the bind orientation that takes the SOMA
            # rest Y-up template to the motion world's Z-up). Apply Z-up→Y-up at
            # render time so the body appears upright in our Y-up camera.
            view_rot = _ZUP_TO_YUP
            rng = np.random.default_rng(args.seed)
            soma_poses, soma_extras, smplx_model = None, None, None

            if args.bvh_motion:
                # ---- BVH: already in SOMA skeleton format. ----
                # Faithful SOMA-X path: full bind_pose_world + t_pose_world (joint_orient)
                # + level-order FK + LBS, mirroring third_party/SOMA-X BatchedSkinning.
                from bvh_parser import load_soma_bvh
                from soma_x_skinning import SomaXSkinning
                from scipy.sparse import csc_matrix
                bvh = load_soma_bvh(args.bvh_motion)
                N_src = bvh["n_frames"]
                # --target-fps overrides --num-frames: pick T so the GIF plays at
                # that frame rate in real time. Capped at --num-frames so very long
                # motions don't blow up render time; the cap subsamples the clip
                # while keeping GIF playback in real time (per-frame ms scales).
                if args.target_fps is not None:
                    T_desired = int(round(bvh["source_duration_s"] * args.target_fps))
                    T = max(2, min(T_desired, args.num_frames, N_src))
                else:
                    T = min(args.num_frames, N_src)
                idx = np.linspace(0, N_src - 1, T).astype(int)
                bvh_poses = bvh["poses"][idx]                                # (T, J, 3) rel-to-T-pose
                bvh_rotmats = bvh["rotmats"][idx]                            # (T, J, 3, 3) exact
                bvh_trans = bvh["root_translation"][idx]                     # (T, 3) meters
                hf_soma = dict(np.load(Path(args.hf_dir) / "SOMA_neutral.npz", allow_pickle=True))
                # SOMA-X rig in meters (HF translations are cm).
                bind_world = hf_soma["bind_pose_world"].astype(np.float32).copy()
                bind_world[:, :3, 3] *= 0.01
                bind_shape_cm = hf_soma["bind_shape"].astype(np.float32)
                bind_shape_m = bind_shape_cm * 0.01
                t_pose_world = hf_soma["t_pose_world"].astype(np.float32)
                W_full = csc_matrix((hf_soma["skinning_weights_data"],
                                     hf_soma["skinning_weights_indices"],
                                     hf_soma["skinning_weights_indptr"]),
                                    shape=tuple(hf_soma["skinning_weights_shape"])).toarray().astype(np.float32)
                names = [str(n) for n in hf_soma["joint_names"]]
                hips_idx = names.index("Hips")
                parents = hf_soma["joint_parent_ids"].astype(int).copy(); parents[0] = 0
                soma_x_rig = dict(
                    parents=parents, weights=W_full, bind_world=bind_world,
                    bind_shape=bind_shape_m, t_pose_world=t_pose_world, hips_idx=hips_idx, names=names,
                    bvh_poses=bvh_poses, bvh_trans=bvh_trans,
                )

                # Use the exact Euler-derived rotmats (no axis-angle round-trip — that
                # was lossy near π for joints like LeftLeg whose magnitude reaches 4.5 rad).
                R_all = bvh_rotmats.astype(np.float32)
                # Per-identity rendering: SOMA uses bind_shape; identities re-skin via SomaXSkinning.rebind.
                motion = {
                    "n_frames": T, "trans": bvh_trans,
                    "source_fps": bvh["source_fps"],
                    "source_total_frames": bvh["source_total_frames"],
                    "source_duration_s": bvh["source_duration_s"],
                }
                # Stash for the SOMA-family loop below.
                soma_poses = None         # signals "use SOMA-X skinning path"
                soma_extras = None
                motion = {
                    "n_frames": T, "trans": bvh["root_translation"][idx],
                    "source_fps": bvh["source_fps"],
                    "source_total_frames": bvh["source_total_frames"],
                    "source_duration_s": bvh["source_duration_s"],
                }
                print(f"  BVH motion: {args.bvh_motion} "
                      f"({N_src} frames @ {bvh['source_fps']:.0f} FPS → subsampled to {T})")
            else:
                # ---- SMPL-X path: native columns + inverse-LBS retargeting. ----
                motion = mp.load_smplx_motion(args.motion, num_frames=args.num_frames)
                T = motion["n_frames"]
                print(f"  Motion: {args.motion} ({T} frames)")
                if args.random_shape:
                    print("  Random-shape mode: sampling identity coefficients per model")

                if args.smplx_model:
                    smplx_model = mp.load_smplx_jax(args.smplx_model)

                # SMPL-family columns, driven natively from the shared SMPL-X pose.
                smpl_native = [
                    ("SMPL",   args.smpl_model,   lambda p: mp.load_smpl_jax(p),  mp.smpl_params),
                    ("SMPL-X", args.smplx_model,  lambda p: smplx_model,          mp.smplx_params),
                ]
                for name, path, loader, builder in smpl_native:
                    if not path:
                        print(f"  [{name}] no model path — skipping column")
                        continue
                    try:
                        import jax.numpy as _jnp
                        model = loader(path)
                        params = builder(model, motion)
                        if args.random_shape:
                            betas = rng.standard_normal((T, model.num_betas)).astype(np.float32) * 1.2
                            params = params._replace(betas=_jnp.asarray(betas))
                        out = model(params)
                        columns.append((name, np.asarray(out.vertices), np.asarray(model.faces),
                                        _MODEL_COLORS[name], np.asarray(out.joints),
                                        np.asarray(model._parents_np)))
                        print(f"  [{name}] native motion → {np.asarray(out.vertices).shape}")
                    except Exception as e:
                        print(f"  [{name}] failed: {e}")

                if smplx_model is not None:
                    mode = f"analytical+{args.refine_iters} autograd FK" if args.refine_iters else "analytical"
                    print(f"  Retargeting SMPL-X motion → SOMA skeleton (inverse-LBS, {mode})...")
                    soma_poses, soma_extras = mp.smplx_motion_to_soma_poses(
                        smplx_model, layer, motion, hf_dir=args.hf_dir,
                        refine_iters=args.refine_iters, return_extras=True)
                    err = soma_extras["per_vertex_error"]
                    print(f"    retargeting error: mean {err.mean()*100:.2f} cm  max {err.max()*100:.2f} cm")

            # Low-LOD (SOMA-X low_lod): swap SOMA-family assets to the low-LOD set.
            n_lod = None
            soma_render_model = args.soma_model
            id_paths = {"mhr": args.mhr_identity, "anny": args.anny_identity,
                        "garment_measurement": args.garment_identity}
            if args.low_lod:
                lod_dir = Path(args.lowlod_dir)
                if (lod_dir / "SOMA_neutral.npz").exists():
                    n_lod = int(np.load(lod_dir / "n_low.npy"))
                    soma_render_model = str(lod_dir / "SOMA_neutral.npz")
                    for t, fn in [("mhr", "identity_mhr.npz"), ("anny", "identity_anny.npz"),
                                  ("garment_measurement", "identity_garment.npz")]:
                        if id_paths[t] and (lod_dir / fn).exists():
                            id_paths[t] = str(lod_dir / fn)
                    print(f"  Low-LOD mode: {n_lod} verts (assets from {lod_dir})")
                else:
                    print(f"  [low-lod] {lod_dir} missing — run tools/build_lowlod.py; using full-res")

            # Unified Pose Correctives (Beta): load the SOMA-X corrective MLP once
            # and share it across all SOMA-family identities to fix LBS artifacts.
            correctives = None
            cpath = Path(args.hf_dir) / "correctives_model.pt"
            if cpath.exists():
                import correctives_jax as cj
                correctives = cj.load_correctives(str(cpath), np.asarray(layer._parents_np), n_lod=n_lod)
                print(f"  Loaded unified pose correctives ({correctives['n_vertices']} verts)")
            else:
                print(f"  [correctives] {cpath} not found — meshes will show raw LBS artifacts")

            soma_identity_specs = [
                ("SOMA",    "soma",                None),
                ("MHR",     "mhr",                 id_paths["mhr"]),
                ("Anny",    "anny",                id_paths["anny"]),
                ("Garment", "garment_measurement", id_paths["garment_measurement"]),
            ]

            # ---- BVH path: faithful SOMA-X skinning per identity. ----
            if args.bvh_motion:
                from soma_x_skinning import SomaXSkinning
                from build_soma_rig import build_regressor as _build_regressor

                # SkeletonTransfer-equivalent: per-joint affine regressor (linear RBF
                # restricted to each joint's skinning support). Fit on the canonical
                # rig so J_reg @ bind_shape_canonical = bind_world joints. Applied to
                # any identity's bind_shape, it produces identity-fit joint positions.
                J = len(soma_x_rig["parents"])
                _children = {j: [k for k in range(J) if soma_x_rig["parents"][k] == j and k != j]
                             for j in range(J)}
                _Jreg = _build_regressor(
                    soma_x_rig["bind_shape"].astype(np.float64),
                    soma_x_rig["bind_world"][:, :3, 3].astype(np.float64),
                    soma_x_rig["weights"], soma_x_rig["parents"], _children,
                ).astype(np.float32)

                # Face joints (eyes, jaw) and other no-skin-weight joints have
                # empty rows in the regressor, so `_Jreg @ bs` gives (0, 0, 0).
                # For those we fall back to "translate canonical bind position
                # along with its parent" so face joints follow Head correctly.
                _Jreg_empty = (np.abs(_Jreg).sum(axis=1) < 1e-6)
                _canonical_joints = soma_x_rig["bind_world"][:, :3, 3].copy()

                def _fit_bind_world(target_bind_shape):
                    """SkeletonTransfer-equivalent: keep canonical bind rotations,
                    swap in identity-fit joint positions. Drop-in for SOMA-X's
                    SkeletonTransfer.fit() when rotation fitting isn't required.
                    """
                    new_joints = _Jreg @ target_bind_shape          # (J, 3)
                    # For empty-regressor joints, replicate their canonical
                    # parent-relative offset on top of the regressed parent.
                    if _Jreg_empty.any():
                        for j in np.flatnonzero(_Jreg_empty):
                            p = int(soma_x_rig["parents"][j])
                            if p < 0 or p == j:
                                new_joints[j] = _canonical_joints[j]
                            else:
                                new_joints[j] = new_joints[p] + (
                                    _canonical_joints[j] - _canonical_joints[p])
                    out = soma_x_rig["bind_world"].copy()
                    out[:, :3, 3] = new_joints
                    return out

                sk = SomaXSkinning(
                    parents=soma_x_rig["parents"],
                    skinning_weights=soma_x_rig["weights"],
                    bind_world_transforms=soma_x_rig["bind_world"],
                    bind_shape=soma_x_rig["bind_shape"],
                    joint_orient_world=soma_x_rig["t_pose_world"],
                    hips_joint=soma_x_rig["hips_idx"],
                )

                # Unified Pose Correctives (Beta) — share one corrective MLP across
                # all SOMA-family identities, exactly like SOMA-X.
                bvh_correctives = None
                cpath = Path(args.hf_dir) / "correctives_model.pt"
                if cpath.exists():
                    import correctives_jax as cj
                    bvh_correctives = cj.load_correctives(str(cpath), soma_x_rig["parents"])
                    print(f"  Loaded unified pose correctives ({bvh_correctives['n_vertices']} verts)")

                view_rot = None    # SOMA-X rig is already Y-up; no view rotation
                for name, id_type, id_path in soma_identity_specs:
                    if id_type != "soma" and not id_path:
                        print(f"  [{name}] no identity asset — skipping column")
                        continue
                    try:
                        # Per-identity bind_shape in SOMA topology (meters).
                        if id_type == "soma":
                            bs = soma_x_rig["bind_shape"]
                        elif name == "MHR" and Path(args.mhr_rig).exists() and (
                                args.mhr_scale is not None or args.mhr_subject is not None):
                            import mhr_jax
                            rig = mhr_jax.load_mhr_rig(args.mhr_rig)
                            pack = dict(np.load(id_path, allow_pickle=False))
                            if args.mhr_subject:
                                sub = np.load(args.mhr_subject, allow_pickle=False)
                                mhr_id = sub["identity_params"][0].astype(np.float32)
                                mhr_sc = sub["scale_params"][0].astype(np.float32)
                                print(f"  [MHR] subject: {args.mhr_subject}")
                            else:
                                mhr_id = None
                                mhr_sc = np.full(rig["num_scale"], args.mhr_scale or 0.0, np.float32)
                            bs = mhr_jax.mhr_soma_rest(rig, pack, identity_coeffs=mhr_id, scale_params=mhr_sc)
                        else:
                            # Get identity-transferred rest in SOMA topology (meters)
                            # via soma_jax's identity model (one-shot, neutral coeffs).
                            tmp = SOMALayer.load(args.soma_model, identity_model_type=id_type,
                                                 identity_model_path=id_path)
                            rv, _ = tmp.prepare_identity(jnp.zeros((1, 10)), repose_to_bind_pose=False, skeleton_fit="linear")
                            bs = np.asarray(rv[0]).astype(np.float32)
                        # Anchor each identity's feet to Y=0 so the regressor-fit joints
                        # land inside the mesh consistently.
                        bs = bs.copy()
                        bs[:, 1] -= bs[:, 1].min()
                        # Fit the SOMA skeleton to THIS identity's bind shape (per-joint
                        # affine regressor over skinning support — SkeletonTransfer equivalent).
                        id_bind_world = _fit_bind_world(bs)

                        verts, joints, tworld_seq = [], [], []
                        for ti in range(T):
                            R = R_all[ti]
                            # Unified Pose Correctives: add the corrective offset to the
                            # identity bind shape, then skin. BVH gives absolute-frame
                            # rotations, so absolute_pose=True for the corrective too.
                            if bvh_correctives is not None:
                                import correctives_jax as cj
                                import jax.numpy as _jnp
                                off = np.asarray(cj.corrective_offsets(
                                    _jnp.asarray(R), bvh_correctives, absolute_pose=True))
                                rest_for_frame = bs + off
                            else:
                                rest_for_frame = bs
                            sk.rebind(id_bind_world, rest_for_frame)
                            # BVH stores rotations as ABSOLUTE skinning-frame
                            # rotations (bind orientation baked in), not T-pose-relative.
                            # hips_translation drives root motion (walk forward, jump up);
                            # without it the character pose-articulates in place.
                            v, Tw = sk.pose(R, hips_translation=soma_x_rig["bvh_trans"][ti],
                                            absolute_pose=True)
                            verts.append(v.astype(np.float32))
                            joints.append(Tw[:, :3, 3].astype(np.float32))
                            if name == "SOMA":
                                tworld_seq.append(Tw[:, :3, :3].astype(np.float32))
                        v_seq = np.stack(verts, axis=0)
                        j_seq = np.stack(joints, axis=0)
                        j_seq = _mask_nonanatomical_joints(j_seq, soma_x_rig["names"])
                        if name == "SOMA":
                            soma_posed_seq = v_seq    # cache canonical SOMA for SMPL/SMPL-X retarget
                            soma_world_rot_seq = np.stack(tworld_seq, axis=0)  # (T, J, 3, 3)
                        columns.append((name, v_seq, np.asarray(layer.faces),
                                        _MODEL_COLORS[name], j_seq, soma_x_rig["parents"]))
                        print(f"  [{name}] SOMA-X skinning → {v_seq.shape}")
                    except Exception as e:
                        print(f"  [{name}] failed: {e}")

                # --- SMPL / SMPL-X columns: direct SOMA→SMPL-X joint-rotation transfer ---
                # The SOMA-X PyTorch repo only ships SMPL→SOMA (smpl2soma). The
                # mathematically clean reverse, since the body articulation is
                # just rotations on shared anatomy, is to read SOMA's posed world
                # rotations (from the SOMA-X FK we already ran for the SOMA column)
                # and re-express them as SMPL-X parent-relative locals via a
                # name-based joint mapping. See tools/soma_to_smplx.py.
                _soma_world_rot_seq = locals().get("soma_world_rot_seq", None)
                if args.smplx_model and _soma_world_rot_seq is not None:
                    try:
                        import motion_pipeline as mp
                        from soma_to_smplx import smplx_poses_from_soma_world
                        from smpl_jax import SMPLXParams, SMPLParams
                        smplx_model = mp.load_smplx_jax(args.smplx_model)
                        print("  Retargeting SOMA→SMPL-X via direct name-based world-rotation transfer...")
                        # SOMA bind world orientation per joint (= t_pose_world[:, :3, :3]).
                        # Needed to strip SOMA's bind rotation from world rotations before
                        # re-applying them in SMPL-X's identity-bind frame.
                        _soma_orient_world = np.asarray(soma_x_rig["t_pose_world"][:, :3, :3])
                        poses_smplx = smplx_poses_from_soma_world(
                            _soma_world_rot_seq,
                            soma_x_rig["names"],
                            np.asarray(smplx_model._parents_np),
                            soma_orient_world=_soma_orient_world,
                        )
                        # SOMA's 78-joint skeleton already carries finger + face
                        # joints (Jaw, LeftEye/RightEye, and 4-segment fingers).
                        # smplx_poses_from_soma_world picks the first three finger
                        # segments off each SOMA finger to feed MANO's 3-joint
                        # chain, and copies jaw/eye rotations through directly,
                        # so the hand+face poses come straight from the BVH.
                        Tn = _soma_world_rot_seq.shape[0]
                        import jax.numpy as _jnp
                        # Use SOMA's per-frame hip translation so SMPL-X walks/jumps
                        # in lockstep with the SOMA-family columns. The Y component
                        # has a small constant offset between SOMA's hip bind height
                        # (~+1.0 m) and SMPL-X's pelvis rest (~-0.35 m); we let
                        # --ground-lock absorb that with its global per-column Y
                        # shift so all columns share a single floor.
                        bvh_trans_full = np.asarray(soma_x_rig["bvh_trans"], np.float32)
                        smplx_params = SMPLXParams(
                            betas=_jnp.zeros((Tn, smplx_model.num_betas), np.float32),
                            body_pose=_jnp.asarray(poses_smplx["body_pose"]),
                            global_orient=_jnp.asarray(poses_smplx["global_orient"]),
                            transl=_jnp.asarray(bvh_trans_full),
                            expression=_jnp.zeros((Tn, smplx_model.num_expression_coeffs), np.float32),
                            jaw_pose=_jnp.asarray(poses_smplx["jaw_pose"]),
                            leye_pose=_jnp.asarray(poses_smplx["leye_pose"]),
                            reye_pose=_jnp.asarray(poses_smplx["reye_pose"]),
                            left_hand_pose=_jnp.asarray(poses_smplx["left_hand_pose"]),
                            right_hand_pose=_jnp.asarray(poses_smplx["right_hand_pose"]),
                        )
                        out_x = smplx_model(smplx_params)
                        columns.append(("SMPL-X", np.asarray(out_x.vertices),
                                        np.asarray(smplx_model.faces), _MODEL_COLORS["SMPL-X"],
                                        np.asarray(out_x.joints),
                                        np.asarray(smplx_model._parents_np)))
                        print(f"  [SMPL-X] retargeted-from-SOMA → {np.asarray(out_x.vertices).shape}")
                        # SMPL: first 21 body joints of SMPL-X map to SMPL's body_pose joints 0..20.
                        if args.smpl_model:
                            smpl_model = mp.load_smpl_jax(args.smpl_model)
                            body23 = np.zeros((Tn, 23 * 3), np.float32)
                            body23[:, : 21 * 3] = poses_smplx["body_pose"]
                            smpl_params = SMPLParams(
                                betas=_jnp.zeros((Tn, smpl_model.num_betas), np.float32),
                                body_pose=_jnp.asarray(body23),
                                global_orient=_jnp.asarray(poses_smplx["global_orient"]),
                                transl=_jnp.asarray(bvh_trans_full),
                            )
                            out_s = smpl_model(smpl_params)
                            columns.append(("SMPL", np.asarray(out_s.vertices),
                                            np.asarray(smpl_model.faces), _MODEL_COLORS["SMPL"],
                                            np.asarray(out_s.joints),
                                            np.asarray(smpl_model._parents_np)))
                            print(f"  [SMPL] retargeted-from-SOMA → {np.asarray(out_s.vertices).shape}")
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"  [SMPL/SMPL-X retarget failed]: {e}")

                soma_identity_specs = []         # skip the SMPL-X-style loop below

            for name, id_type, id_path in soma_identity_specs:
                if soma_poses is None:
                    print(f"  [{name}] needs --smplx-model for retargeting — skipping")
                    continue
                if id_type != "soma" and not id_path:
                    print(f"  [{name}] no identity asset — skipping column")
                    continue
                try:
                    id_coeffs = (rng.standard_normal(10).astype(np.float32) * 1.0
                                 if args.random_shape else None)
                    # MHR full parametric rig with scale_params (pure-JAX port).
                    rest_override = None
                    use_mhr_rig = name == "MHR" and Path(args.mhr_rig).exists() and (
                        args.mhr_scale is not None or args.mhr_subject is not None)
                    if use_mhr_rig:
                        import mhr_jax
                        rig = mhr_jax.load_mhr_rig(args.mhr_rig)
                        pack = dict(np.load(id_path, allow_pickle=False))   # low-LOD pack if --low-lod
                        if args.mhr_subject:
                            # MHR subject: real identity + scale params.
                            sub = np.load(args.mhr_subject, allow_pickle=False)
                            mhr_id = sub["identity_params"][0].astype(np.float32)
                            mhr_sc = sub["scale_params"][0].astype(np.float32)
                            print(f"  [MHR] subject: {args.mhr_subject}")
                        else:
                            mhr_id = (rng.standard_normal(rig["num_identity"]).astype(np.float32) * 0.5
                                      if args.random_shape else None)
                            mhr_sc = np.full(rig["num_scale"], args.mhr_scale or 0.0, np.float32)
                            print(f"  [MHR] full rig: scale_params={args.mhr_scale}")
                        rest_override = mhr_jax.mhr_soma_rest(
                            rig, pack, identity_coeffs=mhr_id, scale_params=mhr_sc)
                    elif name == "SOMA" and soma_extras is not None and "soma_rest_verts" in soma_extras:
                        # SMPL-X→SOMA path: PoseInversion fit the rotations onto the
                        # SMPL-X-transferred SOMA rest (pelvis-centered, Y∈[-1.3,0.42]).
                        # Render on that same rest so rotations land on the right
                        # skeleton — using the canonical SOMA bind (Y∈[0,1.68]) would
                        # apply the rotations to a 1m-translated origin and stretch/
                        # shrink the body geometry.
                        rest_override = soma_extras["soma_rest_verts"]
                    transl_seq = (soma_extras.get("root_translation")
                                  if soma_extras is not None else None)
                    verts_seq, model_faces, joints_seq, soma_parents = pose_soma_identity_sequence(
                        soma_render_model, id_type, id_path, soma_poses, correctives=correctives,
                        identity_coeffs=id_coeffs, rest_verts_override=rest_override,
                        transl_seq=transl_seq,
                    )
                    columns.append((name, verts_seq, model_faces, _MODEL_COLORS[name],
                                    joints_seq, soma_parents))
                    print(f"  [{name}] SOMA-retargeted motion → {verts_seq.shape}")
                except Exception as e:
                    print(f"  [{name}] failed: {e}")

            # Export the retargeted SOMA motion (SOMA-X NPZ format).
            if args.export_soma_npz and soma_extras is not None:
                np.savez(
                    args.export_soma_npz,
                    poses=soma_poses.astype(np.float32),                        # (N, J, 3)
                    root_translation=soma_extras["root_translation"],          # (N, 3)
                    joint_names=soma_extras["joint_names"],
                    per_vertex_error=soma_extras["per_vertex_error"],          # (N, V)
                    identity_coeffs=np.zeros((10,), np.float32),
                )
                print(f"  Exported SOMA motion → {args.export_soma_npz}")
        else:
            # ---- Synthetic sinusoidal motion (legacy path). ----
            T = args.num_frames
            soma_J = poses_seq.shape[1]
            max_J = max(soma_J, 24)
            unified = np.zeros((T, max_J, 3), dtype=np.float32)
            unified[:, :soma_J] = poses_seq
            if max_J > soma_J:
                rng_ext = np.random.default_rng(7)
                amps_ext = rng_ext.uniform(0.0, 0.2, (max_J - soma_J, 3))
                phases_ext = rng_ext.uniform(0, 2 * np.pi, (max_J - soma_J, 3))
                t = np.linspace(0, 2 * np.pi, T)[:, None, None]
                unified[:, soma_J:] = amps_ext[None] * np.sin(t + phases_ext[None])
            transl_seq = transl

            import motion_pipeline as mp
            columns.append(("SOMA", np.stack(posed_verts_seq, axis=0), faces, _MODEL_COLORS["SOMA"], None, None))
            smpl_specs = [
                ("SMPL",   args.smpl_model,   lambda p: mp.load_smpl_jax(p)),
                ("SMPL-H", args.smplh_model,  lambda p: SMPLHModel.load(p)),
                ("SMPL-X", args.smplx_model,  lambda p: mp.load_smplx_jax(p)),
            ]
            for name, path, loader in smpl_specs:
                if not path:
                    print(f"  [{name}] no model path provided — skipping column")
                    continue
                try:
                    print(f"  [{name}] loading {path}")
                    model = loader(path)
                    verts_seq, model_faces = compute_model_sequence(name, model, unified, transl_seq)
                    columns.append((name, verts_seq, model_faces, _MODEL_COLORS[name], None, None))
                    print(f"    posed {T} frames → vertices {verts_seq.shape}")
                except Exception as e:
                    print(f"  [{name}] failed: {e}")

            soma_identity_specs = [
                ("MHR",     "mhr",                 args.mhr_identity),
                ("Anny",    "anny",                args.anny_identity),
                ("Garment", "garment_measurement", args.garment_identity),
            ]
            for name, id_type, id_path in soma_identity_specs:
                if not id_path:
                    print(f"  [{name}] no --{id_type}-identity asset provided — skipping column")
                    continue
                try:
                    print(f"  [{name}] loading SOMA-identity '{id_type}' from {id_path}")
                    verts_seq, model_faces = run_soma_identity_sequence(
                        args.soma_model, id_type, id_path, unified, transl_seq,
                    )
                    columns.append((name, verts_seq, model_faces, _MODEL_COLORS[name], None, None))
                    print(f"    posed {T} frames → vertices {verts_seq.shape}")
                except Exception as e:
                    print(f"  [{name}] failed: {e}")

        # ---- Optional: swap each non-SOMA column's skeleton to SOMA ----
        # For SMPL / SMPL-X columns, compute SOMA joints in that column's
        # coordinate frame via barycentric transfer and replace
        # (joints_seq, parents) so the overlay draws the unified SOMA rig.
        # SOMA-family columns (SOMA, MHR, Anny, Garment) already use the SOMA
        # skeleton (they all run through BatchedSkinning with SOMA poses), so
        # they need no change.
        if args.soma_skeleton_overlay:
            print("  Swapping SMPL/SMPL-X overlays to SOMA 78-joint skeleton...")
            hf_dir = Path(args.hf_dir)
            wrap_paths = {
                "SMPL":   hf_dir / "SMPL"  / "SOMA_wrap.obj",
                "SMPL-X": hf_dir / "SMPLX" / "SOMA_wrap.obj",
            }
            soma_parents = np.asarray(layer._parents_np)
            soma_J_regressor = np.asarray(layer.J_regressor)
            import motion_pipeline as mp
            import jax.numpy as _jnp
            new_columns = []
            for name, vseq, fcs, color, jseq, jparents in columns:
                wp = wrap_paths.get(name)
                if wp is None or not wp.exists():
                    new_columns.append((name, vseq, fcs, color, jseq, jparents))
                    continue
                try:
                    if name == "SMPL":
                        m_ = mp.load_smpl_jax(args.smpl_model)
                        from smpl_jax import SMPLParams
                        rest = m_(SMPLParams(
                            global_orient=_jnp.zeros((1, 3)),
                            body_pose=_jnp.zeros((1, 69)),
                            betas=_jnp.zeros((1, m_.num_betas)),
                            transl=_jnp.zeros((1, 3)),
                        ))
                    else:  # SMPL-X
                        m_ = mp.load_smplx_jax(args.smplx_model)
                        from smpl_jax import SMPLXParams
                        rest = m_(SMPLXParams(
                            betas=_jnp.zeros((1, m_.num_betas)),
                            body_pose=_jnp.zeros((1, 63)),
                            global_orient=_jnp.zeros((1, 3)),
                            transl=_jnp.zeros((1, 3)),
                            expression=_jnp.zeros((1, m_.num_expression_coeffs)),
                            jaw_pose=_jnp.zeros((1, 3)),
                            leye_pose=_jnp.zeros((1, 3)),
                            reye_pose=_jnp.zeros((1, 3)),
                            left_hand_pose=_jnp.zeros((1, 45)),
                            right_hand_pose=_jnp.zeros((1, 45)),
                        ))
                    rest_v = np.asarray(rest.vertices[0])
                    # Canonical SOMA joints let the wrap place the zero-weight
                    # joints (eyes, tips) that the regressor can't; None-safe.
                    _bpw = getattr(layer, "bind_pose_world", None)
                    _canon = np.asarray(_bpw)[:, :3, 3] if _bpw is not None else None
                    soma_j = _soma_joints_for_model(
                        rest_v, fcs, str(wp), soma_J_regressor, vseq,
                        parents=soma_parents, canonical_joints=_canon,
                    )
                    soma_j = _mask_nonanatomical_joints(soma_j, list(layer.joint_names))
                    new_columns.append((name, vseq, fcs, color, soma_j, soma_parents))
                    print(f"    [{name}] SOMA 78-joint skeleton via barycentric transfer "
                          f"({rest_v.shape[0]} -> {soma_J_regressor.shape[1]} verts)")
                except Exception as e:
                    print(f"    [{name}] SOMA-skeleton swap failed: {e} — keeping native")
                    new_columns.append((name, vseq, fcs, color, jseq, jparents))
            columns = new_columns

        # Render each column at each frame, then hstack.
        print(f"  Compositing {len(columns)}-column animation ({T} frames)...")
        # Pre-transform every column once into camera-Y-up space and (if
        # --ground-lock) apply a SINGLE constant Y shift per column so the
        # lowest foot across the whole clip sits on y=0. This shares a ground
        # plane across body models with different bind heights but keeps the
        # actual BVH motion (jumps, root drift, walking forward) intact — the
        # earlier per-frame shift was effectively a moving camera and made
        # vertical motion invisible.
        view_seqs = []
        view_joint_seqs = []
        for name, vs, model_faces, color, jseq, parents in columns:
            v_view = vs @ view_rot.T if view_rot is not None else vs.copy()
            j_view = (jseq @ view_rot.T if (view_rot is not None and jseq is not None)
                      else (jseq if jseq is not None else None))
            if args.ground_lock:
                floor_global = float(v_view[..., 1].min())
                v_view = v_view.copy()
                v_view[..., 1] -= floor_global
                if j_view is not None:
                    j_view = j_view.copy()
                    j_view[..., 1] -= floor_global
            view_seqs.append(v_view)
            view_joint_seqs.append(j_view)

        # Shared camera: one fixed pose for every frame and every column,
        # framed so the union of XYZ extents across ALL frames + ALL columns
        # stays in view. The character can now translate (walk forward, jump
        # up) without the camera following them; a wider FOV / pulled-back
        # distance keeps the full motion in frame.
        fov_y = np.pi / 3.0
        aspect = float(args.width) / float(args.height)
        all_min = np.array([+np.inf, +np.inf, +np.inf], dtype=np.float64)
        all_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        for vs in view_seqs:
            all_min = np.minimum(all_min, vs.reshape(-1, 3).min(axis=0))
            all_max = np.maximum(all_max, vs.reshape(-1, 3).max(axis=0))
        center = (all_min + all_max) * 0.5
        span = all_max - all_min                                  # (3,) world units
        # Distance needed so vertical and horizontal extents both fit; +20%
        # margin so the body silhouette isn't flush against the frame edge.
        margin = 1.20
        dist_v = (span[1] * 0.5 * margin) / np.tan(fov_y * 0.5)
        dist_h = (span[0] * 0.5 * margin) / np.tan(fov_y * 0.5) / aspect
        cam_distance = max(dist_v, dist_h) + span[2] * 0.5
        shared_target = np.array([center[0], center[1], center[2]], dtype=np.float32)
        shared_cam_pose = np.eye(4, dtype=np.float32)
        shared_cam_pose[:3, 3] = shared_target + np.array(
            [0.0, 0.0, cam_distance], dtype=np.float32)
        print(f"  Shared camera: target={shared_target.round(2).tolist()}, "
              f"distance={cam_distance:.2f}m, "
              f"motion XYZ span=({span[0]:.2f}, {span[1]:.2f}, {span[2]:.2f}) m")

        composite_frames = []
        for fi in range(T):
            tiles = []
            for ci, (name, _, model_faces, color, _, parents) in enumerate(columns):
                v = view_seqs[ci][fi]
                jt = view_joint_seqs[ci][fi] if view_joint_seqs[ci] is not None else None
                img = render_mesh_png(
                    v, model_faces, None,
                    args.width, args.height, color=color, wireframe=wireframe_mode,
                    joints=jt, parents=parents, body_alpha=0.6,
                    camera_pose=shared_cam_pose, ground=args.ground_lock,
                )
                tiles.append(_annotate(img, name))
            composite_frames.append(np.concatenate(tiles, axis=1))

        # Still PNG: mid-frame.
        from PIL import Image
        mid = T // 2
        still_path = output_dir / "side_by_side.png"
        Image.fromarray(composite_frames[mid]).save(still_path)
        print(f"  Wrote {still_path}")

        # Animated GIF. Play at real-time speed when the motion duration is known
        # (per-frame ms = source_duration_s * 1000 / T). GIF decoders clamp very
        # short durations (<20ms) — at higher source FPS we end up CPU-bound on
        # the decoder anyway, so 20ms (50 FPS) is a sensible floor.
        gif_path = output_dir / "side_by_side.gif"
        gif_frames = [Image.fromarray(f) for f in composite_frames]
        motion_dur = (motion.get("source_duration_s")
                      if (args.motion or args.bvh_motion) and isinstance(motion, dict)
                      else None)
        if motion_dur:
            duration_ms = max(int(round(motion_dur * 1000 / T)), 20)
            src_fps = motion.get("source_fps", float("nan"))
            print(f"  GIF: {T} frames over {motion_dur:.2f}s "
                  f"@ {src_fps:.0f} FPS source → {duration_ms} ms/frame (real-time)")
        else:
            duration_ms = 80
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=duration_ms, loop=0, optimize=True,
        )
        print(f"  Wrote {gif_path}")

    print(f"\nDone! Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
