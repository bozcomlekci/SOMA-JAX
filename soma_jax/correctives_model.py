"""Pose-corrective MLP for SOMA-JAX, faithful to SOMA-X v0.2.1.

Architecture (matches third_party/SOMA-X/soma/correctives_model.py)::

    # Bind-pose relative rotation: R_local = R_bind.T @ R_pose
    x = bindpose.T @ pose_rotmats               # (B, J, 3, 3)
    x[..., 0, 0] -= 1                            # subtract identity diagonal
    x[..., 1, 1] -= 1
    feat = x[..., :, :2].reshape(B, J*6)         # first two columns -> 6D
    W1   = self.W1 * M1_prior                    # (D=J*6, K=J*C)
    W2   = self.W2 * M2_prior                    # (K, 3V)
    z    = relu(feat @ W1)
    if use_tanh: z = tanh(z)
    y    = z @ W2                                # (B, 3V)
    out  = y.reshape(B, V, 3)

Where masks M1=(J,J) and M2=(J,V) are repeat-interleaved into the (D,K) and
(K,3V) shapes that the matmuls expect.

Trained checkpoints distributed with SOMA-X (HuggingFace ``nvidia/SOMA-X``)
ship as PyTorch ``.pt`` files; convert them to ``.npz`` via
``tools/convert/convert_correctives_pt_to_npz.py`` and pass the resulting path to
``SOMALayer.load(correctives_path=...)``.

Upstream: ``soma/correctives_model.py``
    Faithful port: the checkpoint-path resolver and ``CorrectivesMLP`` with
    upstream's constructor keywords, attributes, dict-returning ``forward``,
    ``apply_w2_vertex_mask`` (returns the masked model — equinox modules are
    immutable), ``save_checkpoint`` and ``load_checkpoint``. Torch-only pieces —
    ``NonPersistentModuleWrapper`` and optimizer / scheduler state — have no
    counterpart. ``offsets()`` (the displacements alone) is a SOMA-JAX helper.
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .units import Unit

logger = logging.getLogger(__name__)

#: Upstream's array-or-tensor alias; here NumPy or JAX arrays.
ArrayLike = np.ndarray | jax.Array
_DEFAULT_CORRECTIVES_MODEL_FILENAME = "correctives_model.pt"


class _DefaultCorrectivesModelPath:
    """Sentinel for "``<data_root>/correctives_model.pt`` when enabled" (upstream)."""

    def __repr__(self) -> str:
        return "_DEFAULT_CORRECTIVES_MODEL_PATH"


_DEFAULT_CORRECTIVES_MODEL_PATH = _DefaultCorrectivesModelPath()


def resolve_correctives_model_path(
    *,
    data_root,
    correctives_model_path,
    load_correctives_model: Optional[bool],
    default_enabled: bool = True,
) -> Optional[Path]:
    """Resolve the corrective checkpoint path while preserving the legacy flag.

    Upstream ``soma.correctives_model._resolve_correctives_model_path``, same
    rules and messages: the default resolves to ``<data_root>/correctives_model.pt``
    when correctives are enabled (the body's procedural rig; always for hands),
    ``None`` disables loading, an explicit path must exist and needs the
    procedural rig, and the deprecated ``load_correctives_model`` flag still
    works with a ``DeprecationWarning``.

    SOMA-JAX extra: reading ``.pt`` checkpoints needs ``torch``; without it the
    *default* checkpoint is skipped (a torch-free install gets pure LBS) rather
    than failing, while an explicit path still fails loudly at load.
    """
    path_is_default = correctives_model_path is _DEFAULT_CORRECTIVES_MODEL_PATH
    path_is_disabled = correctives_model_path is None
    path_is_explicit = not path_is_default and not path_is_disabled

    if load_correctives_model is not None:
        warnings.warn(
            "load_correctives_model is deprecated; use correctives_model_path=None "
            "to disable corrective loading, or pass a checkpoint path.",
            DeprecationWarning, stacklevel=3)
        if path_is_explicit:
            raise ValueError(
                "Do not pass load_correctives_model together with an explicit "
                "correctives_model_path.")
        if load_correctives_model:
            if path_is_disabled:
                raise ValueError(
                    "load_correctives_model=True conflicts with correctives_model_path=None.")
            if not default_enabled:
                raise ValueError(
                    "Correctives require procedural transforms; construct with "
                    "enable_procedural_transforms=True or disable correctives.")
            return Path(data_root) / _DEFAULT_CORRECTIVES_MODEL_FILENAME
        return None

    if path_is_disabled:
        return None

    if path_is_default:
        if not default_enabled:
            return None
        path = Path(data_root) / _DEFAULT_CORRECTIVES_MODEL_FILENAME
        if path.suffix == ".pt":
            try:
                import torch  # noqa: F401
            except ImportError:
                return None
        return path

    if not default_enabled:
        raise ValueError(
            "Correctives require procedural transforms; construct with "
            "enable_procedural_transforms=True or disable correctives.")

    path = Path(correctives_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Correctives model checkpoint not found: {path}")
    return path


class CorrectivesMLP(eqx.Module):
    """Pose-corrective displacement predictor — upstream's ``CorrectivesMLP``.

    Takes upstream's keyword constructor (``bindpose=``, ``cors_per_joint=``,
    ``num_verts=``, ``M1_mask=``, ``M2_mask=``, ``W1_init=``, ``W2_init=``,
    ``dropout_p=``, ``use_tanh=``) as well as SOMA-JAX's
    ``(n_joints, n_vertices, ...)`` form. Calling the model is upstream's
    :meth:`forward`, which returns a dict (``out``, ``z``, ``W2``,
    ``W2_masked``); :meth:`offsets` returns just the displacements.

    Args:
        n_joints: number of joints J (78 for SOMA); taken from ``bindpose``
            when omitted.
        n_vertices: number of mesh vertices V (18056 for SOMA); upstream's
            ``num_verts``.
        cors_per_joint: per-joint correctives C (24 in trained checkpoint).
        bindpose: (J, 3, 3) or (J, 4, 4) bind pose for input remapping.
        W1: (J*6, J*C) layer-1 weights; upstream's ``W1_init``.
        W2: (J*C, 3V) layer-2 weights; upstream's ``W2_init``.
        M1_mask: (J, J) sparse joint-to-joint anatomical mask.
        M2_mask: (J, V) sparse joint-to-vertex anatomical mask.
        use_tanh: apply tanh after relu on hidden activations. Defaults to
            True, matching upstream's constructor and its checkpoint
            fallback; a checkpoint trained with tanh evaluated without it
            produces silently wrong offsets.
        key: PRNG key for the Xavier initialisation of an absent ``W1``.
        dropout_p: dropout on the hidden activations in training mode.
    """

    n_joints: int = eqx.field(static=True)
    n_vertices: int = eqx.field(static=True)
    cors_per_joint: int = eqx.field(static=True)
    use_tanh: bool = eqx.field(static=True)
    dropout_p: float = eqx.field(static=True)

    bindpose: jnp.ndarray            # (J, 3, 3)
    W1: jnp.ndarray                  # (D=J*6, K=J*C)
    W2: jnp.ndarray                  # (K, 3*V)
    M1_prior: Optional[jnp.ndarray]  # (D, K) -- expanded from M1_mask (J,J)
    M2_prior: Optional[jnp.ndarray]  # (K, 3*V) -- expanded from M2_mask (J,V)
    M1_mask: Optional[jnp.ndarray]   # (J, J)
    M2_mask: Optional[jnp.ndarray]   # (J, V)

    def __init__(
        self,
        n_joints: Optional[int] = None,
        n_vertices: Optional[int] = None,
        cors_per_joint: int = 24,
        bindpose: Optional[np.ndarray] = None,
        W1: Optional[np.ndarray] = None,
        W2: Optional[np.ndarray] = None,
        M1_mask: Optional[np.ndarray] = None,
        M2_mask: Optional[np.ndarray] = None,
        use_tanh: bool = True,
        key: Optional[jax.Array] = None,
        *,
        num_verts: Optional[int] = None,
        W1_init: Optional[np.ndarray] = None,
        W2_init: Optional[np.ndarray] = None,
        dropout_p: float = 0.0,
    ):
        if bindpose is not None:
            bindpose = np.asarray(bindpose, np.float32)[:, :3, :3]
        if n_joints is None:
            if bindpose is None:
                raise TypeError("CorrectivesMLP needs n_joints or bindpose.")
            n_joints = bindpose.shape[0]
        if n_vertices is None:
            n_vertices = num_verts
        if n_vertices is None:
            raise TypeError("CorrectivesMLP needs n_vertices (upstream's num_verts).")
        W1 = W1_init if W1 is None else W1
        W2 = W2_init if W2 is None else W2
        self.n_joints = int(n_joints)
        self.n_vertices = int(n_vertices)
        self.cors_per_joint = int(cors_per_joint)
        self.use_tanh = bool(use_tanh)
        self.dropout_p = float(dropout_p)

        # Architectural dims:
        #   D = J * 6              # input feature length (6D rotation per joint)
        #   K = J * cors_per_joint # corrective basis count
        # For the v0.2.1 trained checkpoint: J=78, cors_per_joint=24 → K=1872.
        D = self.n_joints * 6
        K = self.n_joints * self.cors_per_joint

        # Bindpose buffer: per-joint world bind rotation. The forward pass
        # left-multiplies by `bindpose.T`, so passing identity reduces the
        # input feature to the raw 6D rotation. The trained checkpoint
        # supplies its own bindpose (the SOMA bind-pose orientation), so the
        # identity default is only useful for an untrained-from-scratch model.
        if bindpose is None:
            bindpose = np.broadcast_to(np.eye(3, dtype=np.float32),
                                       (self.n_joints, 3, 3)).copy()
        self.bindpose = jnp.asarray(bindpose, dtype=jnp.float32)

        # W1 (D, K) is Xavier-uniform initialized when not provided. This
        # gives a sensible scale for an untrained model; the trained
        # checkpoint always provides W1 explicitly.
        if W1 is None:
            if key is None:
                key = jax.random.PRNGKey(0)
            bound = float(np.sqrt(6.0 / (D + K)))
            self.W1 = jax.random.uniform(key, (D, K), minval=-bound, maxval=bound)
        else:
            self.W1 = jnp.asarray(W1, dtype=jnp.float32)

        # W2 (K, 3V) defaults to ZERO so an unloaded model produces zero
        # correctives — i.e. acts as a no-op layer. This matches SOMA-X's
        # behaviour and lets callers safely skip the checkpoint without
        # corrupting their LBS output.
        if W2 is None:
            self.W2 = jnp.zeros((K, 3 * self.n_vertices), dtype=jnp.float32)
        else:
            self.W2 = jnp.asarray(W2, dtype=jnp.float32)

        # M1 mask: (J, J) joint-to-joint anatomical adjacency. Expanded to
        # the (D=J*6, K=J*C) shape by repeat-interleaving the rows by 6
        # (one feature axis per input column) and the cols by `cors_per_joint`
        # (one column per corrective basis vector at that joint).
        if M1_mask is not None:
            m1 = np.asarray(M1_mask, dtype=np.float32)
            assert m1.shape == (self.n_joints, self.n_joints), \
                f"M1_mask must be (J,J)=({self.n_joints},{self.n_joints}), got {m1.shape}"
            prior = np.repeat(np.repeat(m1, 6, axis=0), self.cors_per_joint, axis=1)
            self.M1_prior = jnp.asarray(prior, dtype=jnp.float32)
            self.M1_mask = jnp.asarray(m1)
        else:
            self.M1_prior = None
            self.M1_mask = None

        # M2 mask: (J, V) joint-to-vertex anatomical influence. Expanded to
        # (K=J*C, 3V) by repeat-interleaving rows by `cors_per_joint` (each
        # joint contributes C basis rows) and cols by 3 (xyz per vertex).
        if M2_mask is not None:
            m2 = np.asarray(M2_mask, dtype=np.float32)
            assert m2.shape == (self.n_joints, self.n_vertices), \
                f"M2_mask must be (J,V)=({self.n_joints},{self.n_vertices}), got {m2.shape}"
            prior = np.repeat(np.repeat(m2, self.cors_per_joint, axis=0), 3, axis=1)
            self.M2_prior = jnp.asarray(prior, dtype=jnp.float32)
            self.M2_mask = jnp.asarray(m2)
        else:
            self.M2_prior = None
            self.M2_mask = None

    # Upstream's dimension names.
    @property
    def J(self) -> int:
        """Number of joints."""
        return self.n_joints

    @property
    def C(self) -> int:
        """Correctives per joint."""
        return self.cors_per_joint

    @property
    def K(self) -> int:
        """Total correctives (``J * C``)."""
        return self.n_joints * self.cors_per_joint

    @property
    def I(self) -> int:  # noqa: E743 - upstream's name
        """Input features per joint (6D rotation)."""
        return 6

    @property
    def D(self) -> int:
        """Total input features (``I * J``)."""
        return 6 * self.n_joints

    @property
    def V(self) -> int:
        """Vertices of the target mesh."""
        return self.n_vertices

    def offsets(self, rotmats: jnp.ndarray) -> jnp.ndarray:
        """Predict pose correctives: the ``out`` of :meth:`forward` (SOMA-JAX helper).

        Args:
            rotmats: (..., J, 3, 3) absolute joint rotations (joint-orient applied).

        Returns:
            (..., V, 3) vertex displacements in meters.
        """
        batch_shape = rotmats.shape[:-3]
        J = self.n_joints

        # Step 1 — bindpose-relative input feature.
        # Compute R_local = bindpose.T @ R_pose; subtract the identity on the
        # first two diagonal entries so the feature is zero at the bind pose.
        # The einsum "jba,...jbc->...jac" is a batched per-joint matmul
        # equivalent to `bindpose[j].T @ R_pose[j]` for each joint j.
        x = jnp.einsum("jba,...jbc->...jac", self.bindpose, rotmats[..., :3, :3])
        x = x.at[..., 0, 0].add(-1.0)
        x = x.at[..., 1, 1].add(-1.0)
        # 6D rotation feature: first two columns flattened.
        feat = x[..., :, :2].reshape(batch_shape + (J * 6,))

        # Step 2 — apply anatomical masks. Each mask is a binary multiplier
        # over the corresponding weight matrix; locations where the mask is 0
        # contribute nothing to the layer's output regardless of W1 / W2
        # values. The masks enforce spatial locality (a finger joint cannot
        # influence vertex positions on the opposite leg).
        W1 = self.W1 * self.M1_prior if self.M1_prior is not None else self.W1
        W2 = self.W2 * self.M2_prior if self.M2_prior is not None else self.W2

        # Step 3 — two-layer MLP: feat (B, D) @ W1 (D, K) -> z (B, K),
        # ReLU + optional tanh, then z @ W2 (K, 3V) -> y (B, 3V).
        z = feat @ W1
        z = jax.nn.relu(z)
        if self.use_tanh:
            z = jnp.tanh(z)
        y = z @ W2

        # Output: per-vertex displacement in the model's output unit (meters
        # for the trained checkpoint shipped with v0.2.1).
        return y.reshape(batch_shape + (self.n_vertices, 3))

    def forward(self, x: jnp.ndarray, V=None, *, training: bool = False,
                key: Optional[jax.Array] = None) -> dict:
        """Upstream ``forward``: (B, J, 3, 3) rotations -> a dict.

        Returns ``out`` (B, V, 3) displacements, the hidden activations ``z``,
        the raw ``W2`` and the masked ``W2_masked`` (its geometric
        contribution). ``training=True`` applies dropout (``dropout_p``) and
        then needs a PRNG ``key``; ``V`` is unused, as upstream.
        """
        B = x.shape[0]
        x = jnp.einsum("jba,...jbc->...jac", self.bindpose, x[..., :3, :3])
        x = x.at[..., 0, 0].add(-1.0)
        x = x.at[..., 1, 1].add(-1.0)
        features = x[..., :, :2].reshape(B, -1)
        W1 = self.W1 * self.M1_prior if self.M1_prior is not None else self.W1
        W2 = self.W2 * self.M2_prior if self.M2_prior is not None else self.W2
        z = jax.nn.relu(features @ W1)
        if self.use_tanh:
            z = jnp.tanh(z)
        if training and self.dropout_p > 0.0:
            if key is None:
                raise ValueError("training-mode dropout needs a PRNG key")
            keep = jax.random.bernoulli(key, 1.0 - self.dropout_p, z.shape)
            z = jnp.where(keep, z / (1.0 - self.dropout_p), 0.0)
        y = z @ W2
        return {"out": y.reshape(B, -1, 3), "z": z, "W2": self.W2, "W2_masked": W2}

    __call__ = forward

    def apply_w2_vertex_mask(self, vertex_mask) -> "CorrectivesMLP":
        """Zero ``W2`` outside a vertex mask (upstream ``apply_w2_vertex_mask``).

        Upstream multiplies its parameter in place; this module is immutable,
        so the masked model is returned.
        """
        vertex_mask = jnp.asarray(vertex_mask)
        if vertex_mask.shape != (self.n_vertices,):
            raise ValueError(
                f"vertex_mask must have shape ({self.n_vertices},), got {vertex_mask.shape}")
        mask = jnp.repeat(vertex_mask.astype(self.W2.dtype), 3)[None]
        return eqx.tree_at(lambda m: m.W2, self, self.W2 * mask)

    def save_checkpoint(self, path: str, *, native_unit: Unit = Unit.CENTIMETERS,
                        optimizer=None, scheduler=None, meta=None,
                        save_masks: bool = True) -> None:
        """Save the weights (upstream ``save_checkpoint``).

        A ``.pt`` path gets upstream's own payload — ``C_max``, ``use_tanh``,
        ``bindpose``, sparse ``W1`` / ``W2``, ``meta`` and, with
        ``save_masks``, the sparse masks — written with ``torch``; any other
        path gets the same arrays as an ``.npz`` (no ``torch``). Without
        ``save_masks`` the masks are folded into ``W1`` / ``W2`` instead. As
        upstream, ``W2`` is stored as held and no unit is recorded (loading
        assumes centimetres; ``native_unit`` is accepted and unused, as
        upstream's is). ``optimizer`` / ``scheduler`` states are torch objects
        with no counterpart here and must be ``None``.
        """
        if optimizer is not None or scheduler is not None:
            raise NotImplementedError(
                "optimizer / scheduler states are torch objects; SOMA-JAX does not save them.")
        masked = not save_masks
        W1 = self.W1 * self.M1_prior if (masked and self.M1_prior is not None) else self.W1
        W2 = self.W2 * self.M2_prior if (masked and self.M2_prior is not None) else self.W2
        arrays = {
            "C_max": np.int32(self.cors_per_joint),
            "use_tanh": np.bool_(self.use_tanh),
            "bindpose": np.array(self.bindpose),
            "W1": np.array(W1),
            "W2": np.array(W2),
        }
        if save_masks:
            if self.M1_mask is not None:
                arrays["M1_mask"] = np.array(self.M1_mask)
            if self.M2_mask is not None:
                arrays["M2_mask"] = np.array(self.M2_mask)
        if str(path).endswith(".pt"):
            import torch
            payload = {
                "C_max": int(self.cors_per_joint),
                "use_tanh": bool(self.use_tanh),
                "bindpose": torch.from_numpy(arrays["bindpose"]),
                "W1": torch.from_numpy(arrays["W1"]).to_sparse(),
                "W2": torch.from_numpy(arrays["W2"]).to_sparse(),
                "meta": meta or {},
            }
            for key in ("M1_mask", "M2_mask"):
                if key in arrays:
                    payload[key] = torch.from_numpy(arrays[key]).to_sparse()
            torch.save(payload, path)
            return
        np.savez(path, **arrays)

    @classmethod
    def load_checkpoint(cls, path: str, v_index_map=None, joint_indices=None, *,
                        optimizer=None, scheduler=None, map_location="cpu",
                        output_unit: Unit = Unit.METERS) -> "CorrectivesMLP":
        """Load a corrective checkpoint (upstream ``load_checkpoint``).

        Accepts upstream's ``correctives_model.pt`` directly (read with
        :func:`load_correctives_pt`, which needs ``torch``), or an ``.npz``
        written by ``tools/convert/convert_correctives_pt_to_npz.py`` /
        :py:meth:`save_checkpoint` (no ``torch``).

        Args:
            path: checkpoint path.
            v_index_map: optional vertex subset to slice the output layer onto,
                e.g. ``lod_mid_to_low`` for a low-LOD layer. Mirrors upstream's
                ``correctives_vertex_index_map``.
            joint_indices: optional joint subset — upstream's
                ``load_checkpoint(joint_indices=...)``, which ``SOMAHandLayer``
                uses to drive the shared body checkpoint from the 25 hand joints.
                Keeps each selected joint's 6 input DOFs, its ``C`` correctives,
                its bind pose and its mask rows/columns.
            optimizer, scheduler: upstream restores torch optimizer/scheduler
                states from training checkpoints; there is nothing to restore
                them into here, so a stored state is skipped with a warning.
            map_location: accepted for signature compatibility.
            output_unit: unit of the predicted offsets; ``W2`` is scaled from
                the checkpoint's unit (centimetres unless it records one).

        Returns:
            The model, or ``None`` when ``path`` does not exist (as upstream).
        """
        if not os.path.exists(path):
            return None                             # upstream: a missing checkpoint is None
        if str(path).endswith(".pt"):
            data = load_correctives_pt(path)        # upstream's own checkpoint
        else:
            npz = np.load(path, allow_pickle=False)
            data = {k: npz[k] for k in npz.files}
        for name, state in (("Optimizer", optimizer), ("Scheduler", scheduler)):
            if state is not None:
                logger.warning("%s state not loaded: SOMA-JAX has no torch %s to restore.",
                               name, name.lower())
        bindpose = np.asarray(data["bindpose"], dtype=np.float32)
        W1 = np.asarray(data["W1"], dtype=np.float32)
        # Both readers yield metres; upstream scales to `output_unit` here.
        W2 = np.asarray(data["W2"], dtype=np.float32) / output_unit.meters_per_unit
        M1 = np.asarray(data["M1_mask"], dtype=np.float32) if "M1_mask" in data else None
        M2 = np.asarray(data["M2_mask"], dtype=np.float32) if "M2_mask" in data else None
        C  = int(np.asarray(data["C_max"]).item())
        ut = bool(np.asarray(data["use_tanh"]).item()) if "use_tanh" in data else True
        J = bindpose.shape[0]

        if v_index_map is not None:
            # Slice the output layer onto a vertex subset (upstream's
            # `_slice_checkpoint_tensors` with `v_index_map`). W2 stores xyz
            # interleaved per vertex, so columns are gathered as 3*v + {0,1,2};
            # M2's mask is per-vertex. Without this a low-LOD layer gets
            # 18056 offsets for a 4505-vertex rest shape.
            v_idx = np.asarray(v_index_map, dtype=np.int64).ravel()
            if v_idx.size and (v_idx.min() < 0 or v_idx.max() >= W2.shape[1] // 3):
                raise ValueError(
                    f"v_index_map out of range for a {W2.shape[1] // 3}-vertex checkpoint.")
            col = (v_idx[:, None] * 3 + np.arange(3)).reshape(-1)
            W2 = W2[:, col]
            if M2 is not None:
                M2 = M2[:, v_idx]

        if joint_indices is not None:
            # Upstream `_slice_checkpoint_tensors(joint_indices=...)`, verbatim:
            # W1 is (6J, CJ) — rows are per-joint 6D rotation inputs, columns
            # per-joint corrective units; W2's rows are those same units.
            j_idx = np.asarray(joint_indices, dtype=np.int64).ravel()
            if j_idx.size and (j_idx.min() < 0 or j_idx.max() >= J):
                raise ValueError(f"joint_indices out of range for a {J}-joint checkpoint.")
            dof_idx = (j_idx[:, None] * 6 + np.arange(6)).reshape(-1)
            corrective_idx = (j_idx[:, None] * C + np.arange(C)).reshape(-1)
            bindpose = bindpose[j_idx]
            W1 = W1[dof_idx][:, corrective_idx]
            W2 = W2[corrective_idx]
            if M1 is not None:
                M1 = M1[j_idx][:, j_idx]
            if M2 is not None:
                M2 = M2[j_idx]
            J = int(j_idx.size)

        V = W2.shape[1] // 3
        return cls(
            n_joints=J, n_vertices=V, cors_per_joint=C,
            bindpose=bindpose, W1=W1, W2=W2,
            M1_mask=M1, M2_mask=M2, use_tanh=ut,
        )


def load_correctives_pt(path) -> dict:
    """Read upstream's ``.pt`` corrective checkpoint into plain arrays.

    Returns the arrays :meth:`CorrectivesMLP.load_checkpoint` consumes, with
    ``W2`` already scaled to **metres** — upstream's ``load_checkpoint`` scales
    it by ``native_unit.meters_per_unit / output_unit.meters_per_unit`` at load
    time; SOMA-JAX works in metres, so the scale is applied here once.

    Needs ``torch`` (imported lazily; only for this call). Mirrors upstream's
    SOMA-X v0.3.3 loading guard: before PyTorch 2.8 sparse validation pins
    memory and re-initializes CUDA inside forked DataLoader workers
    (pytorch/pytorch#153143), so the validator is bypassed only there.

    Args:
        path: the ``.pt`` checkpoint.

    Returns:
        ``C_max``, ``use_tanh``, ``bindpose``, ``W1``, ``W2`` and, when present,
        ``M1_mask``, ``M2_mask``, ``joint_indices``, ``source_num_joints``.
    """
    import torch

    # weights_only=True, as upstream: the checkpoint is tensors plus a meta
    # dict of plain values, and nothing else is ever unpickled.
    if torch.__version__ < "2.8" and torch.cuda._is_in_bad_fork():
        orig = torch._utils._validate_loaded_sparse_tensors
        torch._utils._validate_loaded_sparse_tensors = lambda: None
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        finally:
            torch._utils._validate_loaded_sparse_tensors = orig
            torch._utils._sparse_tensors_to_validate.clear()
    else:
        ck = torch.load(path, map_location="cpu", weights_only=True)

    def dense(t):
        if torch.is_tensor(t) and t.is_sparse:
            t = t.to_dense()
        return np.asarray(t, dtype=np.float32) if torch.is_tensor(t) else np.asarray(t)

    native = str(ck.get("unit", "centimeters")).lower()
    scale = {"centimeters": 0.01, "millimeters": 0.001, "meters": 1.0}.get(native, 0.01)
    out = {
        "C_max": np.int32(int(ck["C_max"])),
        "use_tanh": np.bool_(bool(ck.get("use_tanh", True))),
        "bindpose": dense(ck["bindpose"]),
        "W1": dense(ck["W1"]),
        "W2": dense(ck["W2"]) * scale,
    }
    for key in ("M1_mask", "M2_mask"):
        if key in ck:
            out[key] = dense(ck[key])
    if "joint_indices" in ck:
        out["joint_indices"] = np.asarray(ck["joint_indices"], dtype=np.int64)
    if "source_num_joints" in ck:
        out["source_num_joints"] = np.int32(int(ck["source_num_joints"]))
    return out

