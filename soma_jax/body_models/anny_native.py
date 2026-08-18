"""Anny children's-body identity model, JAX evaluation.

Upstream: ``soma/identity_model.py`` — ``AnnySimplified`` + ``AnnyIdentityModel``.

Upstream does **not** vendor Anny; it imports the package and calls into it::

    import anny
    anny_model = anny.create_fullbody_model(
        all_phenotypes=True, local_changes=True, remove_unattached_vertices=True)
    ...
    blendshape_coeffs = anny_model.get_phenotype_blendshape_coefficients(
        **phenotype_kwargs, local_changes=local_changes_kwargs)
    rest_vertices = anny_model.get_rest_vertices(blendshape_coeffs)

So calling ``anny`` *is* the faithful behaviour — reimplementing its internals
would diverge from upstream, not converge on it. What this module does is split
that call at the seam where JAX actually helps:

===================================  =========================  ===============
step                                 who evaluates it           why
===================================  =========================  ===============
phenotype values -> 1132 coeffs      the ``anny`` package        **not affine** —
                                                                 a nonlinear
                                                                 multi-way blend
                                                                 over phenotype
                                                                 corner shapes.
                                                                 Runs once per
                                                                 identity.
1132 coeffs -> 13718 rest vertices   **JAX**, in this module     exactly
                                                                 ``template + B·c``
                                                                 (verified to 0.0);
                                                                 186 MB of
                                                                 blendshapes, and
                                                                 this is the part
                                                                 that runs per call
===================================  =========================  ===============

The second step is lifted verbatim, so :meth:`AnnyNativeModel.get_rest_vertices`
is bit-identical to ``anny``'s and needs no torch. :meth:`to_npz` caches the
lifted arrays so an inference-only deployment does not need ``anny`` either —
only the coefficients, which callers can precompute.

Native frame is **metres**, **Z-up**, **-Y-forward**, matching
``AnnyIdentityModel.NATIVE_UNIT`` / ``NATIVE_UP`` / ``NATIVE_FORWARD``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import jax.numpy as jnp
import numpy as np

__all__ = ["AnnyNativeModel", "SOMA_PHENOTYPE_LABELS", "ANNY_IGNORED_LOCAL_CHANGES"]

#: The six phenotypes ``AnnySimplified`` exposes, in its declared order. The
#: installed package offers more (cupsize, firmness, african, asian, caucasian);
#: upstream deliberately narrows to these, so the port does too.
SOMA_PHENOTYPE_LABELS = ("gender", "age", "muscle", "weight", "height", "proportions")

#: Substrings ``AnnySimplified`` filters out of ``local_change_labels`` — facial
#: detail Anny's own mesh already carries, which SOMA does not drive.
ANNY_IGNORED_LOCAL_CHANGES = (
    "mouth", "eye", "nipple", "cheek", "chin", "ear", "lip", "nose",
)


class AnnyNativeModel:
    """JAX evaluation of Anny's rest shape.

    Build with :meth:`from_anny` (imports the package once, as upstream does) or
    :meth:`from_npz` (no ``anny``, no torch).
    """

    FIELDS = ("template_vertices", "blendshapes")

    def __init__(self, template_vertices, blendshapes, *,
                 phenotype_labels: Sequence[str] = SOMA_PHENOTYPE_LABELS,
                 local_change_labels: Sequence[str] = (),
                 ignored_change_labels: Sequence[str] = (),
                 _anny: Any = None):
        self.template_vertices = jnp.asarray(template_vertices, jnp.float32)
        self.blendshapes = jnp.asarray(blendshapes, jnp.float32)
        self.phenotype_labels = tuple(phenotype_labels)
        self.local_change_labels = tuple(local_change_labels)
        self.ignored_change_labels = tuple(ignored_change_labels)
        self._anny = _anny

        self.num_vertices = int(self.template_vertices.shape[0])
        self.num_blendshapes = int(self.blendshapes.shape[0])

    @property
    def num_identity_coeffs(self) -> int:
        """Upstream's ``AnnyIdentityModel.num_identity_coeffs``.

        Upstream returns ``len(self.identity_model.phenotype_labels)`` — the
        *simplified* wrapper's six, not the package's full set.
        """
        return len(self.phenotype_labels)

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_anny(cls, anny_model: Any = None, **create_kwargs) -> "AnnyNativeModel":
        """Lift the blendshape basis out of an ``anny`` model.

        Args:
            anny_model: an existing model; created with upstream's arguments
                when omitted.
            **create_kwargs: overrides for ``anny.create_fullbody_model``.

        Notes:
            ``anny`` 0.6 deprecates ``local_changes=True`` in favour of
            ``'default'``; upstream passes the bool. The port passes the string
            so the same set is selected without the warning.
        """
        if anny_model is None:
            import anny
            kwargs = dict(all_phenotypes=True, local_changes="default",
                          remove_unattached_vertices=True)
            kwargs.update(create_kwargs)
            anny_model = anny.create_fullbody_model(**kwargs)

        keep, drop = [], []
        for label in getattr(anny_model, "local_change_labels", ()):
            bucket = drop if any(x in label for x in ANNY_IGNORED_LOCAL_CHANGES) else keep
            if label not in bucket:
                bucket.append(label)

        return cls(
            np.asarray(anny_model.template_vertices.detach().cpu().numpy(), np.float32),
            np.asarray(anny_model.blendshapes.detach().cpu().numpy(), np.float32),
            local_change_labels=keep, ignored_change_labels=drop,
            _anny=anny_model,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "AnnyNativeModel":
        d = np.load(path, allow_pickle=False)
        labels = ([str(x) for x in d["phenotype_labels"]]
                  if "phenotype_labels" in d else SOMA_PHENOTYPE_LABELS)
        return cls(d["template_vertices"], d["blendshapes"], phenotype_labels=labels)

    def to_npz(self, path: str | Path) -> None:
        """Cache the lifted basis so inference needs neither ``anny`` nor torch."""
        np.savez_compressed(
            path,
            template_vertices=np.asarray(self.template_vertices),
            blendshapes=np.asarray(self.blendshapes),
            phenotype_labels=np.asarray(self.phenotype_labels),
        )

    # ---- forward ---------------------------------------------------------
    def get_rest_vertices(self, blendshape_coeffs: jnp.ndarray) -> jnp.ndarray:
        """``template_vertices + einsum("nvd,...n->...vd", blendshapes, coeffs)``.

        Lifted verbatim from ``anny``'s ``get_rest_vertices`` — agreement is
        exact, not approximate (``tests/test_anny_native.py``).

        Args:
            blendshape_coeffs: (..., 1132).
        Returns:
            (..., 13718, 3) rest vertices in metres, Z-up.
        """
        c = jnp.asarray(blendshape_coeffs, jnp.float32)
        return self.template_vertices + jnp.einsum("nvd,...n->...vd", self.blendshapes, c)

    def phenotype_coefficients(
        self,
        phenotypes: Optional[jnp.ndarray] = None,
        local_changes: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Phenotype values -> blendshape coefficients, via the ``anny`` package.

        This is the step upstream delegates and the port cannot lift: it is not
        affine in the phenotype vector (superposition breaks by ~0.2 on random
        pairs), so there is no matrix to extract. It runs once per identity, not
        per frame.

        Args:
            phenotypes: (B, num_identity_coeffs) values in Anny's own range,
                one column per entry of :py:attr:`phenotype_labels`. ``None``
                uses Anny's defaults.
            local_changes: forwarded to
                ``get_phenotype_blendshape_coefficients``.

        Returns:
            (B, 1132) coefficients, ready for :meth:`get_rest_vertices`.

        Raises:
            RuntimeError: when this model was built without the package (e.g.
                via :meth:`from_npz`).
        """
        if self._anny is None:
            raise RuntimeError(
                "phenotype_coefficients needs the `anny` package (this model was "
                "loaded from a cached .npz). Build with AnnyNativeModel.from_anny(), "
                "or pass precomputed coefficients to get_rest_vertices()."
            )
        import torch
        kwargs = self._anny.parse_phenotype_kwargs(None)
        available = set(getattr(self._anny, "phenotype_labels", ()))
        unknown = set(self.phenotype_labels) - available
        if unknown:
            raise ValueError(
                f"Invalid phenotype: {sorted(unknown)}; available: {sorted(available)}")

        if phenotypes is not None:
            values = np.atleast_2d(np.asarray(phenotypes, np.float64))
            if values.shape[1] != len(self.phenotype_labels):
                raise ValueError(
                    f"expected {len(self.phenotype_labels)} phenotype columns "
                    f"({', '.join(self.phenotype_labels)}), got {values.shape[1]}")
            for i, label in enumerate(self.phenotype_labels):
                kwargs[label] = torch.tensor(values[:, i], dtype=torch.float64)
            # Broadcast the untouched phenotypes to the same batch width.
            B = values.shape[0]
            for label, v in kwargs.items():
                if v.shape[0] != B:
                    kwargs[label] = v.expand(B).clone()

        coeffs = self._anny.get_phenotype_blendshape_coefficients(
            **kwargs, local_changes=local_changes)
        return jnp.asarray(coeffs.detach().cpu().numpy(), jnp.float32)

    def get_rest_shape(
        self,
        identity_coeffs: Optional[jnp.ndarray] = None,
        local_changes: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Phenotype values -> rest vertices, i.e. ``AnnySimplified.forward``."""
        return self.get_rest_vertices(
            self.phenotype_coefficients(identity_coeffs, local_changes))
