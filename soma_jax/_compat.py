"""Compatibility shims for the oldest supported JAX (``jax>=0.4.26``)."""
from __future__ import annotations

import jax


def _registered_private_barrier():
    """``jax.lax.optimization_barrier`` before JAX 0.4.33 made it public.

    The private primitive of older releases has no batching or JVP rule, so
    ``vmap`` and ``grad`` through it fail; register the rules 0.4.33 ships —
    batch dimensions pass through, the barrier applies to the primals and the
    tangents pass through unchanged (it is the identity).
    """
    from jax._src import ad_checkpoint
    from jax.interpreters import ad, batching

    p = ad_checkpoint.optimization_barrier_p
    if p not in batching.primitive_batchers:
        batching.primitive_batchers[p] = (
            lambda args, dims, **params: (p.bind(*args, **params), dims))
    if p not in ad.primitive_jvps:
        ad.primitive_jvps[p] = (
            lambda primals, tangents, **params: (p.bind(*primals, **params), list(tangents)))
    return ad_checkpoint._optimization_barrier


#: ``jax.lax.optimization_barrier`` on every supported JAX version.
optimization_barrier = getattr(jax.lax, "optimization_barrier", None) or _registered_private_barrier()
