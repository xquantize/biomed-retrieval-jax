"""Exact search: score every document against every query with a matrix multiply."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from eqx_zoo.layers import l2_normalize
from jaxtyping import Array, Float, Int


@partial(jax.jit, static_argnames=("k", "cosine"))
def _top_k(
    queries: Float[Array, "q dim"], documents: Float[Array, "n dim"], k: int, cosine: bool
) -> tuple[Float[Array, "q k"], Int[Array, "q k"]]:
    if cosine:
        queries, documents = l2_normalize(queries), l2_normalize(documents)
    return jax.lax.top_k(jnp.matmul(queries, documents.T, precision="highest"), k)


def search(
    queries: Float[np.ndarray, "q dim"],
    documents: Float[np.ndarray, "n dim"],
    k: int,
    *,
    cosine: bool,
    batch_size: int = 256,
) -> tuple[Float[np.ndarray, "q k"], Int[np.ndarray, "q k"]]:
    """Find each query's highest-scoring documents.

    Args:
        queries: Query embeddings.
        documents: Document embeddings.
        k: Documents to return per query; all of them if there are fewer.
        cosine: Score by cosine similarity if true, otherwise by dot product.
        batch_size: Queries scored at once.

    Returns:
        Each query's scores and document indices, highest first.
    """
    k = min(k, len(documents))
    documents = jnp.asarray(documents)
    scores, indices = [], []
    for start in range(0, len(queries), batch_size):
        batch = jnp.asarray(queries[start : start + batch_size])
        s, i = _top_k(batch, documents, k, cosine)
        scores.append(np.asarray(s))
        indices.append(np.asarray(i))
    return np.concatenate(scores), np.concatenate(indices)
