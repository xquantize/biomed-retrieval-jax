"""Retrieval models: which encoders embed queries and documents, and how documents are scored."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from biomed_retrieval_jax.benchmarks import Benchmark
from biomed_retrieval_jax.encoders import ENCODERS, TextEncoder
from biomed_retrieval_jax.search import search


@dataclass(frozen=True)
class RetrievalModel:
    """Encoders for queries and documents, and how documents are presented and scored.

    Attributes:
        query_encoder: Key of `ENCODERS` for queries.
        document_encoder: Key of `ENCODERS` for documents.
        pairs: Whether documents are encoded as (title, text) pairs; otherwise as their title
            and text joined by a space, as MTEB does.
        cosine: Whether documents are scored by cosine similarity; otherwise by dot product.
    """

    query_encoder: str
    document_encoder: str
    pairs: bool = False
    cosine: bool = True


MODELS = {
    "all-minilm-l6-v2": RetrievalModel("all-minilm-l6-v2", "all-minilm-l6-v2"),
    "medcpt": RetrievalModel("medcpt-query", "medcpt-article", pairs=True, cosine=False),
    "pubmedbert-embeddings": RetrievalModel("pubmedbert-embeddings", "pubmedbert-embeddings"),
}


def document_input(document: tuple[str, str], pairs: bool) -> str | tuple[str, str]:
    """How a (title, text) document is given to an encoder.

    Args:
        document: Its title and text.
        pairs: Whether the encoder takes (title, text) pairs.

    Returns:
        The pair itself, or the title and text joined by a space, as MTEB does.
    """
    return document if pairs else " ".join(document).strip()


def retrieve(
    model: RetrievalModel,
    benchmark: Benchmark,
    *,
    k: int = 1000,
    batch_size: int = 32,
    dtype: DTypeLike = jnp.float32,
) -> dict[str, dict[str, float]]:
    """Retrieve each query's highest-scoring documents.

    Args:
        model: The retrieval model.
        benchmark: The documents to search, and the queries.
        k: Documents to retrieve per query; MTEB keeps 1000.
        batch_size: Inputs per batch when embedding.
        dtype: Dtype of the encoders' parameters and embeddings. Embeddings are scored in
            float32 whatever the dtype, so that only the encoders' precision varies.

    Returns:
        Each query's retrieved documents and their scores.
    """
    query_encoder = TextEncoder.from_spec(ENCODERS[model.query_encoder], dtype=dtype)
    if model.document_encoder == model.query_encoder:
        document_encoder = query_encoder
    else:
        document_encoder = TextEncoder.from_spec(ENCODERS[model.document_encoder], dtype=dtype)

    document_ids = list(benchmark.documents)
    documents = [document_input(benchmark.documents[i], model.pairs) for i in document_ids]
    query_ids = list(benchmark.queries)
    queries = query_encoder.embed([benchmark.queries[i] for i in query_ids], batch_size=batch_size)
    embeddings = document_encoder.embed(documents, batch_size=batch_size)
    scores, indices = search(
        queries.astype(np.float32), embeddings.astype(np.float32), k, cosine=model.cosine
    )
    return {
        query: {document_ids[j]: float(s) for s, j in zip(scores[row], indices[row], strict=True)}
        for row, query in enumerate(query_ids)
    }
