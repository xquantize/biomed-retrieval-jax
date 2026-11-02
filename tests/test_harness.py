"""Exact search and benchmark parsing (no downloads)."""

import json

import numpy as np
import pytest

from biomed_retrieval_jax.benchmarks import read_benchmark
from biomed_retrieval_jax.search import search


@pytest.mark.parametrize("cosine", [True, False])
@pytest.mark.parametrize(("n_queries", "k", "batch_size"), [(7, 5, 3), (4, 50, 256)])
def test_search_matches_numpy(cosine, n_queries, k, batch_size):
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(n_queries, 16)).astype(np.float32)
    documents = rng.normal(size=(30, 16)).astype(np.float32) * rng.uniform(0.5, 2, (30, 1))
    q, d = queries.astype(np.float64), documents.astype(np.float64)
    if cosine:
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    expected = q @ d.T
    order = np.argsort(-expected, axis=1)[:, : min(k, len(documents))]

    scores, indices = search(queries, documents, k, cosine=cosine, batch_size=batch_size)
    assert indices.shape == order.shape  # k larger than the corpus returns every document
    np.testing.assert_array_equal(indices, order)
    np.testing.assert_allclose(scores, np.take_along_axis(expected, order, 1), rtol=1e-5)


def test_read_benchmark(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq2\td2\t1\n"
    )
    documents = [{"_id": "d1", "title": "T", "text": "one"}, {"_id": "d2", "text": "two"}]
    (tmp_path / "corpus.jsonl").write_text("\n".join(map(json.dumps, documents)))
    queries = [{"_id": q, "text": f"query {q}"} for q in ("q1", "q2", "q3")]
    (tmp_path / "queries.jsonl").write_text("\n".join(map(json.dumps, queries)))

    benchmark = read_benchmark(tmp_path)
    assert benchmark.documents == {"d1": ("T", "one"), "d2": ("", "two")}
    assert benchmark.queries == {"q1": "query q1", "q2": "query q2"}  # q3 is not judged
    assert benchmark.qrels == {"q1": {"d1": 2, "d2": 0}, "q2": {"d2": 1}}
