"""The evaluation harness reproduces MTEB's published scores for all-MiniLM-L6-v2."""

import pytest

from biomed_retrieval_jax.benchmarks import load_benchmark
from biomed_retrieval_jax.metrics import evaluate
from biomed_retrieval_jax.retrieval import MODELS, retrieve

# From github.com/embeddings-benchmark/results, results/sentence-transformers__all-MiniLM-L6-v2/
# 8b3219a92973c328a8e22fadcfa821b5dc75636a/{NFCorpus,SciFact}.json: MTEB 1.12.75, test split,
# at the dataset revisions in `BENCHMARKS`. MTEB rounds to 5 decimals.
PUBLISHED = {
    "nfcorpus": {"ndcg@10": 0.31594, "recall@100": 0.31151},
    "scifact": {"ndcg@10": 0.64508, "recall@100": 0.925},
}


@pytest.mark.checkpoint("all-minilm-l6-v2")
@pytest.mark.parametrize("benchmark", PUBLISHED)
def test_reproduces_published_scores(benchmark):
    data = load_benchmark(benchmark)
    metrics = evaluate(retrieve(MODELS["all-minilm-l6-v2"], data), data.qrels)
    for name, published in PUBLISHED[benchmark].items():
        print(f"{benchmark} {name}: {metrics[name]:.8f} (published {published})")
    assert {name: round(metrics[name], 5) for name in PUBLISHED[benchmark]} == PUBLISHED[benchmark]
