"""Evaluate a retrieval model on a benchmark, as MTEB does, and print nDCG@10 and Recall@100.

Scores are rounded to 5 decimals, as MTEB reports them.

Run with `uv run python scripts/evaluate.py MODEL BENCHMARK`, for example
`uv run python scripts/evaluate.py all-minilm-l6-v2 nfcorpus`.
"""

import argparse
import time

from biomed_retrieval_jax.benchmarks import BENCHMARKS, load_benchmark
from biomed_retrieval_jax.metrics import evaluate
from biomed_retrieval_jax.retrieval import MODELS, retrieve


def main():
    """Parse the arguments, run the evaluation and print the scores."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    start = time.perf_counter()
    run = retrieve(MODELS[args.model], benchmark, batch_size=args.batch_size)
    seconds = time.perf_counter() - start
    metrics = evaluate(run, benchmark.qrels)
    print(
        f"{args.model} on {args.benchmark}: {len(benchmark.documents)} documents, "
        f"{len(benchmark.queries)} queries, {seconds:.0f} s"
    )
    for name in ("ndcg@10", "recall@100"):
        print(f"  {name}: {round(metrics[name], 5)}")


if __name__ == "__main__":
    main()
