"""Compare two runs on the same benchmark: their rankings, metrics, and a significance test.

Each run is `MODEL:DTYPE`, saved by `scripts/evaluate.py` in `runs/`. Reports how many queries'
rankings differ, the difference in mean nDCG@10 and Recall@100, and a paired permutation test of
each difference: the fraction of 10,000 random sign flips of the per-query differences whose mean
is at least as far from zero as the observed one.

Run with `uv run python scripts/compare_runs.py BENCHMARK A B`, for example
`uv run python scripts/compare_runs.py nfcorpus medcpt:float32 medcpt:bfloat16`.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from biomed_retrieval_jax.benchmarks import BENCHMARKS, load_benchmark
from biomed_retrieval_jax.metrics import per_query, rank

ROOT = Path(__file__).parents[1]
PERMUTATIONS = 10_000


def _load(benchmark, run):
    model, dtype = run.split(":")
    with gzip.open(ROOT / "runs" / model / f"{benchmark}-{dtype}.json.gz", "rt") as f:
        return json.load(f)


def _permutation_p_value(differences, rng):
    """Two-sided paired permutation test of the mean of per-query differences."""
    observed = abs(differences.mean())
    signs = rng.choice([-1.0, 1.0], size=(PERMUTATIONS, len(differences)))
    extreme = (np.abs((signs * differences).mean(axis=1)) >= observed - 1e-15).sum()
    return (extreme + 1) / (PERMUTATIONS + 1)


def main():
    """Parse the arguments and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("a", help="first run, as MODEL:DTYPE")
    parser.add_argument("b", help="second run, as MODEL:DTYPE")
    args = parser.parse_args()

    qrels = load_benchmark(args.benchmark).qrels
    runs = {name: _load(args.benchmark, name) for name in (args.a, args.b)}
    metrics = {name: per_query(run, qrels) for name, run in runs.items()}
    queries = sorted(metrics[args.a].keys() & metrics[args.b].keys())
    rankings = {name: {q: rank(run[q]) for q in queries} for name, run in runs.items()}

    def top(name, q, k):
        return rankings[name][q][:k]

    same_order = np.mean([top(args.a, q, 10) == top(args.b, q, 10) for q in queries])
    same_set = np.mean([set(top(args.a, q, 10)) == set(top(args.b, q, 10)) for q in queries])
    overlap = {
        k: np.mean([len(set(top(args.a, q, k)) & set(top(args.b, q, k))) / k for q in queries])
        for k in (10, 100)
    }
    print(f"{args.benchmark}: {args.b} against {args.a}, {len(queries)} queries")
    print(
        f"  top 10 identical in {same_order:.1%} of queries, same set in {same_set:.1%};"
        f" mean overlap: top 10 {overlap[10]:.3f}, top 100 {overlap[100]:.3f}"
    )
    rng = np.random.default_rng(0)
    for name in ("ndcg@10", "recall@100"):
        a = np.array([metrics[args.a][q][name] for q in queries])
        b = np.array([metrics[args.b][q][name] for q in queries])
        d = b - a
        print(
            f"  {name}: {a.mean():.5f} -> {b.mean():.5f} ({d.mean():+.5f}); changed for"
            f" {(d != 0).sum()} queries (largest {np.abs(d).max():.3f});"
            f" permutation p = {_permutation_p_value(d, rng):.4f}"
        )


if __name__ == "__main__":
    main()
