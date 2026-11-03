"""Evaluate a retrieval model on a benchmark, as MTEB does, and save the results.

Prints nDCG@10 and Recall@100, rounded to 5 decimals as MTEB reports them, and writes every
metric, unrounded, with the run's provenance to `results/<model>/<benchmark>-<dtype>.json`. The
retrieved documents are saved to `runs/<model>/<benchmark>-<dtype>.json.gz` (not committed), for
comparing runs.

Run with `uv run python scripts/evaluate.py MODEL BENCHMARK [--dtype bfloat16]`, for example
`uv run python scripts/evaluate.py medcpt nfcorpus`.
"""

import argparse
import gzip
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import jax.numpy as jnp
from huggingface_hub import HfApi

from biomed_retrieval_jax.benchmarks import BENCHMARKS, load_benchmark
from biomed_retrieval_jax.encoders import ENCODERS
from biomed_retrieval_jax.metrics import evaluate
from biomed_retrieval_jax.retrieval import MODELS, retrieve

ROOT = Path(__file__).parents[1]
DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}


def _provenance(model_name, benchmark_name):
    """The exact checkpoints, data, code and platform behind a result."""
    model = MODELS[model_name]
    api = HfApi()
    encoders = {}
    for name in dict.fromkeys((model.query_encoder, model.document_encoder)):
        spec = ENCODERS[name]
        commit = api.model_info(spec.repo, revision=spec.revision).sha
        encoders[name] = {"repo": spec.repo, "revision": commit}
    dataset = BENCHMARKS[benchmark_name]
    git = subprocess.run(
        ["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {
        "encoders": encoders,
        "dataset": {"repo": dataset.repo, "revision": dataset.revision},
        "versions": {p: version(p) for p in ("biomed-retrieval-jax", "eqx-zoo", "jax", "jaxlib")},
        "platform": f"{platform.system()} {platform.machine()}",
        "git": git,
    }


def main():
    """Parse the arguments, run the evaluation, and print and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    start = time.perf_counter()
    run = retrieve(
        MODELS[args.model], benchmark, batch_size=args.batch_size, dtype=DTYPES[args.dtype]
    )
    seconds = time.perf_counter() - start
    metrics = evaluate(run, benchmark.qrels)

    stem = f"{args.benchmark}-{args.dtype}"
    result = {
        "model": args.model,
        "benchmark": args.benchmark,
        "split": "test",
        "dtype": args.dtype,
        "metrics": metrics,
        "documents": len(benchmark.documents),
        "queries": len(benchmark.queries),
        "seconds": round(seconds, 1),  # loading, compiling, embedding and searching
        **_provenance(args.model, args.benchmark),
    }
    out = ROOT / "results" / args.model / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    runs = ROOT / "runs" / args.model / f"{stem}.json.gz"
    runs.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(runs, "wt", encoding="utf-8") as f:
        json.dump(run, f)

    print(
        f"{args.model} on {args.benchmark} ({args.dtype}): {result['documents']} documents, "
        f"{result['queries']} queries, {seconds:.0f} s; "
        f"ndcg@10 {round(metrics['ndcg@10'], 5)}, recall@100 {round(metrics['recall@100'], 5)}"
    )


if __name__ == "__main__":
    main()
