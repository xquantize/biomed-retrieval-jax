"""Retrieval benchmarks: MTEB's copies of BEIR datasets, at the revisions MTEB evaluated."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class BenchmarkSpec:
    """Where a benchmark's files are, and the data's terms of use.

    Attributes:
        repo: Hugging Face Hub dataset repository with `corpus.jsonl`, `queries.jsonl` and
            `qrels/<split>.tsv` in BEIR's format.
        revision: Git revision of the repository.
        terms: The data's terms of use, from its original source.
    """

    repo: str
    revision: str
    terms: str


BENCHMARKS = {
    # Revisions from MTEB's published results (MTEB 1.12.75).
    "nfcorpus": BenchmarkSpec(
        "mteb/nfcorpus",
        "ec0fa4fe99da2ff19ca1214b7966684033a58814",
        "Free to use for academic purposes: https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
    ),
    "scifact": BenchmarkSpec(
        "mteb/scifact",
        "0228b52cf27578f30900b9e5271d331663a030d7",
        "CC BY-NC 2.0: https://github.com/allenai/scifact",
    ),
}


@dataclass(frozen=True)
class Benchmark:
    """A retrieval benchmark's documents, and one split's queries and relevance judgments.

    Attributes:
        documents: Each document's title and text, by id.
        queries: Each query's text, by id, for the queries judged in the split.
        qrels: Each query's judged documents and their relevance grades.
    """

    documents: dict[str, tuple[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_benchmark(name: str, split: str = "test") -> Benchmark:
    """Download a benchmark's files, or read them from the cache, and parse one split.

    Args:
        name: A key of `BENCHMARKS`.
        split: Which relevance judgments to use.

    Returns:
        The benchmark.
    """
    spec = BENCHMARKS[name]
    files = ["corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"]
    directory = snapshot_download(
        spec.repo, repo_type="dataset", revision=spec.revision, allow_patterns=files
    )
    return read_benchmark(Path(directory), split)


def read_benchmark(directory: Path, split: str = "test") -> Benchmark:
    """Parse a benchmark in BEIR's format from a directory.

    Args:
        directory: Contains `corpus.jsonl`, `queries.jsonl` and `qrels/<split>.tsv`.
        split: Which relevance judgments to use.

    Returns:
        The benchmark. A document without a title has an empty one; only the queries judged in
        the split are kept.
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(directory / "qrels" / f"{split}.tsv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    with open(directory / "corpus.jsonl", encoding="utf-8") as f:
        documents = {d["_id"]: (d.get("title", ""), d["text"]) for d in map(json.loads, f)}
    with open(directory / "queries.jsonl", encoding="utf-8") as f:
        queries = {q["_id"]: q["text"] for q in map(json.loads, f) if q["_id"] in qrels}
    return Benchmark(documents, queries, qrels)
