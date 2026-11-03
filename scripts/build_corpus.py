"""Build a corpus of CC0 and CC BY abstracts released in PMC in a date range.

Writes `documents.jsonl` and `manifest.json` (every parameter, the exclusions and the licence
counts) to the output directory. Set `NCBI_EMAIL`, and optionally `NCBI_API_KEY`, as NCBI asks.

Responses are cached in `data/cache` (see `--cache`), so an interrupted build resumes.

Run with `uv run python scripts/build_corpus.py --size 1000 --out data/corpus-1k`.
"""

import argparse
import statistics
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import biomed_retrieval_jax.corpus as corpus
from biomed_retrieval_jax.corpus import QUERY, build_corpus, search_pmc, write_corpus


def main():
    """Parse the arguments, build the corpus, write it and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 1, 7))
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    corpus.CACHE_DIRECTORY = args.cache

    pmcids = search_pmc(args.start, args.end)
    print(f"{len(pmcids)} candidates released {args.start} to {args.end}")
    documents, excluded = build_corpus(pmcids, args.size, seed=args.seed)
    licences = Counter(d.license for d in documents)
    manifest = {
        "query": QUERY,
        "released": [args.start.isoformat(), args.end.isoformat()],
        "seed": args.seed,
        "size": args.size,
        "candidates": len(pmcids),
        "documents": len(documents),
        "excluded": excluded,
        "licences": dict(licences),
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_corpus(args.out, documents, manifest)

    lengths = [len(d.abstract) for d in documents]
    print(f"{len(documents)} documents written to {args.out}")
    print(f"  excluded among those examined: {excluded}")
    print(f"  licences: {dict(licences)}")
    print(f"  abstract characters: median {statistics.median(lengths):.0f}, max {max(lengths)}")
    print(f"  years: {dict(Counter(d.year for d in documents).most_common(4))}")
    print(f"  journals: {Counter(d.journal for d in documents).most_common(5)}")
    print(f"  example: {documents[0].title[:80]} ({documents[0].citation[:60]}...)")


if __name__ == "__main__":
    main()
