"""The `biomed-search` command: index a corpus of abstracts, and search it.

A research demo, not for clinical use.
"""

import argparse
import json
import subprocess
import textwrap
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

from biomed_retrieval_jax.corpus import read_corpus
from biomed_retrieval_jax.encoders import ENCODERS, TextEncoder
from biomed_retrieval_jax.index import Searcher, embed_documents, save_index
from biomed_retrieval_jax.retrieval import MODELS

NOTICE = "Research demo, not for clinical use: results must not inform medical decisions."


def _revision(name: str) -> dict:
    """An encoder's repository and the exact commit that is loaded."""
    spec = ENCODERS[name]
    return {
        "repo": spec.repo,
        "revision": HfApi().model_info(spec.repo, revision=spec.revision).sha,
    }


def _index(args: argparse.Namespace) -> None:
    documents = read_corpus(args.corpus)
    model = MODELS[args.model]
    encoder = TextEncoder.from_spec(ENCODERS[model.document_encoder])
    start = time.perf_counter()
    embeddings = embed_documents(documents, encoder, pairs=model.pairs, batch_size=args.batch_size)
    seconds = time.perf_counter() - start
    git = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "model": args.model,
        "query_encoder": _revision(model.query_encoder),
        "document_encoder": _revision(model.document_encoder),
        "corpus": json.loads((args.corpus / "manifest.json").read_text()),
        "documents": len(documents),
        "dimension": int(embeddings.shape[1]),
        "seconds": round(seconds, 1),
        "git": git,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_index(args.index, documents, embeddings, manifest)
    print(
        f"indexed {len(documents)} documents with {args.model} in {seconds:.0f} s"
        f" ({len(documents) / seconds:.1f} per second) -> {args.index}"
    )


def _search(args: argparse.Namespace) -> None:
    results = Searcher.from_directory(args.index).search(args.query, k=args.k)
    print(NOTICE)
    for rank, (document, score) in enumerate(results, start=1):
        preprint = " | Preprint (not peer-reviewed)" if document.preprint else ""
        print(f"\n{rank}. {document.title} [{score:.2f}]")
        print(
            f"   {document.journal}, {document.year} | {document.license}{preprint} |"
            f" https://pmc.ncbi.nlm.nih.gov/articles/{document.pmcid}/"
        )
        print(textwrap.indent(textwrap.fill(textwrap.shorten(document.abstract, 240), 96), "   "))


def main(argv: Sequence[str] | None = None) -> None:
    """Run `biomed-search index CORPUS INDEX` or `biomed-search search INDEX QUERY`.

    Args:
        argv: The arguments, without the program name; by default, the command line's.
    """
    parser = argparse.ArgumentParser(prog="biomed-search", description=f"{__doc__.split('.')[0]}.")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="embed a corpus and save it as an index")
    index.add_argument("corpus", type=Path, help="a corpus directory from scripts/build_corpus.py")
    index.add_argument("index", type=Path, help="where to write the index")
    index.add_argument("--model", choices=MODELS, default="medcpt")
    index.add_argument("--batch-size", type=int, default=32)
    query = commands.add_parser("search", help="search an index")
    query.add_argument("index", type=Path, help="an index directory")
    query.add_argument("query", help="the query, in quotes")
    query.add_argument("-k", type=int, default=10, help="number of results")
    args = parser.parse_args(argv)
    _index(args) if args.command == "index" else _search(args)
