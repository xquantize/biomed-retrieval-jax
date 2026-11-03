"""Build the static demo: run the example queries and render the results as one HTML page.

Writes `index.html` and `results.json` (every result with its score) to `data/demo`.

Run with `uv run python scripts/build_demo.py`.
"""

import json
from pathlib import Path

from biomed_retrieval_jax.demo import render_page
from biomed_retrieval_jax.index import Searcher, load_index

INDEX = Path("data/index-10k")
OUT = Path("data/demo")
REPOSITORY = "https://github.com/xquantize/biomed-retrieval-jax"
DATASET = "https://huggingface.co/datasets/xquantize/biomed-retrieval-jax-pmc-10k"
QUERIES = [
    "sleep problems in older adults",
    "antibiotic resistance in hospital patients",
    "statins and breast cancer survival",
    "exercise and depression",
    "air pollution and asthma in children",
    "gut microbiome and obesity",
    "long-term effects of COVID-19",
    "screening for diabetic retinopathy with deep learning",
]
ABOUT = [
    "Each example shows the five abstracts that "
    '<a href="https://huggingface.co/ncbi/MedCPT-Query-Encoder">MedCPT</a> ranks highest for the'
    " query, precomputed with the <code>biomed-search</code> command. Queries are embedded with"
    " MedCPT’s query encoder and abstracts with its article encoder, both running in JAX with"
    ' <a href="https://github.com/xquantize/eqx-zoo">eqx-zoo</a> and verified against Hugging'
    " Face Transformers.",
    "The corpus is small: 10,000 openly licensed abstracts from two weeks of publications. Some"
    " questions have no directly relevant article in it, and the search still returns its"
    " closest matches. It includes animal and laboratory research. Preprints, which have not"
    " been peer-reviewed, are labelled.",
    "Retrieval quality, the method and the limitations are described in the"
    f' <a href="{REPOSITORY}">'
    "repository</a>, which also explains how to search with your own queries. The corpus and its"
    f' embeddings are published as a <a href="{DATASET}">dataset</a>.',
]


def main():
    """Run the example queries, and write the page and the raw results."""
    documents, _, manifest = load_index(INDEX)
    searcher = Searcher.from_directory(INDEX)
    results = [(query, searcher.search(query, k=5)) for query in QUERIES]
    corpus = manifest["corpus"]
    licences = ", ".join(f"{n:,} {name}" for name, n in sorted(corpus["licences"].items()))
    provenance = [
        (
            "Corpus",
            f"{corpus['documents']:,} abstracts released in PubMed Central from"
            f" {corpus['released'][0]} to {corpus['released'][1]}",
        ),
        ("Licences", licences),
        ("Preprints", f"{sum(d.preprint for d in documents):,}, labelled"),
        (
            "Query encoder",
            f"{manifest['query_encoder']['repo']} at {manifest['query_encoder']['revision'][:8]}",
        ),
        (
            "Article encoder",
            f"{manifest['document_encoder']['repo']} at"
            f" {manifest['document_encoder']['revision'][:8]}",
        ),
        ("Embeddings computed", f"{manifest['built'][:10]}, at commit {manifest['git']}"),
        *(
            [("Documents updated", f"{r['date'][:10]}, at commit {r['git']}: {r['reason']}")]
            if (r := manifest.get("documents_rewritten"))
            else []
        ),
    ]
    page = render_page([(q, [d for d, _ in r]) for q, r in results], ABOUT, provenance)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    raw = [
        {"query": q, "results": [{"pmcid": d.pmcid, "title": d.title, "score": s} for d, s in r]}
        for q, r in results
    ]
    (OUT / "results.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT / 'index.html'} ({len(page):,} characters) and {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
