"""Publish the corpus and its embeddings as a dataset, and the demo page as a static Space.

Writes both to `data/publish` for review. With `--upload`, also uploads them to the Hugging Face
Hub, creating the repositories if needed.

Run with `uv run python scripts/publish.py`, then `uv run python scripts/publish.py --upload`.
"""

import argparse
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

from biomed_retrieval_jax.index import load_index

INDEX = Path("data/index-10k")
DEMO = Path("data/demo")
STAGING = Path("data/publish")
REPOSITORY = "https://github.com/xquantize/biomed-retrieval-jax"
DATASET_ID = "xquantize/biomed-retrieval-jax-pmc-10k"
SPACE_ID = "xquantize/biomed-retrieval-jax-demo"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE_ID}"


def _phrase(reason: str) -> str:
    """An exclusion reason from the manifest, phrased to follow a count."""
    if reason.startswith("publication type"):
        return f"of {reason}"
    return {"no abstract": "with no abstract", "retracted": "retracted articles"}.get(
        reason, reason
    )


def dataset_card(documents, manifest) -> str:
    """The dataset's README, filled in from the index's manifest."""
    corpus = manifest["corpus"]
    start, end = corpus["released"]
    licences = corpus["licences"]
    excluded = "; ".join(
        f"{n:,} {_phrase(reason)}"
        for reason, n in sorted(corpus["excluded"].items(), key=lambda x: -x[1])
    )
    preprints = sum(d.preprint for d in documents)
    journals = Counter(d.journal for d in documents).most_common(2)
    article = manifest["document_encoder"]
    query = manifest["query_encoder"]
    return f"""---
license: other
license_name: cc0-and-cc-by
license_link: {REPOSITORY}#data
language:
- en
pretty_name: PMC open-access abstracts (10,000) with MedCPT embeddings
size_categories:
- 10K<n<100K
task_categories:
- text-retrieval
tags:
- biomedical
- pubmed
- medcpt
- embeddings
configs:
- config_name: default
  data_files: documents.jsonl
---

# PMC open-access abstracts (10,000) with MedCPT embeddings

> **Research data, not for clinical use.** These abstracts and embeddings support a retrieval
> demo and must not be used to inform medical decisions.

Titles and abstracts of {len(documents):,} articles released in PubMed Central from {start} to
{end} under CC0 or CC BY licences, with embeddings from NCBI's MedCPT article encoder, computed
in JAX and verified against Hugging Face Transformers. Built with
[biomed-retrieval-jax]({REPOSITORY}), which searches them in its `biomed-search` command and its
[demo]({SPACE_URL}).

## Files

- `documents.jsonl`: one article per line, with `pmcid`, `pmid`, `doi`, `title`, `abstract`
  (structured abstracts' sections labelled as "LABEL: text"), `journal`, `year`, `citation`,
  `license` ("CC0" or "CC BY"), `publication_types` and `authors` ("Surname Initials", in
  PubMed's order).
- `embeddings.npy`: a float32 array of shape ({len(documents):,}, {manifest["dimension"]}); row
  *i* is the embedding of line *i* of `documents.jsonl`, from
  [{article["repo"]}](https://huggingface.co/{article["repo"]}) at revision
  `{article["revision"]}`, on (title, abstract) pairs, with [CLS] pooling, truncated at 512
  tokens.
- `manifest.json`: how the corpus and embeddings were built: the query, dates and seed, every
  exclusion, the exact model revisions and the commits.

## How the articles were selected

Candidates were the {corpus["candidates"]:,} articles released in PubMed Central from {start} to
{end} with a CC0 or CC BY licence and no embargo, found with NCBI's E-utilities. In a seeded
random order, each article's licence and retraction status were checked in its metadata on
PMC's Cloud Service, and its title, abstract, authors and publication types were taken from
PubMed, until {len(documents):,} articles passed. An article was kept only if it is indexed in
PubMed, has an abstract, and is not retracted, a retraction notice, an erratum or an expression
of concern. Of the articles examined, these were excluded: {excluded}. In a sample of 145 of
the articles not indexed in PubMed, 136 came from one journal's supplement of meeting
abstracts.

## Licences and attribution

Each article keeps its own licence, given in its `license` field: {licences.get("CC BY", 0):,}
are licensed CC BY and {licences.get("CC0", 0):,} CC0. Each record includes the article's
authors, title, journal citation and DOI; if you reuse the texts, please credit their authors.
The texts are taken from the articles' PubMed records. The code that built this dataset is
licensed Apache-2.0. MedCPT is released by NCBI under a public-domain notice; please cite Jin et
al., [MedCPT](https://doi.org/10.1093/bioinformatics/btad651), *Bioinformatics* 39(11), 2023.

## Use

```python
import json

import numpy as np
from huggingface_hub import snapshot_download

path = snapshot_download("{DATASET_ID}", repo_type="dataset")
documents = [json.loads(line) for line in open(f"{{path}}/documents.jsonl")]
embeddings = np.load(f"{{path}}/embeddings.npy")
```

To search, embed a query with [{query["repo"]}](https://huggingface.co/{query["repo"]})
(revision `{query["revision"]}`; at most 64 tokens, [CLS] pooling) and score the documents by
dot product, as `biomed-search` does.

## Limitations

- Two weeks of publications: many questions have no directly relevant article here.
- Only articles indexed in PubMed; retractions as known when the corpus was built.
- {preprints} articles are preprints, which have not been peer-reviewed: their
  `publication_types` include "Preprint".
- Embeddings are of the (title, abstract) pair truncated at 512 tokens; the texts are complete.
- The largest journals are {journals[0][0]} ({journals[0][1]:,} articles) and {journals[1][0]}
  ({journals[1][1]:,}).
"""


def space_card() -> str:
    """The Space's README: its configuration and a short description."""
    return f"""---
title: Biomedical literature search
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: MedCPT search in JAX over 10k open-access abstracts
---

A static demo of [biomed-retrieval-jax]({REPOSITORY}): precomputed results for eight example
queries over the [{DATASET_ID}]({DATASET_URL}) corpus. **Research demo, not for clinical use.**
"""


def main():
    """Stage the dataset and the Space, and upload them with `--upload`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upload", action="store_true", help="upload to the Hugging Face Hub")
    args = parser.parse_args()

    documents, _, manifest = load_index(INDEX)
    dataset, space = STAGING / "dataset", STAGING / "space"
    shutil.rmtree(STAGING, ignore_errors=True)
    dataset.mkdir(parents=True)
    space.mkdir(parents=True)
    for name in ("documents.jsonl", "embeddings.npy", "manifest.json"):
        shutil.copy2(INDEX / name, dataset / name)
    (dataset / "README.md").write_text(dataset_card(documents, manifest), encoding="utf-8")
    for name in ("index.html", "results.json"):
        shutil.copy2(DEMO / name, space / name)
    (space / "README.md").write_text(space_card(), encoding="utf-8")
    for folder in (dataset, space):
        sizes = ", ".join(
            f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted(folder.iterdir())
        )
        print(f"staged {folder}: {sizes}")

    if args.upload:
        git = subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True
        ).stdout.strip()
        api = HfApi()
        for repo_id, repo_type, folder in (
            (DATASET_ID, "dataset", dataset),
            (SPACE_ID, "space", space),
        ):
            extra = {"space_sdk": "static"} if repo_type == "space" else {}
            api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, **extra)
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=folder,
                commit_message=f"Publish from biomed-retrieval-jax at {git}",
            )
            print(f"uploaded {folder} -> {commit.commit_url}")


if __name__ == "__main__":
    main()
