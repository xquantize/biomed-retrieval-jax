# biomed-retrieval-jax

Verified biomedical literature retrieval in JAX, built on
[eqx-zoo](https://github.com/xquantize/eqx-zoo).

> **Research demo, not for clinical use.** This project uses public research data only and
> must not be used to inform medical decisions.

Work in progress: a search command-line tool and a demo are next.

## Verified encoders

| Encoder | Checkpoint | Pooling | Maximum tokens |
|---|---|---|---|
| `medcpt-query` | [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder) | [CLS] | 64 |
| `medcpt-article` | [ncbi/MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) | [CLS], on (title, abstract) pairs | 512 |
| `pubmedbert-embeddings` | [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) | mean | 512 |
| `all-minilm-l6-v2` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | mean, normalised | 256 |

Each encoder is tested against Hugging Face Transformers, on inputs that cover every
truncation case. Tokenization must match exactly. Float32 embeddings must have an RMS error at
most twice PyTorch float32's, both measured against a float64 reference. Float64 embeddings
must match that reference to a relative error of 1e-13.

Errors over the test inputs (float32: RMS, the largest over batch sizes 1, 3 and 8; float64:
largest error relative to the largest value):

| Encoder | Platform | Ours, float32 | PyTorch, float32 | Ours, float64 |
|---|---|---|---|---|
| `medcpt-query` | arm64 (macOS) | 2.4e-7 | 6.8e-7 | 4.0e-16 |
| | x86-64 (CI) | 2.3e-7 | 2.8e-7 | 3.8e-16 |
| `medcpt-article` | arm64 (macOS) | 1.6e-7 | 6.1e-7 | 3.3e-16 |
| | x86-64 (CI) | 1.9e-7 | 2.4e-7 | 2.1e-16 |
| `pubmedbert-embeddings` | arm64 (macOS) | 7.0e-7 | 1.7e-6 | 4.3e-15 |
| | x86-64 (CI) | 6.1e-7 | 9.2e-7 | 7.1e-15 |
| `all-minilm-l6-v2` | arm64 (macOS) | 3.6e-8 | 4.4e-8 | 1.7e-15 |
| | x86-64 (CI) | 3.6e-8 | 3.0e-8 | 1.6e-15 |

In float32, our embeddings are as accurate as PyTorch's: between 0.22 and 1.19 times its RMS
error. In float64 they agree with Transformers to rounding, so both compute the same function.

Deliberately introduced bugs show what the tests catch
([`scripts/mutation_checks.py`](https://github.com/xquantize/biomed-retrieval-jax/blob/main/scripts/mutation_checks.py),
measured on arm64): a LayerNorm epsilon of 1e-6 instead of 1e-12 is caught by the float32 test
for every encoder, but 1e-8 is not; the float64 test catches it at 1e-10, with errors at least
2,000 times its tolerance. Dropping MedCPT's token types is caught at once.

## Evaluation harness

Retrieval is evaluated on the test splits of [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
(3,633 documents, 323 queries) and [SciFact](https://github.com/allenai/scifact) (5,183
documents, 300 queries), using MTEB's copies at the revisions MTEB evaluated. Every document is
scored against every query (exact search); each query keeps its top 1000, and nDCG@10 and
Recall@100 are computed as trec_eval computes them (checked against pytrec_eval, the reference
for BEIR and MTEB, to 2.2e-16).

The harness is validated by reproducing MTEB's published scores for
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (revision
`8b3219a9`, MTEB 1.12.75, from [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results)):

| Benchmark | Metric | Ours | Published |
|---|---|---|---|
| NFCorpus | nDCG@10 | 0.31594 | 0.31594 |
| NFCorpus | Recall@100 | 0.31151 | 0.31151 |
| SciFact | nDCG@10 | 0.64508 | 0.64508 |
| SciFact | Recall@100 | 0.925 | 0.925 |

The unrounded scores are identical on arm64 and x86-64 to eight decimals. Plausible method
mistakes change them visibly: on NFCorpus, documents without their titles give 0.30945 /
0.30782, and truncation at 128 tokens instead of 256 gives 0.31453 / 0.30943.

To evaluate a model: `uv run python scripts/evaluate.py medcpt nfcorpus` (add `--dtype bfloat16`
for bf16). To compare two runs, with the statistics below:
`uv run python scripts/compare_runs.py nfcorpus all-minilm-l6-v2:float32 medcpt:float32`.

## Results

Test splits, float32, on arm64 (macOS); every metric, unrounded and with its provenance (exact
model and dataset revisions, package versions, commit), is in
[`results/`](https://github.com/xquantize/biomed-retrieval-jax/tree/main/results). MedCPT is
scored by dot product, as its model card does; the others by cosine similarity. No published
results exist for MedCPT or NeuML's PubMedBERT under MTEB's setup, so these are new measurements,
made with the harness validated above.

| Model | NFCorpus nDCG@10 | NFCorpus Recall@100 | SciFact nDCG@10 | SciFact Recall@100 |
|---|---|---|---|---|
| MedCPT | **0.36545** | **0.34876** | **0.73699** | **0.98167** |
| all-MiniLM-L6-v2 (general-purpose baseline) | 0.31594 | 0.31151 | 0.64508 | 0.92500 |
| PubMedBERT embeddings (NeuML) | 0.26600 | 0.27002 | 0.68335 | 0.95667 |

Differences in nDCG@10, paired over queries: the mean difference, a 95% bootstrap interval
(10,000 resamples of the queries), and a two-sided permutation test (10,000 sign flips; 0.0001
is the smallest p it can give):

| Comparison | NFCorpus | SciFact |
|---|---|---|
| MedCPT − MiniLM | +0.0495 [+0.029, +0.072], p ≤ 0.0001 | +0.0919 [+0.059, +0.126], p ≤ 0.0001 |
| PubMedBERT − MiniLM | −0.0499 [−0.070, −0.029], p ≤ 0.0001 | +0.0383 [+0.010, +0.068], p = 0.008 |
| MedCPT − PubMedBERT | +0.0995 [+0.077, +0.123], p ≤ 0.0001 | +0.0537 [+0.022, +0.085], p = 0.001 |

MedCPT, trained on PubMed search logs, retrieves best on both benchmarks. PubMedBERT embeddings,
trained on PubMed text, are better than the general-purpose MiniLM on SciFact but significantly
worse on NFCorpus, whose queries are written in lay language: domain-specific training alone did
not guarantee better retrieval. With a Bonferroni correction for the 24 tests behind these
results (12 comparisons, two metrics; threshold 0.002), every difference above stays
significant except PubMedBERT − MiniLM on SciFact. Differences in Recall@100 follow the same
pattern, except on SciFact, where few queries change and the tests are inconclusive.

### bf16

Running the encoders in bf16 (embeddings still scored in float32) reshuffles near-tied
documents, but does not measurably change retrieval quality:

| Model | NFCorpus: nDCG@10 change | Same top-10 set | SciFact: nDCG@10 change | Same top-10 set |
|---|---|---|---|---|
| MedCPT | −0.0004 [−0.0021, +0.0014] | 75% | +0.0011 [−0.0014, +0.0040] | 84% |
| PubMedBERT | +0.0009 [−0.0000, +0.0019] | 86% | −0.0008 [−0.0024, +0.0003] | 81% |
| all-MiniLM-L6-v2 | −0.0006 [−0.0024, +0.0008] | 90% | +0.0015 [−0.0016, +0.0054] | 89% |

Only 24–63% of top-10 lists keep their exact order, but 97–99% of their documents remain in the
top 10, and no difference is significant (every p > 0.05). On this CPU, bf16 was 8–22% slower
than float32, so it is worth using only to save memory, or on hardware with native bf16.

## Limitations

- Two small benchmarks: 323 and 300 test queries. NFCorpus's relevance judgments were extracted
  automatically from links between NutritionFacts.org pages and the articles they cite.
- About 4% of documents exceed 512 tokens and are truncated (MedCPT and PubMedBERT); no query
  exceeds MedCPT's 64.
- MedCPT and PubMedBERT were evaluated on one platform (arm64); the reproduction of MTEB's
  scores ran identically on x86-64.
- MedCPT's released checkpoints differ slightly from the version in its paper, so the paper's
  scores are not directly comparable.
- Exact search only, without re-ranking.

## Data

The benchmarks are downloaded at evaluation time and never redistributed. Their original terms
apply: NFCorpus is free to use for academic purposes; SciFact is released under CC BY-NC 2.0.

## Licences and citation

- MedCPT: released by NCBI under a public-domain notice. Please cite Jin et al.,
  [MedCPT](https://doi.org/10.1093/bioinformatics/btad651), *Bioinformatics* 39(11), 2023.
- NeuML/pubmedbert-base-embeddings and sentence-transformers/all-MiniLM-L6-v2: Apache-2.0.
