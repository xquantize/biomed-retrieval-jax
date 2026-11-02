# biomed-retrieval-jax

Verified biomedical literature retrieval in JAX, built on
[eqx-zoo](https://github.com/xquantize/eqx-zoo).

> **Research demo, not for clinical use.** This project uses public research data only and
> must not be used to inform medical decisions.

Work in progress: retrieval results for MedCPT and PubMedBERT are next.

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

To evaluate a model: `uv run python scripts/evaluate.py all-minilm-l6-v2 nfcorpus`.

## Data

The benchmarks are downloaded at evaluation time and never redistributed. Their original terms
apply: NFCorpus is free to use for academic purposes; SciFact is released under CC BY-NC 2.0.

## Licences and citation

- MedCPT: released by NCBI under a public-domain notice. Please cite Jin et al.,
  [MedCPT](https://doi.org/10.1093/bioinformatics/btad651), *Bioinformatics* 39(11), 2023.
- NeuML/pubmedbert-base-embeddings and sentence-transformers/all-MiniLM-L6-v2: Apache-2.0.
