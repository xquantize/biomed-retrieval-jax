# biomed-retrieval-jax

Verified biomedical literature retrieval in JAX, built on
[eqx-zoo](https://github.com/xquantize/eqx-zoo).

> **Research demo, not for clinical use.** This project uses public research data only and
> must not be used to inform medical decisions.

Work in progress: retrieval results, method and limitations will be documented here.

## Verified encoders

| Encoder | Checkpoint | Pooling | Maximum tokens |
|---|---|---|---|
| `medcpt-query` | [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder) | [CLS] | 64 |
| `medcpt-article` | [ncbi/MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) | [CLS], on (title, abstract) pairs | 512 |
| `pubmedbert-embeddings` | [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) | mean | 512 |

Each encoder is tested against Hugging Face Transformers, on inputs that cover every
truncation case. Tokenization must match exactly. Float32 embeddings must be within twice
PyTorch float32's error, both measured against a float64 reference. Float64 embeddings must
match that reference to a relative error of 1e-13.

Largest error over the test inputs (float32: absolute; float64: relative to the largest value):

| Encoder | Platform | Ours, float32 | PyTorch, float32 | Ours, float64 |
|---|---|---|---|---|
| `medcpt-query` | arm64 (macOS) | 1.0e-6 | 6.3e-6 | 4.0e-16 |
| | x86-64 (CI) | 1.1e-6 | 1.3e-6 | 3.8e-16 |
| `medcpt-article` | arm64 (macOS) | 9.2e-7 | 2.1e-6 | 3.3e-16 |
| | x86-64 (CI) | 6.8e-7 | 1.1e-6 | 2.1e-16 |
| `pubmedbert-embeddings` | arm64 (macOS) | 8.6e-6 | 1.3e-5 | 4.3e-15 |
| | x86-64 (CI) | 5.2e-6 | 5.2e-6 | 7.1e-15 |

In float32, our embeddings are as accurate as PyTorch's: more accurate on arm64, about the same
on x86-64. In float64 they agree with Transformers to rounding, so both compute the same
function.

Deliberately introduced bugs show what the tests catch
([`scripts/mutation_checks.py`](https://github.com/xquantize/biomed-retrieval-jax/blob/main/scripts/mutation_checks.py)):
a LayerNorm epsilon of 1e-10 instead of 1e-12 is invisible in float32 but gives a float64 error
of 2e-10, 2,000 times the tolerance; the float32 test catches it from 1e-6 (measured on
arm64).

## Licences and citation

- MedCPT: released by NCBI under a public-domain notice. Please cite Jin et al.,
  [MedCPT](https://doi.org/10.1093/bioinformatics/btad651), *Bioinformatics* 39(11), 2023.
- NeuML/pubmedbert-base-embeddings: Apache-2.0.
