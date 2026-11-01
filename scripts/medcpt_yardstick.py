"""Measure how accurate our JAX MedCPT embeddings are, against PyTorch and float64.

Measures, for the query and article encoders:

- that the `tokenizers` library, configured explicitly, reproduces Transformers' tokenization
  of queries and of (title, abstract) pairs, including truncation;
- the yardstick: how much PyTorch's float32 embeddings disagree with themselves across
  equally valid ways of running them (alone or batched with padding, eager or SDPA attention);
- the error of eqx-zoo's float32 embeddings against PyTorch's, alone and padded;
- the error of both float32 implementations against float64, and how closely the two float64
  implementations agree, which separates rounding from genuine algorithmic differences.

PyTorch's float32 self-disagreement underestimates its error (all its float32 code paths share
it), so float64 is the better yardstick: on arm64, eqx-zoo's float32 embeddings were 3-9 times
closer to it than PyTorch's, and the two float64 implementations agreed to ~1e-15.

Run with `uv run python scripts/medcpt_yardstick.py`.
"""

import platform

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from eqx_zoo import Encoder
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer

jax.config.update("jax_default_matmul_precision", "highest")

REPEATED = "persistent fatigue, joint pain and an intermittent low-grade fever"
QUERIES = [
    "diabetes treatment",
    "Which drugs lower blood pressure in older adults with chronic kidney disease?",
    "Hello!",
    "Symptoms: " + ", ".join([REPEATED] * 10),  # over 64 tokens: truncated
]
ARTICLES = [
    (
        "Vitamin D supplementation and bone density in older adults",
        "We randomised 400 adults aged over 65 to vitamin D or placebo for two years. "
        "Bone mineral density at the hip did not differ between the groups.",
    ),
    ("Short title", "Short abstract."),
    ("A long abstract", " ".join([REPEATED] * 80)),  # over 512 tokens: abstract truncated
    (" ".join([REPEATED] * 25), " ".join([REPEATED] * 80)),  # both long: both truncated
]
# Hub repository, inputs, maximum length, and whether inputs are (title, abstract) pairs.
CASES = [
    ("ncbi/MedCPT-Query-Encoder", QUERIES, 64, False),
    ("ncbi/MedCPT-Article-Encoder", ARTICLES, 512, True),
]


def _tokenizer(repo, max_length):
    """The `tokenizers` tokenizer with padding off and truncation set explicitly."""
    tok = Tokenizer.from_pretrained(repo)
    tok.no_padding()
    tok.enable_truncation(max_length=max_length, strategy="longest_first")
    return tok


def _check_tokenization(repo, inputs, max_length, pairs):
    """Compare the `tokenizers` library, configured explicitly, with Transformers."""
    hf = AutoTokenizer.from_pretrained(repo)
    rs = _tokenizer(repo, max_length)
    for x in inputs:
        args = x if pairs else (x,)
        full = len(hf(*args)["input_ids"])
        h = hf(*args, truncation=True, max_length=max_length)
        r = rs.encode(*args)
        match = h["input_ids"] == r.ids and h["token_type_ids"] == r.type_ids
        print(f"  {full:4d} tokens -> {len(r.ids):3d}; ids and token types match: {match}")


def _cls(model, hf, inputs, max_length, pairs):
    """PyTorch [CLS] embeddings as on the model cards, for a batch padded to its longest."""
    texts = [list(x) for x in inputs] if pairs else list(inputs)
    enc = hf(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        return model(**enc).last_hidden_state[:, 0, :].numpy()


@eqx.filter_jit
def _jax_batch(encoder, ids, types, mask):
    """Embeddings of a batch of token sequences, with `Encoder.embed`."""
    return jax.vmap(encoder.embed)(ids, mask, types)


def _jax_cls(encoder, tok, inputs, pairs, pad_to):
    """JAX embeddings, padded to `pad_to` tokens, or unpadded if `pad_to` is None."""
    if pad_to is None:
        tok.no_padding()
    else:
        tok.enable_padding(length=pad_to, pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    encodings = tok.encode_batch([tuple(x) for x in inputs] if pairs else list(inputs))
    ids = jnp.array([e.ids for e in encodings])
    types = jnp.array([e.type_ids for e in encodings])
    mask = jnp.array([e.attention_mask for e in encodings], dtype=bool)
    return np.asarray(_jax_batch(encoder, ids, types, mask))


def _report(name, diff, scale):
    """Print the largest absolute difference, and relative to the largest reference value."""
    worst = np.abs(diff).max()
    print(f"  {name:28s} max |diff| {worst:.2e}, relative {worst / scale:.2e}")


def _float32(repo, inputs, max_length, pairs):
    """Check tokenization, measure PyTorch against itself, and JAX against PyTorch."""
    print(f"\n{repo}")
    _check_tokenization(repo, inputs, max_length, pairs)
    hf = AutoTokenizer.from_pretrained(repo)
    models = {
        impl: AutoModel.from_pretrained(repo, dtype=torch.float32, attn_implementation=impl).eval()
        for impl in ("eager", "sdpa")
    }
    alone = np.concatenate([_cls(models["eager"], hf, [x], max_length, pairs) for x in inputs])
    again = np.concatenate([_cls(models["eager"], hf, [x], max_length, pairs) for x in inputs])
    batched = _cls(models["eager"], hf, inputs, max_length, pairs)
    batched_sdpa = _cls(models["sdpa"], hf, inputs, max_length, pairs)
    scale = np.abs(alone).max()
    norms = np.round(np.linalg.norm(alone, axis=1), 2)
    print(f"  embedding norms {norms}, largest |value| {scale:.2f}")
    print("  PyTorch against itself (the yardstick):")
    _report("repeat (alone, twice)", again - alone, scale)
    _report("batched vs alone", batched - alone, scale)
    _report("SDPA vs eager (batched)", batched_sdpa - batched, scale)

    encoder = Encoder.from_pretrained(repo, pooling="cls")
    tok = _tokenizer(repo, max_length)
    jax_alone = np.concatenate([_jax_cls(encoder, tok, [x], pairs, None) for x in inputs])
    jax_padded = _jax_cls(encoder, tok, inputs, pairs, max_length)
    print("  JAX against PyTorch (alone, eager):")
    _report("JAX alone", jax_alone - alone, scale)
    _report(f"JAX padded to {max_length}", jax_padded - alone, scale)
    return alone, jax_padded


def _float64(repo, inputs, max_length, pairs, alone, jax_padded):
    """Measure both float32 implementations against float64, and the two float64 ones."""
    hf = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo, dtype=torch.float64, attn_implementation="eager")
    exact = np.concatenate([_cls(model.eval(), hf, [x], max_length, pairs) for x in inputs])
    encoder = Encoder.from_pretrained(repo, dtype=jnp.float64, pooling="cls")
    jax64 = _jax_cls(encoder, _tokenizer(repo, max_length), inputs, pairs, max_length)
    assert exact.dtype == jax64.dtype == np.float64, (exact.dtype, jax64.dtype)
    scale = np.abs(exact).max()
    print(f"\n{repo}, against PyTorch float64:")
    _report("PyTorch float32 (alone)", alone - exact, scale)
    _report(f"JAX float32 (padded to {max_length})", jax_padded - exact, scale)
    _report("JAX float64 (padded)", jax64 - exact, scale)
    return exact


def main():
    """Print every measurement, float32 first and float64 after."""
    print(f"torch {torch.__version__}, jax {jax.__version__} on {platform.machine()}")
    float32 = [_float32(*case) for case in CASES]

    # Float64 mode is global, so it is switched on only after every float32 measurement.
    jax.config.update("jax_enable_x64", True)
    exact = [_float64(*case, *results) for case, results in zip(CASES, float32, strict=True)]

    (queries, jax_queries), (articles, jax_articles) = float32
    exact_scores = exact[0] @ exact[1].T
    scale = np.abs(exact_scores).max()
    print("\nquery-article scores (float64, dot products, rows are queries):")
    print(np.round(exact_scores, 2))
    _report("PyTorch float32 scores", queries @ articles.T - exact_scores, scale)
    _report("JAX float32 scores", jax_queries @ jax_articles.T - exact_scores, scale)


if __name__ == "__main__":
    main()
