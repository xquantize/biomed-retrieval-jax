"""Pretrained encoders, verified against Transformers in float32 and float64."""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from support.reference import capture, inputs
from support.registry import ENCODER_CHECKPOINTS

from biomed_retrieval_jax.encoders import ENCODERS, TextEncoder

# Our float32 RMS error may be at most this many times PyTorch float32's, both against float64.
# The RMS averages every element's rounding error; the largest error, compared at first, is one
# element's, and varied up to 2.2 times between batch sizes on identical code (and reached 1.92
# times PyTorch's for all-minilm-l6-v2 on x86-64). Largest RMS ratios measured over batch sizes
# 1, 3 and 8, on arm64: medcpt-query 0.35, medcpt-article 0.26, pubmedbert-embeddings 0.42,
# all-minilm-l6-v2 0.82; on x86-64: 0.84, 0.78, 0.67, 1.19 (same order). The float32 test
# catches a LayerNorm epsilon of 1e-6 instead of 1e-12 for every encoder, but not 1e-8; the
# float64 test catches both.
FLOAT32_RMS_RATIO = 2.0
# Largest float64 error (relative to the largest reference value) measured for medcpt-query,
# medcpt-article, pubmedbert-embeddings and all-minilm-l6-v2: 4.0e-16, 3.3e-16, 4.3e-15, 1.7e-15
# on arm64; 3.8e-16, 2.1e-16, 7.1e-15, 1.6e-15 on x86-64. A LayerNorm epsilon of 1e-10 instead
# of 1e-12 gives 2e-10 or more, so this separates correct from buggy by four orders.
FLOAT64_RELATIVE_TOLERANCE = 1e-13
# Sentence-transformers' float32 error over our float32 PyTorch reference's, measured: 1.07 and
# 1.01 on arm64, 0.90 and 1.16 on x86-64 (pubmedbert-embeddings, all-minilm-l6-v2). A wrong
# pooling would make it roughly 1e4.
SENTENCE_TRANSFORMERS_ERROR_RATIO = 10.0


@pytest.fixture(
    scope="session",
    params=[pytest.param(n, marks=pytest.mark.checkpoint(n)) for n in ENCODER_CHECKPOINTS],
)
def name(request):
    return request.param


@pytest.fixture(scope="session")
def reference(cached_reference, name):
    return cached_reference(name, lambda: capture(name))


@pytest.fixture(scope="session")
def encoder(name):
    return TextEncoder.from_spec(ENCODERS[name])


def max_error(value, reference):
    return float(np.abs(value - reference).max())


def rms_error(value, reference):
    return float(np.sqrt(np.mean((value - reference) ** 2)))


def test_tokenization_matches_transformers(name, encoder, reference):
    encodings = encoder.tokenizer.encode_batch(inputs(name))
    assert [len(e.ids) for e in encodings] == reference["lengths"].tolist()
    assert np.concatenate([e.ids for e in encodings]).tolist() == reference["token_ids"].tolist()
    types = np.concatenate([e.type_ids for e in encodings])
    assert types.tolist() == reference["token_types"].tolist()


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_float32_error_comparable_to_pytorch(name, encoder, reference, batch_size):
    exact = reference["embeddings_float64"]
    ours = encoder.embed(inputs(name), batch_size=batch_size)
    pytorch = reference["embeddings_float32"]
    ours_rms, pytorch_rms = rms_error(ours, exact), rms_error(pytorch, exact)
    print(
        f"{name}, batch size {batch_size}: float32 rms error {ours_rms:.2e}, PyTorch's"
        f" {pytorch_rms:.2e} (ratio {ours_rms / pytorch_rms:.2f}); largest error"
        f" {max_error(ours, exact):.2e}, PyTorch's {max_error(pytorch, exact):.2e}"
    )
    assert ours_rms <= FLOAT32_RMS_RATIO * pytorch_rms


def test_float64_matches_reference(name, reference, tmp_path):
    # Float64 mode is global in JAX, so this runs in its own process.
    out = tmp_path / "embeddings.npy"
    code = f"""
        import sys

        import jax
        import jax.numpy as jnp
        import numpy as np

        jax.config.update("jax_enable_x64", True)
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from support.reference import inputs

        from biomed_retrieval_jax.encoders import ENCODERS, TextEncoder

        encoder = TextEncoder.from_spec(ENCODERS[{name!r}], dtype=jnp.float64)
        np.save({str(out)!r}, encoder.embed(inputs({name!r}), batch_size=3))
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)
    ours = np.load(out)
    exact = reference["embeddings_float64"]
    assert ours.dtype == np.float64
    relative = max_error(ours, exact) / float(np.abs(exact).max())
    print(f"{name}: float64 relative error {relative:.2e}")
    assert relative < FLOAT64_RELATIVE_TOLERANCE


def test_reference_pooling_matches_sentence_transformers(name, reference):
    if "sentence_transformers_float32" not in reference:
        pytest.skip("no sentence-transformers pipeline")
    exact = reference["embeddings_float64"]
    st = max_error(reference["sentence_transformers_float32"], exact)
    pytorch = max_error(reference["embeddings_float32"], exact)
    print(f"{name}: sentence-transformers error {st:.2e}, our PyTorch reference's {pytorch:.2e}")
    assert st <= SENTENCE_TRANSFORMERS_ERROR_RATIO * pytorch
