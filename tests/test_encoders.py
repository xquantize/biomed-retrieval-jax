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

# Largest float64 error measured on arm64 (relative to the largest reference value): 4.0e-16
# (MedCPT query), 3.3e-16 (MedCPT article), 4.3e-15 (PubMedBERT embeddings). Provisional:
# still to be calibrated on x86 CI and checked with a deliberately introduced bug.
FLOAT64_RELATIVE_TOLERANCE = 1e-13
# Sentence-transformers' float32 error over our float32 PyTorch reference's, measured: 1.07.
# A wrong pooling would make it roughly 1e4.
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


def test_tokenization_matches_transformers(name, encoder, reference):
    encodings = encoder.tokenizer.encode_batch(inputs(name))
    assert [len(e.ids) for e in encodings] == reference["lengths"].tolist()
    assert np.concatenate([e.ids for e in encodings]).tolist() == reference["token_ids"].tolist()
    types = np.concatenate([e.type_ids for e in encodings])
    assert types.tolist() == reference["token_types"].tolist()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_float32_at_least_as_accurate_as_pytorch(name, encoder, reference, batch_size):
    exact = reference["embeddings_float64"]
    ours = max_error(encoder.embed(inputs(name), batch_size=batch_size), exact)
    pytorch = max_error(reference["embeddings_float32"], exact)
    print(f"{name}, batch size {batch_size}: float32 error {ours:.2e}, PyTorch's {pytorch:.2e}")
    assert ours <= pytorch


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
