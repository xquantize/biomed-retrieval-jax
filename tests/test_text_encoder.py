"""TextEncoder's tokenization and batching, on a tiny random BERT (no downloads)."""

import jax.numpy as jnp
import numpy as np
import pytest
from eqx_zoo import Encoder
from support.tiny import save_tiny_checkpoint
from tokenizers import Tokenizer

from biomed_retrieval_jax.encoders import TextEncoder, bucket_length

# Batched embeddings must match each input embedded alone, to this fraction of the largest
# value. Float32 rounding from different padded lengths is ~1e-7 to 1e-6; ordering or padding
# bugs give ~1e-1. Numerical accuracy is checked against Transformers by the checkpoint tests.
BATCHING_TOLERANCE = 1e-5

TEXTS = [
    "fever",
    "joint pain and fever in older adults",
    "the",
    "chronic kidney disease and blood pressure drugs for older adults with diabetes",
    "vitamin and bone density",
    "treatment",
    "pain",
]
PAIRS = [
    ("vitamin", "bone density in older adults"),
    ("short title", "short abstract"),
    ("chronic kidney disease in older adults with diabetes", "blood pressure"),
]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return save_tiny_checkpoint(tmp_path_factory.mktemp("tiny-bert"))


def make_encoder(checkpoint, pooling="cls", max_length=32):
    encoder = Encoder.from_pretrained(checkpoint, pooling=pooling)
    return TextEncoder(encoder, Tokenizer.from_file(str(checkpoint / "tokenizer.json")), max_length)


def alone(encoder, x):
    """Embed one input directly with eqx-zoo, without padding or batching."""
    e = encoder.tokenizer.encode(*(x if isinstance(x, tuple) else (x,)))
    ids, types = jnp.array(e.ids), jnp.array(e.type_ids)
    return np.asarray(encoder.encoder.embed(ids, jnp.ones(len(e.ids), dtype=bool), types))


@pytest.mark.parametrize(
    ("length", "max_length", "expected"),
    [
        (1, 512, 16),
        (16, 512, 16),
        (17, 512, 32),
        (100, 512, 128),
        (512, 512, 512),
        (40, 64, 64),
        (5, 10, 10),
    ],
)
def test_bucket_length(length, max_length, expected):
    assert bucket_length(length, max_length) == expected


@pytest.mark.parametrize("pooling", ["cls", "mean"])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
@pytest.mark.parametrize("inputs", [TEXTS, PAIRS], ids=["texts", "pairs"])
def test_batched_embeddings_match_each_input_alone(checkpoint, pooling, batch_size, inputs):
    encoder = make_encoder(checkpoint, pooling)
    batched = encoder.embed(inputs, batch_size=batch_size)
    expected = np.stack([alone(encoder, x) for x in inputs])
    assert batched.shape == expected.shape
    relative = float(np.abs(batched - expected).max() / np.abs(expected).max())
    print(f"{pooling}, batch size {batch_size}: relative difference {relative:.1e}")
    assert relative < BATCHING_TOLERANCE


def test_pairs_use_both_token_types(checkpoint):
    encoder = make_encoder(checkpoint)
    e = encoder.tokenizer.encode(*PAIRS[0])
    first_sep = e.ids.index(e.ids[-1])  # the [SEP] after the title
    assert set(e.type_ids[: first_sep + 1]) == {0}
    assert set(e.type_ids[first_sep + 1 :]) == {1}


def test_truncates_to_max_length(checkpoint):
    encoder = make_encoder(checkpoint, max_length=8)
    assert len(encoder.tokenizer.encode(TEXTS[3]).ids) == 8
    first, second = PAIRS[2]
    e = encoder.tokenizer.encode(first, second)
    assert len(e.ids) == 8 and set(e.type_ids) == {0, 1}  # longest first: both segments kept


def test_pad_id_comes_from_the_tokenizer(checkpoint):
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    tokenizer.enable_padding(pad_id=3, pad_token="[SEP]")
    encoder = TextEncoder(Encoder.from_pretrained(checkpoint, pooling="cls"), tokenizer, 32)
    assert encoder.pad_id == 3
    assert encoder.tokenizer.padding is None  # stored padding settings are switched off
    assert make_encoder(checkpoint).pad_id == 0  # no stored settings: the [PAD] token


def test_rejects_empty_inputs(checkpoint):
    with pytest.raises(ValueError, match="no inputs"):
        make_encoder(checkpoint).embed([])
