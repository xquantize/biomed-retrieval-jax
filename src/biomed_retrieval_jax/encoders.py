"""Embed texts with eqx-zoo encoders: tokenization, length-bucketed batching and pooling."""

from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from eqx_zoo import Encoder
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int
from tokenizers import Encoding, Tokenizer

MIN_BUCKET = 16


@dataclass(frozen=True)
class EncoderSpec:
    """How to run a pretrained checkpoint as a text encoder.

    Attributes:
        repo: Hugging Face Hub repository.
        max_length: Maximum tokens per input, special tokens included. Longer inputs are
            truncated; (title, abstract) pairs lose tokens from the longer part first.
        pooling: Pooling for a checkpoint without a sentence-transformers pipeline
            (`"cls"`, `"mean"` or `"last_token"`), or `None` to use the checkpoint's own.
    """

    repo: str
    max_length: int
    pooling: str | None = None


ENCODERS = {
    # MedCPT: [CLS] pooling without normalisation, and maximum lengths, from the model cards.
    "medcpt-query": EncoderSpec("ncbi/MedCPT-Query-Encoder", max_length=64, pooling="cls"),
    "medcpt-article": EncoderSpec("ncbi/MedCPT-Article-Encoder", max_length=512, pooling="cls"),
    # Mean pooling without normalisation, from its sentence-transformers pipeline.
    "pubmedbert-embeddings": EncoderSpec("NeuML/pubmedbert-base-embeddings", max_length=512),
}


def bucket_length(length: int, max_length: int) -> int:
    """Return the padded length for a batch whose longest input has `length` tokens.

    Lengths are rounded up to a power of two, at least `MIN_BUCKET` and at most
    `max_length`, so that only a few input shapes are ever compiled.

    Args:
        length: Tokens in the batch's longest input.
        max_length: The encoder's maximum length.

    Returns:
        The length to pad the batch to.
    """
    bucket = MIN_BUCKET
    while bucket < length:
        bucket *= 2
    return min(bucket, max_length)


@eqx.filter_jit
def _embed_batch(
    encoder: Encoder,
    ids: Int[Array, "batch seq"],
    types: Int[Array, "batch seq"],
    mask: Bool[Array, "batch seq"],
) -> Float[Array, "batch dim"]:
    return jax.vmap(encoder.embed)(ids, mask, types)


class TextEncoder:
    """A pretrained encoder with its tokenizer, embedding texts or (title, abstract) pairs.

    Inputs are sorted by length and embedded in fixed-size batches, each padded to a bucketed
    length, so that few shapes are compiled and little padding is computed. Embeddings are
    returned in the order of the inputs.
    """

    def __init__(self, encoder: Encoder, tokenizer: Tokenizer, max_length: int):
        """Wrap an encoder and its tokenizer.

        Args:
            encoder: The eqx-zoo encoder. Its `embed` is used, so it needs a pooling.
            tokenizer: Its tokenizer. Truncation is set to `max_length` and padding is
                switched off, replacing any settings stored with the tokenizer.
            max_length: Maximum tokens per input, special tokens included.

        Raises:
            ValueError: If the tokenizer defines no padding token.
        """
        padding = tokenizer.padding
        pad_id = padding["pad_id"] if padding else tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError("the tokenizer defines no padding token")
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = pad_id

    @classmethod
    def from_spec(cls, spec: EncoderSpec, *, dtype: DTypeLike = jnp.float32) -> "TextEncoder":
        """Load a pretrained encoder and its tokenizer from the Hugging Face Hub.

        Args:
            spec: Which checkpoint to load, and how to run it.
            dtype: Dtype of the parameters and embeddings.

        Returns:
            The text encoder.
        """
        encoder = Encoder.from_pretrained(spec.repo, dtype=dtype, pooling=spec.pooling)
        return cls(encoder, Tokenizer.from_pretrained(spec.repo), spec.max_length)

    def embed(
        self, inputs: Sequence[str] | Sequence[tuple[str, str]], *, batch_size: int = 32
    ) -> Float[np.ndarray, "n dim"]:
        """Embed texts, or (title, abstract) pairs.

        Args:
            inputs: Texts, or pairs whose two parts are encoded as two segments.
            batch_size: Inputs per batch. The last batch is filled up with copies of its first
                input, which are discarded, so that every batch has the same shape.

        Returns:
            One embedding per input, in the order given.

        Raises:
            ValueError: If there are no inputs.
        """
        if not inputs:
            raise ValueError("no inputs to embed")
        encodings = self.tokenizer.encode_batch(list(inputs))
        order = np.argsort([len(e.ids) for e in encodings], kind="stable")
        chunks = []
        for start in range(0, len(order), batch_size):
            batch = [encodings[i] for i in order[start : start + batch_size]]
            length = bucket_length(max(len(e.ids) for e in batch), self.max_length)
            filled = batch + [batch[0]] * (batch_size - len(batch))
            embedded = _embed_batch(self.encoder, *self._pad(filled, length))
            chunks.append(np.asarray(embedded)[: len(batch)])
        sorted_embeddings = np.concatenate(chunks)
        embeddings = np.empty_like(sorted_embeddings)
        embeddings[order] = sorted_embeddings
        return embeddings

    def _pad(
        self, encodings: Sequence[Encoding], length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Token ids, token types and attention masks, padded on the right to `length`."""
        ids = np.full((len(encodings), length), self.pad_id, dtype=np.int32)
        types = np.zeros((len(encodings), length), dtype=np.int32)
        mask = np.zeros((len(encodings), length), dtype=bool)
        for row, e in enumerate(encodings):
            ids[row, : len(e.ids)] = e.ids
            types[row, : len(e.ids)] = e.type_ids
            mask[row, : len(e.ids)] = True
        return ids, types, mask
