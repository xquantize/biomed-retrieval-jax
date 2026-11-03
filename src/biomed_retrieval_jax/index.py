"""Search a corpus: embed its documents once, save them, and answer queries against them."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float

from biomed_retrieval_jax.corpus import Document, read_corpus, write_corpus
from biomed_retrieval_jax.encoders import ENCODERS, TextEncoder
from biomed_retrieval_jax.retrieval import MODELS, document_input
from biomed_retrieval_jax.search import search


def embed_documents(
    documents: Sequence[Document], encoder: TextEncoder, *, pairs: bool, batch_size: int = 32
) -> Float[np.ndarray, "n dim"]:
    """Embed documents' titles and abstracts.

    Args:
        documents: The documents.
        encoder: The document encoder.
        pairs: Whether the encoder takes (title, abstract) pairs; otherwise both are joined.
        batch_size: Documents per batch.

    Returns:
        One float32 embedding per document, in order.
    """
    inputs = [document_input((d.title, d.abstract), pairs) for d in documents]
    return encoder.embed(inputs, batch_size=batch_size).astype(np.float32)


def save_index(
    directory: Path,
    documents: Sequence[Document],
    embeddings: Float[np.ndarray, "n dim"],
    manifest: dict,
) -> None:
    """Write an index: `documents.jsonl`, `embeddings.npy` and `manifest.json`.

    Args:
        directory: Where to write; created if needed.
        documents: The documents, in the order of their embeddings.
        embeddings: Their embeddings.
        manifest: How the index was built; must name its retrieval model under `"model"`.
    """
    write_corpus(directory, documents, manifest)
    np.save(directory / "embeddings.npy", embeddings)


def load_index(directory: Path) -> tuple[list[Document], Float[np.ndarray, "n dim"], dict]:
    """Read an index written by `save_index`.

    Args:
        directory: The index directory.

    Returns:
        The documents, their embeddings and the manifest.

    Raises:
        ValueError: If the numbers of documents and embeddings differ.
    """
    documents = read_corpus(directory)
    embeddings = np.load(directory / "embeddings.npy")
    if len(documents) != len(embeddings):
        raise ValueError(f"{len(documents)} documents but {len(embeddings)} embeddings")
    return documents, embeddings, json.loads((directory / "manifest.json").read_text())


class Searcher:
    """Answers queries against an index with a query encoder."""

    def __init__(
        self,
        documents: Sequence[Document],
        embeddings: Float[np.ndarray, "n dim"],
        query_encoder: TextEncoder,
        *,
        cosine: bool,
    ):
        """Search documents with precomputed embeddings.

        Args:
            documents: The documents.
            embeddings: Their embeddings, in the same order.
            query_encoder: Embeds queries in the same space as the documents.
            cosine: Whether to score by cosine similarity; otherwise by dot product.
        """
        self.documents = list(documents)
        self.embeddings = embeddings
        self.query_encoder = query_encoder
        self.cosine = cosine

    @classmethod
    def from_directory(cls, directory: Path) -> "Searcher":
        """Load an index, and the query encoder of the retrieval model it was built with.

        Args:
            directory: The index directory.

        Returns:
            The searcher.
        """
        documents, embeddings, manifest = load_index(directory)
        model = MODELS[manifest["model"]]
        encoder = TextEncoder.from_spec(ENCODERS[model.query_encoder])
        return cls(documents, embeddings, encoder, cosine=model.cosine)

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Find the documents that best match a query.

        Args:
            query: The query.
            k: Number of documents to return.

        Returns:
            The best documents and their scores, highest first.
        """
        embedding = self.query_encoder.embed([query], batch_size=1).astype(np.float32)
        scores, indices = search(embedding, self.embeddings, k, cosine=self.cosine)
        return [(self.documents[i], float(s)) for s, i in zip(scores[0], indices[0], strict=True)]
