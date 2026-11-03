"""Indexing and searching, on a tiny random BERT (no downloads)."""

import numpy as np
import pytest
from eqx_zoo import Encoder
from support.tiny import save_tiny_checkpoint
from tokenizers import Tokenizer

import biomed_retrieval_jax.cli as cli
from biomed_retrieval_jax.corpus import Document
from biomed_retrieval_jax.encoders import TextEncoder
from biomed_retrieval_jax.index import Searcher, embed_documents, load_index, save_index

TEXTS = [
    ("fever", "joint pain in older adults"),
    ("blood pressure", "chronic kidney disease"),
    ("vitamin", "bone density"),
    ("diabetes", "treatment of diabetes with drugs"),
    ("pain", "fever and joint pain"),
]
DOCUMENTS = [
    Document(f"PMC{i}", str(i), None, title, abstract, "J", "2026", "C", "CC BY")
    for i, (title, abstract) in enumerate(TEXTS)
]


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    checkpoint = save_tiny_checkpoint(tmp_path_factory.mktemp("tiny-bert"))
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    return TextEncoder(Encoder.from_pretrained(checkpoint, pooling="cls"), tokenizer, 32)


def test_save_and_load_index(encoder, tmp_path):
    embeddings = embed_documents(DOCUMENTS, encoder, pairs=True)
    assert embeddings.shape == (len(DOCUMENTS), 32) and embeddings.dtype == np.float32
    save_index(tmp_path, DOCUMENTS, embeddings, {"model": "test"})
    documents, loaded, manifest = load_index(tmp_path)
    assert documents == DOCUMENTS and np.array_equal(loaded, embeddings)
    assert manifest == {"model": "test"}
    np.save(tmp_path / "embeddings.npy", embeddings[:2])
    with pytest.raises(ValueError, match="5 documents but 2 embeddings"):
        load_index(tmp_path)


def test_each_document_is_its_own_nearest_neighbour(encoder):
    embeddings = embed_documents(DOCUMENTS, encoder, pairs=False)
    searcher = Searcher(DOCUMENTS, embeddings, encoder, cosine=True)
    for document in DOCUMENTS:
        results = searcher.search(f"{document.title} {document.abstract}", k=3)
        assert len(results) == 3
        assert results[0][0] == document
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)  # cosine similarity with itself
        assert results[0][1] >= results[1][1] >= results[2][1]


def test_cli_search_prints_the_notice_and_results(monkeypatch, capsys, tmp_path):
    class Stub:
        def search(self, query, k):
            return [(DOCUMENTS[1], 0.5)]

    monkeypatch.setattr(Searcher, "from_directory", classmethod(lambda cls, directory: Stub()))
    cli.main(["search", str(tmp_path), "kidney disease", "-k", "1"])
    out = capsys.readouterr().out
    assert out.startswith(cli.NOTICE)
    assert "1. blood pressure [0.50]" in out
    assert "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/" in out


def test_cli_labels_preprints(monkeypatch, capsys, tmp_path):
    preprint = Document(
        "PMC9",
        "9",
        None,
        "A preprint",
        "Text",
        "bioRxiv",
        "2026",
        "C",
        "CC BY",
        ["Journal Article", "Preprint"],
    )

    class Stub:
        def search(self, query, k):
            return [(preprint, 0.5), (DOCUMENTS[0], 0.4)]

    monkeypatch.setattr(Searcher, "from_directory", classmethod(lambda cls, directory: Stub()))
    cli.main(["search", str(tmp_path), "anything"])
    out = capsys.readouterr().out
    assert "bioRxiv, 2026 | CC BY | Preprint (not peer-reviewed) |" in out
    assert out.count("Preprint (not peer-reviewed)") == 1  # not on the journal article


def test_cli_bylines_end_with_one_full_stop(monkeypatch, capsys, tmp_path):
    several = Document(
        "PMC9", "9", None, "T", "A", "J", "2026", "C", "CC BY", [], ["Smith JA", "Doe B"]
    )
    one = Document("PMC8", "8", None, "T", "A", "J", "2026", "C", "CC BY", [], ["Lee S"])

    class Stub:
        def search(self, query, k):
            return [(several, 0.5), (one, 0.4)]

    monkeypatch.setattr(Searcher, "from_directory", classmethod(lambda cls, directory: Stub()))
    cli.main(["search", str(tmp_path), "anything"])
    out = capsys.readouterr().out
    assert "Smith JA et al. J, 2026" in out and "Lee S. J, 2026" in out
    assert ".." not in out
