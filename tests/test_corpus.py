"""Corpus building: PubMed parsing, version choice and filtering (no network)."""

import json

import pytest

import biomed_retrieval_jax.corpus as corpus

PUBMED = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue>
      <Title>Journal One</Title></Journal>
    <ArticleTitle>A <i>structured</i> abstract</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">Some <i>Escherichia coli</i>
        strains.</AbstractText>
      <AbstractText Label="RESULTS">CO<sub>2</sub> rose.</AbstractText>
    </Abstract>
    <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
  </Article></MedlineCitation></PubmedArticle>
  <PubmedArticle><MedlineCitation><PMID>2</PMID><Article>
    <Journal><JournalIssue><PubDate><MedlineDate>2025 Dec-2026 Jan</MedlineDate></PubDate>
      </JournalIssue><Title>Journal Two</Title></Journal>
    <ArticleTitle>No abstract</ArticleTitle>
    <PublicationTypeList><PublicationType>Editorial</PublicationType></PublicationTypeList>
  </Article></MedlineCitation></PubmedArticle>
  <PubmedArticle><MedlineCitation><PMID>6</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue>
      <Title>Journal Three</Title></Journal>
    <ArticleTitle>Retraction: an earlier article</ArticleTitle>
    <Abstract><AbstractText>This article has been retracted.</AbstractText></Abstract>
    <PublicationTypeList><PublicationType>Journal Article</PublicationType>
      <PublicationType>Retraction Notice</PublicationType></PublicationTypeList>
  </Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_keeps_inline_text_and_labels():
    records = corpus.parse_pubmed(PUBMED)
    assert records["1"] == {
        "title": "A structured abstract",
        "abstract": "BACKGROUND: Some Escherichia coli strains. RESULTS: CO2 rose.",
        "journal": "Journal One",
        "year": "2026",
        "types": ["Journal Article"],
    }
    assert records["2"]["abstract"] == "" and records["2"]["year"] == "2025"


def metadata(pmcid, version, *, manuscript=False, licence="CC BY", retracted=False, pmid=1):
    return {
        "pmcid": pmcid,
        "version": version,
        "pmid": pmid,
        "doi": None,
        "title": "T",
        "citation": "C",
        "is_manuscript": manuscript,
        "is_retracted": retracted,
        "license_code": licence,
    }


@pytest.mark.parametrize(
    ("manuscripts", "expected"),
    [
        ({1}, 10),  # numeric order: a string sort would put "10" before "2"
        ({10}, 2),  # a published version is preferred to a later manuscript
        ({1, 2, 10}, 10),  # only manuscripts: the latest
    ],
)
def test_article_metadata_chooses_the_preferred_version(monkeypatch, manuscripts, expected):
    objects = {f"PMC9.{v}": metadata("PMC9", v, manuscript=v in manuscripts) for v in (1, 2, 10)}

    def get(url, params=None):
        if params:  # the bucket listing, deliberately not in numeric order
            return "".join(
                f"<Prefix>{p}/</Prefix>" for p in ("PMC9.10", "PMC9.1", "PMC9.2")
            ).encode()
        return json.dumps(objects[url.rsplit("/", 1)[1].removesuffix(".json")]).encode()

    monkeypatch.setattr(corpus, "_get", get)
    assert corpus.article_metadata("PMC9")["version"] == expected


def test_article_metadata_without_versions(monkeypatch):
    monkeypatch.setattr(corpus, "_get", lambda url, params=None: b"<ListBucketResult/>")
    assert corpus.article_metadata("PMC9") is None


def test_build_corpus_filters_and_counts(monkeypatch):
    cases = {
        "PMC1": metadata("PMC1", 1, pmid=1),  # kept
        "PMC2": metadata("PMC2", 1, pmid=2),  # no abstract in PubMed
        "PMC3": metadata("PMC3", 1, licence="CC BY-NC", pmid=3),
        "PMC4": metadata("PMC4", 1, retracted=True, pmid=4),
        "PMC5": None,
        "PMC6": metadata("PMC6", 1, pmid=6),  # a retraction notice
        "PMC7": metadata("PMC7", 1, pmid=None),  # not indexed in PubMed
    }
    monkeypatch.setattr(corpus, "article_metadata", cases.get)
    monkeypatch.setattr(corpus, "_eutils", lambda endpoint, **params: PUBMED)
    documents, excluded = corpus.build_corpus(list(cases), size=10)
    assert [d.pmcid for d in documents] == ["PMC1"]
    assert documents[0].license == "CC BY" and documents[0].abstract.startswith("BACKGROUND:")
    assert excluded == {
        "no abstract": 1,
        "licence CC BY-NC": 1,
        "retracted": 1,
        "no version in the Cloud Service": 1,
        "publication type Retraction Notice": 1,
        "not indexed in PubMed (no PMID)": 1,
    }


def test_write_and_read_corpus(tmp_path):
    document = corpus.Document("PMC1", "1", None, "T", "A", "J", "2026", "C", "CC0")
    corpus.write_corpus(tmp_path, [document], {"seed": 0})
    assert corpus.read_corpus(tmp_path) == [document]
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"seed": 0}


def test_excluded_types_are_retractions_and_corrections():
    assert "Retracted Publication" in corpus.EXCLUDED_TYPES
    assert "Retraction Notice" in corpus.EXCLUDED_TYPES
    assert "Journal Article" not in corpus.EXCLUDED_TYPES
    with pytest.raises(AttributeError):
        corpus.EXCLUDED_TYPES.add("x")  # frozen
