"""The demo page: escaping, labels and structure."""

import pytest

from biomed_retrieval_jax.corpus import Document
from biomed_retrieval_jax.demo import NOTICE, render_page


def document(
    pmcid="PMC1", title="Title", abstract="Abstract.", types=("Journal Article",), authors=()
):
    return Document(
        pmcid,
        "1",
        None,
        title,
        abstract,
        "Journal",
        "2026",
        "C",
        "CC BY",
        list(types),
        list(authors),
    )


def test_text_from_the_data_is_escaped():
    hostile = document(title='<script>alert("x")</script>', abstract="Lower risk (p<0.05) & more.")
    page = render_page([("q", [hostile])], [], [("Corpus", "<b>not bold</b>")])
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page
    assert "(p&lt;0.05) &amp; more." in page
    assert "&lt;b&gt;not bold&lt;/b&gt;" in page


def test_malformed_pmcids_are_refused():
    with pytest.raises(ValueError, match="not a PMCID"):
        render_page([("q", [document(pmcid='PMC1" onclick="x')])], [], [])


def test_structure_labels_and_excerpts():
    preprint = document(pmcid="PMC2", types=("Journal Article", "Preprint"))
    long = document(pmcid="PMC3", abstract="word " * 200)
    page = render_page([("first query", [document(), preprint]), ("second", [long])], [], [])
    assert NOTICE in page
    assert '<a href="#query-1">first query</a>' in page and 'id="query-2"' in page
    assert page.count("Preprint (not peer-reviewed)") == 1
    assert "https://pmc.ncbi.nlm.nih.gov/articles/PMC2/" in page
    assert "word " * 200 not in page and " …</p>" in page  # the long abstract is shortened


def test_bylines_are_shown_and_escaped():
    page = render_page([("q", [document(authors=["O<Brien K", "Lee S"])])], [], [])
    assert "O&lt;Brien K et al.. Journal, 2026" not in page  # no doubled full stop
    assert "O&lt;Brien K et al. Journal, 2026 · CC BY" in page
