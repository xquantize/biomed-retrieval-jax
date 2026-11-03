"""Render the static demo: precomputed results for example queries, as one HTML page."""

import html
import re
import textwrap
from collections.abc import Sequence

from biomed_retrieval_jax.corpus import Document

NOTICE = "Research demo, not for clinical use: results must not inform medical decisions."
EXCERPT = 320  # characters of each abstract shown
_PMCID = re.compile(r"PMC\d+")

STYLE = """
:root { color-scheme: light dark; --bg: #fdfdfc; --fg: #1d1d1b; --muted: #5d5d58;
  --accent: #0b5cad; --line: #e3e2de; --warn-bg: #fff4d6; --warn-fg: #5c4400; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecebe7; --muted: #a8a7a1;
  --accent: #7db4f0; --line: #33322f; --warn-bg: #3a2f10; --warn-fg: #f2d68a; } }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1 { font-size: 1.75rem; margin: 0 0 0.5rem; letter-spacing: -0.01em; }
h2 { font-size: 1.2rem; margin: 2.5rem 0 0.75rem; padding-top: 1rem;
  border-top: 1px solid var(--line); }
.notice { background: var(--warn-bg); color: var(--warn-fg); padding: 0.6rem 0.9rem;
  border-radius: 6px; font-weight: 600; }
nav ul { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 0.4rem; }
nav a { display: inline-block; padding: 0.25rem 0.7rem; border: 1px solid var(--line);
  border-radius: 999px; color: var(--fg); text-decoration: none; font-size: 0.9rem; }
nav a:hover { border-color: var(--accent); }
ol { padding-left: 1.4rem; }
li { margin-bottom: 1.1rem; }
a { color: var(--accent); }
a.title { font-weight: 600; text-decoration: none; }
a.title:hover { text-decoration: underline; }
.meta { margin: 0.2rem 0; color: var(--muted); font-size: 0.88rem; }
.preprint { color: var(--warn-fg); background: var(--warn-bg); padding: 0 0.35rem;
  border-radius: 4px; }
.excerpt { margin: 0.2rem 0; font-size: 0.93rem; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1rem;
  font-size: 0.9rem; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
"""


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _result(document: Document) -> str:
    """One result: a linked title, its journal, year and licence, and an excerpt."""
    if not _PMCID.fullmatch(document.pmcid):
        raise ValueError(f"not a PMCID: {document.pmcid!r}")
    source = f"{_escape(document.journal)}, {_escape(document.year)}"
    if document.byline:
        # A byline may end with "et al.": end it with one full stop, not two.
        source = f"{_escape(document.byline.removesuffix('.'))}. {source}"
    meta = f"{source} · {_escape(document.license)}"
    if document.preprint:
        meta += ' · <span class="preprint">Preprint (not peer-reviewed)</span>'
    excerpt = textwrap.shorten(document.abstract, EXCERPT, placeholder=" …")
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{document.pmcid}/"
    return (
        f'<li><a class="title" href="{url}">{_escape(document.title)}</a>'
        f'<p class="meta">{meta}</p><p class="excerpt">{_escape(excerpt)}</p></li>'
    )


def render_page(
    results: Sequence[tuple[str, Sequence[Document]]],
    about: Sequence[str],
    provenance: Sequence[tuple[str, str]],
) -> str:
    """Render the demo page.

    Args:
        results: Each example query and its documents, in rank order. Scores are not shown:
            they rank the results of one query and are not comparable across queries.
        about: Paragraphs explaining the results, as trusted HTML (not escaped).
        provenance: Labels and values describing the corpus and the models, as plain text.

    Returns:
        The page, as HTML.

    Raises:
        ValueError: If a document's PMCID is malformed.
    """
    links = "".join(
        f'<li><a href="#query-{i}">{_escape(query)}</a></li>'
        for i, (query, _) in enumerate(results, start=1)
    )
    sections = "".join(
        f'<section id="query-{i}"><h2>“{_escape(query)}”</h2>'
        f"<ol>{''.join(_result(d) for d in documents)}</ol></section>"
        for i, (query, documents) in enumerate(results, start=1)
    )
    paragraphs = "".join(f"<p>{paragraph}</p>" for paragraph in about)
    details = "".join(f"<dt>{_escape(k)}</dt><dd>{_escape(v)}</dd>" for k, v in provenance)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Biomedical literature search: a verified JAX demo</title>"
        f"<style>{STYLE}</style></head><body><main>"
        "<header><h1>Biomedical literature search</h1>"
        f'<p class="notice">{_escape(NOTICE)}</p>{paragraphs}</header>'
        f'<nav aria-label="Example queries"><ul>{links}</ul></nav>{sections}'
        f'<section id="about"><h2>About these results</h2><dl>{details}</dl></section>'
        "</main></body></html>\n"
    )
