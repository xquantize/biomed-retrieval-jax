"""Build a corpus of openly licensed biomedical abstracts from PubMed Central and PubMed.

Candidates are selected with E-utilities: CC0 or CC BY articles, not embargoed, released in PMC
in a date range. Each candidate's licence and retraction status are checked in its metadata on
PMC's Cloud Service, and its title and abstract are fetched from PubMed. Every document keeps its
licence and citation, for attribution.

Only articles indexed in PubMed are kept. PMC also holds articles that PubMed does not index,
such as supplements of meeting abstracts, which would otherwise dominate a small corpus.
"""

import hashlib
import http.client
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/eutils"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
LICENSES = ("CC0", "CC BY")
QUERY = "(cc0_license[Filter] OR cc_by_license[Filter]) NOT pmc embargo[filter]"
# PubMed publication types of articles that are retracted, or that retract, correct or cast
# doubt on others. Each name was checked against PubMed (November 2026): Retracted Publication
# 35,400 records, Retraction Notice 34,161, Published Erratum 215,223, Expression of Concern
# 4,087.
EXCLUDED_TYPES = frozenset(
    {
        "Retracted Publication",
        "Retraction Notice",
        "Published Erratum",
        "Expression of Concern",
    }
)
BATCH = 200  # articles per EFetch request
# If set, responses are cached in this directory and reused, so that an interrupted build
# resumes where it stopped. PMC's metadata can change (an article may be retracted later),
# so delete the directory to fetch everything afresh.
CACHE_DIRECTORY: Path | None = None


@dataclass(frozen=True)
class Document:
    """An article's title and abstract, with what is needed to cite and attribute it.

    Attributes:
        pmcid: PubMed Central accession ID.
        pmid: PubMed ID.
        doi: Digital Object Identifier, if any.
        title: Article title.
        abstract: Abstract; a structured abstract's sections are labelled, as "LABEL: text".
        journal: Journal title.
        year: Year of publication.
        citation: Journal citation, from PMC.
        license: Creative Commons licence code, "CC0" or "CC BY".
    """

    pmcid: str
    pmid: str
    doi: str | None
    title: str
    abstract: str
    journal: str
    year: str
    citation: str
    license: str


def _get(url: str, params: dict | None = None) -> bytes:
    """Fetch a URL, retrying transient failures; with `CACHE_DIRECTORY` set, cache responses."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    cached = None
    if CACHE_DIRECTORY is not None:
        cached = CACHE_DIRECTORY / hashlib.sha256(url.encode()).hexdigest()
        if cached.exists():
            return cached.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "biomed-retrieval-jax"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
            break
        except urllib.error.HTTPError as error:
            if (error.code < 500 and error.code != 429) or attempt == 4:
                raise
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(2**attempt)
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        partial = cached.with_name(f"{cached.name}.{threading.get_ident()}.tmp")
        partial.write_bytes(body)
        partial.replace(cached)  # atomic: an interruption never leaves a truncated response
    return body


def _eutils(endpoint: str, **params) -> bytes:
    """Call an E-utilities endpoint, within NCBI's rate limits (3 per second, 10 with a key)."""
    params = {"tool": "biomed-retrieval-jax", **params}
    if email := os.environ.get("NCBI_EMAIL"):
        params["email"] = email
    if key := os.environ.get("NCBI_API_KEY"):
        params["api_key"] = key
    time.sleep(0.11 if key else 0.34)
    return _get(f"{EUTILS}/{endpoint}", params)


def search_pmc(start: date, end: date) -> list[str]:
    """Find CC0 and CC BY articles released in PMC between two dates.

    Queries one day at a time, since ESearch returns at most 10,000 results per query.

    Args:
        start: First release date.
        end: Last release date, included.

    Returns:
        The articles' PMCIDs, sorted.

    Raises:
        RuntimeError: If a single day has more than 10,000 articles.
    """
    pmcids = set()
    day = start
    while day <= end:
        term = f"{QUERY} AND {day:%Y/%m/%d}[pmcrdat]"
        response = _eutils("esearch.fcgi", db="pmc", term=term, retmax=10_000, retmode="json")
        result = json.loads(response)["esearchresult"]
        if int(result["count"]) > len(result["idlist"]):
            raise RuntimeError(f"more than 10,000 articles on {day}")
        pmcids.update(f"PMC{i}" for i in result["idlist"])
        day += timedelta(days=1)
    return sorted(pmcids)


def article_metadata(pmcid: str) -> dict | None:
    """Return the metadata of an article's preferred version on PMC's Cloud Service.

    Args:
        pmcid: The article's PMCID.

    Returns:
        The JSON metadata of the highest-numbered published (not author manuscript) version,
        or of the highest-numbered manuscript if there is no published version; None if the
        article has no version in the Cloud Service.
    """
    listing = _get(f"{S3}/", {"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"})
    prefixes = re.findall(r"<Prefix>(PMC\d+\.(\d+))/</Prefix>", listing.decode())
    versions = [
        json.loads(_get(f"{S3}/metadata/{p}.json"))
        for p, _ in sorted(prefixes, key=lambda prefix: int(prefix[1]))
    ]
    published = [v for v in versions if not v["is_manuscript"]]
    return (published or versions)[-1] if versions else None


def _text(element: ET.Element | None) -> str:
    """All text inside an element, including inline markup, with whitespace collapsed."""
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def parse_pubmed(xml: bytes) -> dict[str, dict]:
    """Parse an EFetch response from PubMed.

    Args:
        xml: The response, in PubMed's XML format.

    Returns:
        For each article, by PMID: its title, abstract (structured sections labelled), journal,
        year of publication and publication types.
    """
    records = {}
    for article in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = article.findtext("MedlineCitation/PMID")
        details = article.find("MedlineCitation/Article")
        if pmid is None or details is None:
            continue
        sections = []
        for part in details.findall("Abstract/AbstractText"):
            text, label = _text(part), part.get("Label")
            if text:
                sections.append(f"{label}: {text}" if label else text)
        published = "Journal/JournalIssue/PubDate"
        year = (
            details.findtext(f"{published}/Year")
            or (details.findtext(f"{published}/MedlineDate") or "")[:4]
        )
        records[pmid] = {
            "title": _text(details.find("ArticleTitle")),
            "abstract": " ".join(sections),
            "journal": details.findtext("Journal/Title") or "",
            "year": year,
            "types": [t.text or "" for t in details.findall("PublicationTypeList/PublicationType")],
        }
    return records


def build_corpus(
    pmcids: Sequence[str], size: int, seed: int = 0, workers: int = 16
) -> tuple[list[Document], dict[str, int]]:
    """Select documents from candidate articles, in a seeded random order, until enough pass.

    An article is kept if its preferred version is licensed CC0 or CC BY, is not retracted and
    has a PMID, and its PubMed record has an abstract and none of `EXCLUDED_TYPES`.

    Args:
        pmcids: Candidate articles.
        size: Number of documents wanted.
        seed: Seed of the random order.
        workers: Concurrent requests for metadata.

    Returns:
        The documents (at most `size`, in the random order) and, for the candidates examined,
        how many were excluded for each reason.
    """
    order = random.Random(seed).sample(list(pmcids), len(pmcids))
    documents: list[Document] = []
    excluded: Counter[str] = Counter()
    for start in range(0, len(order), BATCH):
        if len(documents) >= size:
            break
        batch = order[start : start + BATCH]
        with ThreadPoolExecutor(workers) as pool:
            metadata = list(pool.map(article_metadata, batch))
        kept = {}
        for pmcid, m in zip(batch, metadata, strict=True):
            if m is None:
                excluded["no version in the Cloud Service"] += 1
            elif m["license_code"] not in LICENSES:
                excluded[f"licence {m['license_code']}"] += 1
            elif m["is_retracted"]:
                excluded["retracted"] += 1
            elif not m["pmid"]:
                excluded["not indexed in PubMed (no PMID)"] += 1
            else:
                kept[str(m["pmid"])] = (pmcid, m)
        records = {}
        if kept:
            response = _eutils("efetch.fcgi", db="pubmed", id=",".join(kept), retmode="xml")
            records = parse_pubmed(response)
        for pmid, (pmcid, m) in kept.items():
            record = records.get(pmid)
            if record is None:
                excluded["not in PubMed"] += 1
            elif excluded_types := sorted(EXCLUDED_TYPES & set(record["types"])):
                excluded[f"publication type {', '.join(excluded_types)}"] += 1
            elif not record["abstract"]:
                excluded["no abstract"] += 1
            else:
                documents.append(
                    Document(
                        pmcid=pmcid,
                        pmid=pmid,
                        doi=m.get("doi"),
                        title=record["title"] or m["title"],
                        abstract=record["abstract"],
                        journal=record["journal"],
                        year=record["year"],
                        citation=m["citation"],
                        license=m["license_code"],
                    )
                )
    return documents[:size], dict(excluded)


def write_corpus(directory: Path, documents: Sequence[Document], manifest: dict) -> None:
    """Write documents to `documents.jsonl` and a manifest to `manifest.json` in a directory.

    Args:
        directory: Where to write; created if needed.
        documents: The documents.
        manifest: How the corpus was built.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "documents.jsonl", "w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_corpus(directory: Path) -> list[Document]:
    """Read the documents written by `write_corpus`.

    Args:
        directory: The corpus directory.

    Returns:
        The documents, in order.
    """
    with open(directory / "documents.jsonl", encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]
