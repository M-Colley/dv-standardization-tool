"""Systematic search of research-data repositories for open HCI user-study datasets.

Stage 1  DataCite REST API, which indexes Zenodo, OSF, figshare, Dryad, 4TU.ResearchData,
         Dataverse and Mendeley Data: a fixed list of instrument and domain queries,
         resource type Dataset.
Stage 2  Metadata screen: one record per concept (latest version); an open licence; a linked
         peer-reviewed paper (Crossref type journal-article or proceedings-article); not
         already a source of this project.
Stage 3  File listing through each repository's API (``opendv.sources``): tabular files and sizes.

Only metadata and file listings are read; no data file is downloaded. Every record is kept
in ``hits.csv`` with the stage it stopped at and why, so the counts of each stage can be
reported. Responses are cached under ``data/cache/candidate_search`` (gitignored), so a
rerun with ``--offline`` reproduces the screen from the same responses.

    python -m analyses.candidate_search.search_open_data --out analyses/candidate_search
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import httpx
import pandas as pd

from opendv.http_utils import HTTP_USER_AGENT

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "candidate_search"
DATACITE = "https://api.datacite.org/dois"
CROSSREF = "https://api.crossref.org/works"

# Instruments the DV schema covers, and the study domains of the development set.
QUERIES: dict[str, str] = {
    "nasa_tlx": '"NASA-TLX" OR "NASA TLX" OR "task load index" OR "raw TLX"',
    "sus": '"system usability scale"',
    "ueq": '"user experience questionnaire" OR "UEQ-S"',
    "attrakdiff": "AttrakDiff",
    "ssq": '"simulator sickness questionnaire" OR "simulator sickness"',
    "fms": '"fast motion sickness"',
    "trust_automation": '"trust in automation" OR "trust in automated"',
    "perceived_safety": '"perceived safety"',
    "van_der_laan": '"van der laan" AND (acceptance OR usefulness OR satisfying)',
    "situation_awareness": '"situation awareness rating technique" OR SAGAT'
                           ' OR ("situation awareness" AND questionnaire)',
    "presence": '"igroup presence" OR "presence questionnaire"',
    "pxi": '"player experience inventory"',
    "tam_utaut": '"technology acceptance model" OR UTAUT',
    "ati": '"affinity for technology interaction"',
    "ehmi": 'eHMI OR "external human-machine interface" OR "external human machine interface"',
    "automated_driving": '("automated driving" OR "automated vehicle" OR "autonomous vehicle" OR "self-driving")'
                         ' AND (participants OR questionnaire OR "user study")',
    "takeover": '("take-over" OR takeover) AND (driving OR driver) AND participants',
    "driving_simulator": '"driving simulator" AND (participants OR questionnaire)',
    "vr_user_study": '("virtual reality" OR "mixed reality" OR "augmented reality")'
                     ' AND ("user study" OR "within-subjects" OR "within-subject" OR questionnaire)',
    "hci_venue": '("Human Factors in Computing Systems" OR AutomotiveUI OR "Automotive User Interfaces"'
                 ' OR "CHI 20*" OR UIST OR "Human-Computer Interaction") AND (participants OR questionnaire)',
    "hri": '"human-robot interaction" AND (questionnaire OR participants)',
}

DATACITE_FIELDS = ",".join([
    "doi", "titles", "publisher", "publicationYear", "rightsList", "relatedIdentifiers",
    "descriptions", "url", "types", "registered", "version", "subjects",
])

# DOI prefixes of the data repositories (a related DOI with one of these is not a paper).
REPOSITORY_PREFIXES = (
    "10.5281/", "10.17605/", "10.6084/", "10.5061/", "10.4121/", "10.7910/", "10.17632/",
    "10.34894/", "10.25835/", "10.18419/", "10.11588/", "10.23668/", "10.48420/", "10.15131/",
    "10.5878/", "10.18710/", "10.21227/",
)
PAPER_RELATIONS = {
    "IsSupplementTo", "IsReferencedBy", "IsCitedBy", "IsDocumentedBy", "IsDescribedBy",
    "IsSourceOf", "IsDerivedFrom", "References", "Cites", "Documents", "Describes",
}
PAPER_TYPES = {"journal-article", "proceedings-article"}

OPEN_LICENCE = re.compile(
    r"cc[-_ ]?by(?![-_ ]?n[cd])(?:[-_ ]?sa)?|cc0|creativecommons\.org/(?:publicdomain|licenses/by(?:-sa)?/)|"
    r"\bmit\b|\bbsd\b|apache|odc[-_ ]?by|pddl|public domain", re.IGNORECASE,
)
RESTRICTED_LICENCE = re.compile(r"\bn[cd]\b|-n[cd]\b|non[- ]?commercial|no[- ]?deriv", re.IGNORECASE)

TABULAR = {".csv", ".tsv", ".xlsx", ".xls", ".ods", ".sav", ".dta", ".txt", ".json", ".parquet", ".feather"}
R_DATA = {".rds", ".rdata", ".rda"}
ARCHIVES = {".zip", ".7z", ".tar", ".gz", ".tgz", ".rar"}
UNSAFE = {".pkl", ".pickle", ".npy", ".npz", ".joblib"}

# Instrument and design cues looked for in titles, descriptions and file names.
CUES: dict[str, str] = {
    "nasa_tlx": r"nasa[- _]?tlx|task load index|\btlx\b|mental demand",
    "sus": r"system usability scale|\bsus\b",
    "ueq": r"user experience questionnaire|\bueq",
    "attrakdiff": r"attrakdiff",
    "ssq": r"simulator sickness|\bssq\b|\bfms\b|motion sickness",
    "trust": r"trust",
    "perceived_safety": r"perceived safety|feeling (?:of )?safe",
    "van_der_laan": r"van der laan",
    "situation_awareness": r"situation(?:al)? awareness|\bsart\b",
    "presence": r"presence questionnaire|igroup|\bipq\b",
    "pxi": r"player experience inventory|\bpxi\b",
    "acceptance": r"technology acceptance|utaut|\btam\b|acceptance",
    "performance": r"reaction time|response time|completion time|error rate|accuracy|takeover time|crossing",
    "repeated_measures": r"within[- ]subjects?|repeated[- ]measures?|conditions",
}

SESSION: httpx.Client | None = None


def _client() -> httpx.Client:
    global SESSION
    if SESSION is None:
        SESSION = httpx.Client(headers={"User-Agent": f"{HTTP_USER_AGENT} (open-data search)",
                                        "Accept": "application/json"},
                               timeout=60.0, follow_redirects=True)
    return SESSION


def cached_json(url: str, offline: bool = False, pause: float = 0.2) -> Any:
    """GET a JSON document, cached by URL hash; ``None`` when the host refuses or lacks it."""
    key = hashlib.sha256(url.encode()).hexdigest()[:32]
    path = CACHE / f"{key}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["body"]
    if offline:
        return None
    body: Any = None
    for attempt in range(4):
        try:
            response = _client().get(url)
        except httpx.HTTPError:
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt * 2)
            continue
        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError:
                body = None
        break
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "fetched": date.today().isoformat(), "body": body}),
                    encoding="utf-8")
    time.sleep(pause)
    return body


# ── Stage 1: DataCite ───────────────────────────────────────────────────────

def datacite_search(query: str, offline: bool = False) -> list[dict[str, Any]]:
    params = {"query": query, "resource-type-id": "dataset", "page[size]": 1000,
              "page[cursor]": 1, "fields[dois]": DATACITE_FIELDS}
    url: str | None = f"{DATACITE}?{urlencode(params, safe='[]')}"
    records: list[dict[str, Any]] = []
    while url:
        body = cached_json(url, offline)
        if not body:
            break
        records.extend(body.get("data") or [])
        url = (body.get("links") or {}).get("next")
        if not (body.get("data") or []):
            break
    return records


SHARE = "https://share.osf.io/api/v3/index-card-search"
# OSF projects without a DOI are missing from DataCite; OSF's own index takes plain phrases.
SHARE_PHRASES: dict[str, list[str]] = {
    "nasa_tlx": ['"NASA-TLX"', '"NASA TLX"', '"task load index"'],
    "sus": ['"system usability scale"'],
    "ueq": ['"user experience questionnaire"', '"UEQ-S"'],
    "attrakdiff": ["AttrakDiff"],
    "ssq": ['"simulator sickness"'],
    "trust_automation": ['"trust in automation"'],
    "perceived_safety": ['"perceived safety"'],
    "situation_awareness": ['"situation awareness"'],
    "presence": ['"presence questionnaire"'],
    "pxi": ['"player experience inventory"'],
    "tam_utaut": ['"technology acceptance model"', "UTAUT"],
    "ati": ['"affinity for technology interaction"'],
    "ehmi": ["eHMI", '"external human-machine interface"'],
    "automated_driving": ['"automated vehicle"', '"autonomous vehicle"', '"automated driving"',
                          '"self-driving"'],
    "takeover": ['"takeover request"', '"take-over request"'],
    "driving_simulator": ['"driving simulator"'],
    "hri": ['"human-robot interaction"'],
}
DOI_IN_TEXT = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>,;)\]]+)", re.IGNORECASE)


def share_search(phrase: str, offline: bool = False, limit: int = 3000) -> list[dict[str, Any]]:
    params = {"cardSearchText": phrase, "cardSearchFilter[resourceType]": "Project,ProjectComponent",
              "page[size]": 100}
    url: str | None = f"{SHARE}?{urlencode(params, safe='[]')}"
    cards: list[dict[str, Any]] = []
    while url and len(cards) < limit:
        body = cached_json(url, offline)
        if not body or not body.get("data"):
            break
        cards.extend(body["data"])
        url = (body.get("links") or {}).get("next")
    return cards


def share_attrs(card: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """A SHARE index card as a DataCite-shaped record keyed by the OSF DOI form."""
    match = re.search(r"osf\.io/([a-z0-9]{5})\b", str(card.get("@id", "")).lower())
    if not match:
        return None

    def values(key: str) -> list[str]:
        return [str(x.get("@value", "")) for x in card.get(key) or [] if isinstance(x, dict)]

    description = " ".join(values("description"))
    rights = [{"rightsUri": r.get("@id") or " ".join(str(n.get("@value", "")) for n in r.get("name") or [])}
              for r in card.get("rights") or [] if isinstance(r, dict)]
    related = [{"relationType": "IsReferencedBy", "relatedIdentifierType": "DOI",
                "relatedIdentifier": doi.rstrip(".")} for doi in DOI_IN_TEXT.findall(description)]
    created = (values("dateCreated") or [""])[0]
    attrs = {
        "titles": [{"title": t} for t in values("title")],
        "descriptions": [{"description": description}],
        "rightsList": rights,
        "relatedIdentifiers": related,
        "url": f"https://osf.io/{match.group(1)}/",
        "publisher": "OSF",
        "publicationYear": created[:4] or None,
        "registered": (values("dateModified") or [created])[0],
    }
    return f"10.17605/osf.io/{match.group(1)}", attrs


# ── Stage 2: metadata screen ────────────────────────────────────────────────

def _text(attrs: dict[str, Any]) -> str:
    titles = " ".join(t.get("title", "") for t in attrs.get("titles") or [])
    descriptions = " ".join(d.get("description", "") or "" for d in attrs.get("descriptions") or [])
    return f"{titles}\n{descriptions}"


def licence_of(attrs: dict[str, Any]) -> tuple[str, str]:
    """(label, status) with status open / restricted / unknown."""
    labels = []
    for right in attrs.get("rightsList") or []:
        labels.append(right.get("rightsIdentifier") or right.get("rightsUri") or right.get("rights") or "")
    label = "; ".join(x for x in labels if x)
    if not label:
        return "", "unknown"
    if RESTRICTED_LICENCE.search(label):
        return label, "restricted"
    if OPEN_LICENCE.search(label):
        return label, "open"
    return label, "unknown"


def concept_key(doi: str, attrs: dict[str, Any]) -> str:
    key = doi.lower()
    for rel in attrs.get("relatedIdentifiers") or []:
        if rel.get("relationType") == "IsVersionOf" and rel.get("relatedIdentifierType") == "DOI":
            key = str(rel.get("relatedIdentifier")).lower()
            break
    # figshare, 4TU and Mendeley Data register every version as DOI + ".vN" / ".N"
    # (4TU sometimes links the base DOI as a version of ".v1").
    key = re.sub(r"\.v\d+$", "", key)
    return re.sub(r"^(10\.17632/[a-z0-9]+)\.\d+$", r"\1", key)


def linked_paper_dois(attrs: dict[str, Any]) -> list[str]:
    out = []
    for rel in attrs.get("relatedIdentifiers") or []:
        if rel.get("relationType") not in PAPER_RELATIONS:
            continue
        value = str(rel.get("relatedIdentifier") or "").strip()
        if rel.get("relatedIdentifierType") == "URL":
            match = re.search(r"doi\.org/(10\.\S+)", value)
            if not match:
                continue
            value = match.group(1)
        elif rel.get("relatedIdentifierType") != "DOI":
            continue
        value = value.lower().removeprefix("https://doi.org/").removeprefix("doi:")
        if value.startswith("10.") and not value.startswith(REPOSITORY_PREFIXES):
            out.append(value)
    return list(dict.fromkeys(out))


def crossref_work(doi: str, offline: bool) -> dict[str, Any] | None:
    body = cached_json(f"{CROSSREF}/{quote(doi, safe='/')}", offline)
    return (body or {}).get("message")


def _clean_title(title: str) -> str:
    title = re.sub(r"^(?:data(?:set)?|replication data|supplementary (?:data|material)|data and code|"
                   r"code and data|study data|raw data)\s+(?:for|from|of|to)\s*[:\-]?\s*", "", title,
                   flags=re.IGNORECASE)
    return title.strip(" \"'“”:")


def crossref_by_title(title: str, offline: bool) -> dict[str, Any] | None:
    """The best Crossref paper for a dataset title, when the titles nearly match."""
    cleaned = _clean_title(title)
    if len(cleaned) < 25:
        return None
    params = {"query.bibliographic": cleaned, "rows": 3,
              "filter": "type:journal-article,type:proceedings-article"}
    url = f"{CROSSREF}?{urlencode(params)}"
    body = cached_json(url, offline)
    best, best_ratio = None, 0.0
    for item in ((body or {}).get("message") or {}).get("items") or []:
        candidate = " ".join(item.get("title") or [])
        ratio = difflib.SequenceMatcher(None, cleaned.lower(), candidate.lower()).ratio()
        if ratio > best_ratio:
            best, best_ratio = item, ratio
    return best if best_ratio >= 0.85 else None


def existing_sources() -> set[str]:
    """DOIs, record ids and project ids of every source already in the project."""
    keys: set[str] = set()
    files = [ROOT / "sources_manifest_no4tu.yaml", ROOT / "analyses/development_sources/sources.yaml",
             ROOT / "analyses/confirmatory/sources.yaml"]
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        keys.update(re.findall(r"10\.\d{4,9}/[^\s\"',)\]}]+", text))
        keys.update(f"zenodo.{m}" for m in re.findall(r"zenodo\.org/(?:api/)?records?/(\d+)", text))
        keys.update(f"osf.{m}" for m in re.findall(r"osf\.io/([a-z0-9]{5})\b", text))
    return {k.rstrip(".") for k in keys}


def record_keys(doi: str, attrs: dict[str, Any], papers: Iterable[str]) -> set[str]:
    keys = {doi.lower(), *papers}
    if m := re.search(r"zenodo\.(\d+)", doi.lower()):
        keys.add(f"zenodo.{m.group(1)}")
    if m := re.search(r"osf\.io/([a-z0-9]{5})", doi.lower()):
        keys.add(f"osf.{m.group(1)}")
    url = str(attrs.get("url") or "").lower()
    if m := re.search(r"zenodo\.org/records?/(\d+)", url):
        keys.add(f"zenodo.{m.group(1)}")
    if m := re.search(r"osf\.io/([a-z0-9]{5})", url):
        keys.add(f"osf.{m.group(1)}")
    return keys


# ── Stage 3: file listing ───────────────────────────────────────────────────

def list_files(doi: str, attrs: dict[str, Any], offline: bool) -> tuple[list[tuple[str, int | None]], str]:
    """(files as (path, size), note) through the repository's API."""
    records, note = file_records(doi, attrs, offline)
    return [(r["path"], r["size"]) for r in records], note


def _record(path: Any, size: Any, url: Any, **checksums: Any) -> dict[str, Any]:
    """One file: path, size, download URL and the checksums the repository publishes."""
    record: dict[str, Any] = {"path": str(path), "size": size, "url": url}
    record.update({kind: str(value).lower() for kind, value in checksums.items() if value})
    return record


def file_records(doi: str, attrs: dict[str, Any], offline: bool) -> tuple[list[dict[str, Any]], str]:
    """(files as dicts with path, size, url and md5/sha256, note) through the repository's API."""
    doi_l = doi.lower()
    url = str(attrs.get("url") or "")
    if m := re.match(r"10\.5281/zenodo\.(\d+)", doi_l):
        record_id = m.group(1)
        if "/doi/" in url:  # a concept DOI: list the latest version
            search = cached_json(f"https://zenodo.org/api/records?q=conceptrecid:{record_id}&size=1", offline)
            hits = (((search or {}).get("hits") or {}).get("hits")) or []
            record_id = str(hits[0]["id"]) if hits else record_id
        body = cached_json(f"https://zenodo.org/api/records/{record_id}", offline)
        if not body:
            return [], "zenodo record unavailable"
        entries = body.get("files") or []
        if isinstance(entries, dict):
            entries = list((entries.get("entries") or {}).values())
        access = (body.get("metadata") or {}).get("access_right") or ""
        files = []
        for e in entries:
            algorithm, _, digest = str(e.get("checksum") or "").partition(":")
            files.append(_record(e.get("key"), e.get("size"), (e.get("links") or {}).get("self"),
                                 **{algorithm: digest} if algorithm in ("md5", "sha256") else {}))
        return files, "" if files else f"no public files ({access})"
    figshare_id = re.match(r"10\.6084/m9\.figshare\.(\d+)", doi_l)
    if figshare_id is None and not doi_l.startswith("10.4121/"):
        # Institutional figshare portals (Loughborough, Monash, UCL, Open University, ...).
        figshare_id = re.search(r"/articles/[^/]+/[^/]+/(\d+)(?:/\d+)?/?$", url)
    if figshare_id:
        body = cached_json(f"https://api.figshare.com/v2/articles/{figshare_id.group(1)}", offline)
        files = [_record(e.get("name"), e.get("size"), e.get("download_url"),
                         md5=e.get("computed_md5") or e.get("supplied_md5"))
                 for e in (body or {}).get("files") or [] if not e.get("is_link_only")]
        return files, "" if files else ("no public files" if body else "figshare article unavailable")
    if doi_l.startswith("10.4121/"):
        m = (re.search(r"data\.4tu\.nl/datasets/([0-9a-f-]{36})", url)
             or re.search(r"data\.4tu\.nl/articles/[^/]+/(\d+)", url))
        if not m:
            return [], "4TU id not found"
        body = cached_json(f"https://data.4tu.nl/v2/articles/{m.group(1)}/files", offline)
        files = [_record(e.get("name"), e.get("size"), e.get("download_url"),
                         md5=e.get("computed_md5") or e.get("supplied_md5"))
                 for e in body or [] if not e.get("is_link_only")]
        return files, "" if body else "4TU listing unavailable"
    if doi_l.startswith("10.5061/dryad") or "datadryad.org" in url:
        body = cached_json(f"https://datadryad.org/api/v2/datasets/{quote('doi:' + doi, safe='')}", offline)
        version = (((body or {}).get("_links") or {}).get("stash:version") or {}).get("href")
        if not version:
            return [], "dryad dataset unavailable"
        listing = cached_json(f"https://datadryad.org{version}/files", offline)
        files = []
        for f in ((listing or {}).get("_embedded") or {}).get("stash:files") or []:
            download = ((f.get("_links") or {}).get("stash:download") or {}).get("href")
            kind = {"sha-256": "sha256", "md5": "md5"}.get(str(f.get("digestType")).lower(), "")
            files.append(_record(f.get("path"), f.get("size"), download and f"https://datadryad.org{download}",
                                 **{kind: f.get("digest")} if kind else {}))
        return files, "" if listing else "dryad listing refused"
    if m := re.match(r"10\.17605/osf\.io/([a-z0-9]{5})", doi_l):
        return osf_files(m.group(1), offline)
    if m := re.search(r"data\.mendeley\.com/datasets/([a-z0-9]+)/(\d+)", url):
        body = cached_json(f"https://data.mendeley.com/public-api/datasets/{m.group(1)}/files"
                           f"?folder_id=root&version={m.group(2)}", offline)
        files = [_record(e.get("filename"), (e.get("content_details") or {}).get("size"),
                         (e.get("content_details") or {}).get("download_url"),
                         sha256=(e.get("content_details") or {}).get("sha256_hash"))
                 for e in body or []]
        return files, "" if files else "mendeley listing unavailable"
    if "dataverse" in url or "persistentId=" in url or doi_l.startswith("10.7910/"):
        server = re.match(r"(https?://[^/]+)", url)
        if not server:
            return [], "dataverse server unknown"
        body = cached_json(f"{server.group(1)}/api/datasets/:persistentId/?persistentId=doi:{doi}", offline)
        version = ((body or {}).get("data") or {}).get("latestVersion") or {}
        files = []
        for entry in version.get("files") or []:
            data_file = entry.get("dataFile") or {}
            name = data_file.get("originalFileName") or data_file.get("filename") or entry.get("label")
            folder = entry.get("directoryLabel")
            # Ingested tables are served as .tab unless the original format is asked for; the
            # published checksum is that of the original file.
            original = "?format=original" if data_file.get("tabularData") else ""
            checksum = data_file.get("checksum") or {}
            kind = {"MD5": "md5", "SHA-256": "sha256"}.get(str(checksum.get("type")), "")
            record = _record(f"{folder}/{name}" if folder else name,
                             data_file.get("originalFileSize") or data_file.get("filesize"),
                             f"{server.group(1)}/api/access/datafile/{data_file.get('id')}{original}",
                             **{kind: checksum.get("value")} if kind else {})
            if entry.get("restricted"):
                record["restricted"] = True  # listed, but not downloadable without a request
            files.append(record)
        return files, "" if body else "dataverse dataset unavailable"
    return [], "repository listing not supported"


def osf_files(node: str, offline: bool, depth: int = 0) -> tuple[list[dict[str, Any]], str]:
    """Files of an OSF node and its components (OSF Storage only)."""
    base = "https://api.osf.io/v2"
    files: list[dict[str, Any]] = []
    queue = [f"{base}/nodes/{node}/files/osfstorage/?page[size]=100"]
    seen = 0
    while queue and seen < 60:
        page: str | None = queue.pop(0)
        while page and seen < 60:
            seen += 1
            body = cached_json(page, offline)
            if not body:
                if not files and depth == 0:
                    # Registrations live under /registrations/.
                    alt = page.replace("/nodes/", "/registrations/")
                    body = cached_json(alt, offline) if alt != page else None
                if not body:
                    break
            for item in body.get("data") or []:
                attrs = item.get("attributes") or {}
                if attrs.get("kind") == "file":
                    hashes = (attrs.get("extra") or {}).get("hashes") or {}
                    files.append(_record(str(attrs.get("materialized_path") or attrs.get("name")).lstrip("/"),
                                         attrs.get("size"), (item.get("links") or {}).get("download"),
                                         sha256=hashes.get("sha256"), md5=hashes.get("md5")))
                elif attrs.get("kind") == "folder":
                    href = (((item.get("relationships") or {}).get("files") or {}).get("links") or {}
                            ).get("related", {}).get("href")
                    if href:
                        queue.append(href + ("&" if "?" in href else "?") + "page[size]=100")
            page = (body.get("links") or {}).get("next")
    if depth == 0:
        children = cached_json(f"{base}/nodes/{node}/children/?page[size]=50", offline)
        for child in (children or {}).get("data") or []:
            child_files, _ = osf_files(str(child.get("id")), offline, depth + 1)
            files.extend(child_files)
    return files, "" if files else "osf listing empty or unavailable"


def file_profile(files: list[tuple[str, int | None]]) -> dict[str, Any]:
    def ext(name: str) -> str:
        lower = name.lower()
        for double in (".tar.gz",):
            if lower.endswith(double):
                return double
        return Path(lower).suffix

    tabular = [(n, s) for n, s in files if ext(n) in TABULAR]
    return {
        "n_files": len(files),
        "total_bytes": int(sum(s or 0 for _, s in files)),
        "tabular_files": "; ".join(f"{n} ({s})" for n, s in tabular[:25]),
        "n_tabular": len(tabular),
        "tabular_bytes": int(sum(s or 0 for _, s in tabular)),
        "has_archive": any(ext(n) in ARCHIVES or n.lower().endswith(".tar.gz") for n, _ in files),
        "has_r_data": any(ext(n) in R_DATA for n, _ in files),
        "has_unsafe": any(ext(n) in UNSAFE for n, _ in files),
        "file_names": " ".join(n for n, _ in files[:200]),
    }


def cues(text: str) -> str:
    return ",".join(name for name, pattern in CUES.items() if re.search(pattern, text, re.IGNORECASE))


# ── Driver ──────────────────────────────────────────────────────────────────

def run(out_dir: Path, offline: bool = False, title_match: bool = True, share: bool = True,
        workers: int = 8) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    query_counts: dict[str, int] = {}
    by_doi: dict[str, dict[str, Any]] = {}
    matched: dict[str, list[str]] = {}
    for name, query in QUERIES.items():
        records = datacite_search(query, offline)
        query_counts[name] = len(records)
        for record in records:
            doi = str(record.get("id") or record.get("attributes", {}).get("doi")).lower()
            by_doi[doi] = record.get("attributes") or {}
            matched.setdefault(doi, []).append(name)
        print(f"{name:22s} {len(records):6d} records", flush=True)

    share_counts: dict[str, int] = {}
    if share:
        for name, phrases in SHARE_PHRASES.items():
            n = 0
            for phrase in phrases:
                for card in share_search(phrase, offline):
                    converted = share_attrs(card)
                    if converted is None:
                        continue
                    doi, attrs = converted
                    n += 1
                    if doi not in by_doi:
                        by_doi[doi] = attrs
                    elif not by_doi[doi].get("rightsList") and attrs["rightsList"]:
                        by_doi[doi]["rightsList"] = attrs["rightsList"]
                    matched.setdefault(doi, []).append(name)
            share_counts[name] = n
            print(f"OSF {name:18s} {n:6d} projects", flush=True)

    # One record per concept: the latest registered version.
    concepts: dict[str, tuple[str, dict[str, Any]]] = {}
    for doi, attrs in by_doi.items():
        key = concept_key(doi, attrs)
        current = concepts.get(key)
        if current is None or str(attrs.get("registered") or "") > str(current[1].get("registered") or ""):
            concepts[key] = (doi, attrs)
    known = existing_sources()

    concept_queries: dict[str, set[str]] = {}
    for doi, attrs in by_doi.items():
        concept_queries.setdefault(concept_key(doi, attrs), set()).update(matched[doi])

    rows = []
    for key, (doi, attrs) in concepts.items():
        queries = sorted(concept_queries[key])
        title = " ".join(t.get("title", "") for t in attrs.get("titles") or [])
        licence, licence_status = licence_of(attrs)
        row: dict[str, Any] = {
            "doi": doi, "concept": key, "title": title, "publisher": attrs.get("publisher"),
            "year": attrs.get("publicationYear"), "url": attrs.get("url"), "queries": ",".join(queries),
            "n_queries": len(queries), "licence": licence, "licence_status": licence_status,
        }
        rows.append((row, attrs))

    out = []
    for row, attrs in rows:
        doi = row["doi"]
        papers = linked_paper_dois(attrs)
        row["stage"], row["reason"] = "screen", ""
        if record_keys(doi, attrs, papers) & known or row["concept"] in known:
            row["stage"], row["reason"] = "excluded", "already a source"
        elif row["licence_status"] == "restricted":
            row["stage"], row["reason"] = "excluded", "licence restricts reuse (NC/ND)"
        elif row["licence_status"] == "unknown":
            row["stage"], row["reason"] = "excluded", "no open licence stated"
        out.append((row, attrs, papers))

    def screen(item: tuple[dict[str, Any], dict[str, Any], list[str]]) -> dict[str, Any]:
        row, attrs, papers = item
        if row["stage"] == "excluded":
            return row
        paper, how = None, ""
        for candidate in papers:
            work = crossref_work(candidate, offline)
            if work and work.get("type") in PAPER_TYPES:
                paper, how = work, "linked"
                break
        if paper is None and title_match:
            paper = crossref_by_title(row["title"], offline)
            how = "title match" if paper else ""
        if paper is None:
            row["stage"], row["reason"] = "excluded", "no peer-reviewed paper found"
            return row
        row.update({
            "paper_doi": str(paper.get("DOI", "")).lower(), "paper_link": how,
            "paper_title": " ".join(paper.get("title") or []),
            "venue": " ".join(paper.get("container-title") or []),
            "paper_year": ((paper.get("issued") or {}).get("date-parts") or [[None]])[0][0],
            "paper_type": paper.get("type"),
        })
        if row["paper_doi"] in known:
            row["stage"], row["reason"] = "excluded", "already a source"
            return row
        files, note = list_files(row["doi"], attrs, offline)
        row.update(file_profile(files))
        row["listing_note"] = note
        text = f"{_text(attrs)}\n{row['paper_title']}\n{row.get('file_names', '')}"
        row["cues"] = cues(text)
        row["description"] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", _text(attrs)))[:1500]
        if not files:
            row["stage"], row["reason"] = "excluded", f"no file listing ({note})"
        elif row["n_tabular"] == 0 and not row["has_archive"] and not row["has_r_data"]:
            row["stage"], row["reason"] = "excluded", "no tabular file or archive"
        else:
            row["stage"], row["reason"] = "candidate", ""
        return row

    pending = sum(row["stage"] != "excluded" for row, _, _ in out)
    print(f"{len(out)} concepts, {pending} to look up", flush=True)
    final = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(screen, out), 1):
            final.append(row)
            if i % 250 == 0:
                print(f"  screened {i}/{len(out)}", flush=True)

    frame = pd.DataFrame(final).drop(columns=["file_names"], errors="ignore")
    # One dataset per paper and repository (some records carry two DOIs, e.g. 4TU uuid and number).
    ordered = frame[frame["stage"] == "candidate"].sort_values("n_files", ascending=False)
    duplicate = []
    for _, group in ordered.groupby(["paper_doi", "publisher"]):
        kept: list[str] = []
        for index, title in group["title"].items():
            if any(difflib.SequenceMatcher(None, title.lower(), k.lower()).ratio() >= 0.9 for k in kept):
                duplicate.append(index)
            else:
                kept.append(title)
    frame.loc[duplicate, "stage"] = "excluded"
    frame.loc[duplicate, "reason"] = "duplicate record of the same paper"
    frame = frame.sort_values(["stage", "n_queries"], ascending=[True, False])
    frame.to_csv(out_dir / "hits.csv", index=False)
    log = {
        "date": date.today().isoformat(),
        "sources": ["DataCite REST API, resource type Dataset",
                    "OSF SHARE index-card search, resource types Project and ProjectComponent"],
        "queries": QUERIES,
        "records_per_query": query_counts,
        "osf_phrases": SHARE_PHRASES if share else {},
        "osf_projects_per_query": share_counts,
        "unique_dois": len(by_doi),
        "unique_concepts": len(concepts),
        "stage_counts": frame.groupby(["stage", "reason"]).size().rename("n").reset_index().to_dict("records"),
    }
    (out_dir / "search_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "analyses" / "candidate_search")
    parser.add_argument("--offline", action="store_true", help="use cached responses only")
    parser.add_argument("--no-title-match", action="store_true")
    parser.add_argument("--no-osf", action="store_true", help="skip OSF's own search index")
    args = parser.parse_args()
    frame = run(args.out, args.offline, not args.no_title_match, not args.no_osf)
    print(frame.groupby(["stage", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
