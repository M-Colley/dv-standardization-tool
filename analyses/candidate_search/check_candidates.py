"""Gather the documentation of the 79 "to check" datasets and the five datasets dropped from the first held-out test.

For the second held-out test (``docs/preregistration_heldout_relations_2.md``) every dataset the
2026-11-05 search marked ``check`` is checked from metadata, READMEs and papers only. This script
collects the repository-side part of that evidence; no participant data are opened:

- the repository's file listing (``search_open_data.file_records``);
- the members of every zip archive, from its central directory (``build_sources.list_zip``,
  HTTP range requests; nothing else of the archive is read);
- the text of documentation files: READMEs, codebooks, data dictionaries, variable lists,
  metadata and questionnaires, at most 30 kB (PDFs at most 2 MB), top-level or inside zip
  archives. Text files are decoded, ``.docx`` and ``.pdf`` reduced to their text, ``.xlsx``
  codebooks read with pandas. Tables of participant data are never fetched.

Everything is cached under ``data/cache/candidate_search`` (gitignored); ``--offline`` reruns
from the cache. The dossier, one entry per dataset, is written as JSON to the cache directory.
The decisions themselves are in ``check_decisions.csv``, written by hand from the dossier and
the papers.

    python -m analyses.candidate_search.check_candidates [--offline]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from analyses.candidate_search.build_sources import list_zip
from analyses.candidate_search.search_open_data import CACHE, _client, file_records

HERE = Path(__file__).resolve().parent
DECISIONS = HERE / "screening_decisions.csv"
HITS = HERE / "hits.csv"
SPLIT = HERE / "heldout_split.csv"
DOSSIER = CACHE / "check_dossier.json"

# Held-out datasets of the first test dropped before download because their tables exist only as .mat/.edf.
FIRST_TEST_DROPS = {
    "10.5281/zenodo.20627050": "driver_trust_index_2026",
    "10.4121/6dfe7bd5-16f7-4e8d-9ab7-f8203b902310.v1": "ehmi_flash_text_2023",
    "10.4121/858e4b36-895b-4c84-928a-d1cdcf0920d6.v1": "ehmi_gimmick_2022",
    "10.4121/12682826": "ehmi_location_2019",
    "10.4121/18bb5d48-c1e5-4d50-ad71-d9010e4494d9.v2": "partial_automation_ui_2024",
}

DOC_NAME = re.compile(r"read[ _-]?me|code[ _-]?book|dictionar|data[ _-]?description|variables|metadata|"
                      r"questionn|dataset[ _-]?details|description", re.IGNORECASE)
TEXT = {".txt", ".md", ".rtf", ".csv", ".tsv", ".json", ".html"}
MAX_TEXT = 30_000
MAX_PDF = 2_000_000


def _cache_path(kind: str, url: str) -> Path:
    return CACHE / f"{kind}_{hashlib.sha256(url.encode()).hexdigest()[:32]}.json"


def _to_text(name: str, payload: bytes) -> str:
    """Text of a documentation file, without running anything it contains."""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(payload))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:30])
    if suffix == ".docx":
        with zipfile.ZipFile(io.BytesIO(payload)) as docx:
            xml = docx.read("word/document.xml").decode("utf-8", "replace")
        return re.sub(r"<[^>]+>", "", xml.replace("</w:p>", "\n"))
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(io.BytesIO(payload), sheet_name=None, header=None)
        return "\n\n".join(f"## sheet {sheet}\n{frame.to_csv(index=False, header=False)}"
                           for sheet, frame in sheets.items())
    text = payload.decode("utf-8", "replace")
    if suffix == ".rtf":
        text = re.sub(r"\\[a-z]+-?\d* ?|[{}]", "", text)
    if suffix == ".html":
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def is_documentation(name: str, size: int | None) -> bool:
    base = PurePosixPath(name).name
    suffix = PurePosixPath(name).suffix.lower()
    if "__MACOSX" in name or base.startswith("._") or not DOC_NAME.search(base):
        return False
    size = int(size or 0)
    if suffix == ".pdf":
        return size <= MAX_PDF
    return suffix in TEXT | {".docx", ".xlsx"} and size <= MAX_TEXT


def fetch_document(url: str, name: str, offline: bool) -> str:
    path = _cache_path("doc", url)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]
    if offline:
        return "[not cached]"
    try:
        response = _client().get(url, headers={"Accept": "*/*"})
        response.raise_for_status()
        text = _to_text(name, response.content)
    except Exception as error:  # noqa: BLE001 - recorded, the screen goes on
        text = f"[unreadable: {type(error).__name__}: {error}]"[:300]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "name": name, "fetched": date.today().isoformat(),
                                "text": text[:60_000]}), encoding="utf-8")
    return text


def fetch_member(url: str, member: str, offline: bool) -> str:
    key = f"{url}#{member}"
    path = _cache_path("member", key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]
    if offline:
        return "[not cached]"
    from remotezip import RemoteZip

    try:
        with RemoteZip(url, support_suffix_range=False, timeout=120) as archive:
            text = _to_text(member, archive.read(member))
    except Exception as error:  # noqa: BLE001
        text = f"[unreadable: {type(error).__name__}: {error}]"[:300]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "member": member, "fetched": date.today().isoformat(),
                                "text": text[:60_000]}), encoding="utf-8")
    return text


def candidates() -> pd.DataFrame:
    decisions = pd.read_csv(DECISIONS)
    hits = pd.read_csv(HITS)[["doi", "url"]]
    check = decisions[decisions["decision"] == "check"].drop(columns=["url"]).merge(hits, on="doi", how="left")
    check["origin"] = "to_check"
    split = pd.read_csv(SPLIT)
    drops = split[split["doi"].isin(FIRST_TEST_DROPS)].copy()
    drops["origin"] = "first_test_drop"
    drops["source_id"] = drops["doi"].map(FIRST_TEST_DROPS)
    return pd.concat([check, drops], ignore_index=True)


def dossier(offline: bool = False) -> list[dict[str, Any]]:
    entries = []
    for row in candidates().itertuples():
        files, note = file_records(row.doi, {"url": row.url}, offline)
        entry: dict[str, Any] = {
            "doi": row.doi, "origin": row.origin, "title": row.title, "paper_doi": row.paper_doi,
            "venue": row.venue, "licence": row.licence, "listing_note": note,
            "files": [{"path": f["path"], "size": f["size"]} for f in files],
            "archives": {}, "documents": {},
        }
        for f in files:
            if not f.get("url"):
                continue
            if f["path"].lower().endswith(".zip"):
                listing = list_zip(f["url"], offline)
                entry["archives"][f["path"]] = listing
                for member in listing.get("members", []):
                    if is_documentation(member["path"], member["size"]):
                        entry["documents"][f"{f['path']}::{member['path']}"] = fetch_member(
                            f["url"], member["path"], offline)
            elif is_documentation(f["path"], f["size"]):
                entry["documents"][f["path"]] = fetch_document(f["url"], f["path"], offline)
        entries.append(entry)
        print(f"{row.doi}: {len(files)} files, {len(entry['archives'])} archives, "
              f"{len(entry['documents'])} documents", flush=True)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use cached responses only")
    args = parser.parse_args()
    entries = dossier(args.offline)
    DOSSIER.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(entries)} datasets -> {DOSSIER}")


if __name__ == "__main__":
    main()
