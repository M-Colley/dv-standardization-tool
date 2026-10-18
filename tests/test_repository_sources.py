"""Zenodo, figshare, 4TU.ResearchData and Dataverse sources (HTTP mocked).

Set OPENDV_NETWORK_TESTS=1 to also run one live, metadata-only API check.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import pandas as pd
import pytest
import yaml

from scripts import run_batch_standardization as rbs
from scripts.http_utils import SourceAccessError
from scripts.sources import repositories as repo

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"
CSV = b"UserID,TLX1,TLX2\n1,10,4\n2,12,6\n3,8,5\n"
MD5 = hashlib.md5(CSV).hexdigest()


# ── Location parsing ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("location, expected", [
    ("https://zenodo.org/records/18344381/files/RunIntoTraffic.zip?download=1", "18344381"),
    ("https://zenodo.org/record/123456", "123456"),
    ("https://doi.org/10.5281/zenodo.99", "99"),
    ("777", "777"),
])
def test_parse_zenodo(location, expected):
    assert repo.parse_zenodo_location(location) == expected


@pytest.mark.parametrize("location, expected", [
    ("https://figshare.com/articles/dataset/My_data/12345678/2", ("12345678", "2")),
    ("https://springernature.figshare.com/articles/dataset/x/23456789", ("23456789", None)),
    ("10.6084/m9.figshare.12345678.v3", ("12345678", "3")),
    ("12345678", ("12345678", None)),
])
def test_parse_figshare(location, expected):
    assert repo.parse_figshare_location(location) == expected


@pytest.mark.parametrize("location, expected", [
    ("https://data.4tu.nl/datasets/0b9f2b9e-1c45-4bd6-9b8a-0cb1a3d1f3ab/2",
     ("0b9f2b9e-1c45-4bd6-9b8a-0cb1a3d1f3ab", "2")),
    ("https://data.4tu.nl/articles/_/20224281", ("20224281", None)),
])
def test_parse_fourtu(location, expected):
    assert repo.parse_fourtu_location(location) == expected


def test_parse_dataverse():
    url = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ABCDEF&version=1.0"
    assert repo.parse_dataverse_location(url) == ("https://dataverse.harvard.edu", "doi:10.7910/DVN/ABCDEF")


@pytest.mark.parametrize("fn, bad", [
    (repo.parse_zenodo_location, "https://example.org/x"),
    (repo.parse_figshare_location, "https://figshare.com/about"),
    (repo.parse_fourtu_location, "https://data.4tu.nl/"),
    (repo.parse_dataverse_location, "https://dataverse.harvard.edu/dataverse/root"),
])
def test_unparseable_locations_raise(fn, bad):
    with pytest.raises(ValueError):
        fn(bad)


# ── File listings from API payloads ──────────────────────────────────────────

@pytest.fixture()
def api(monkeypatch):
    """Route repo._get_json to canned payloads keyed by URL."""
    payloads: dict[str, object] = {}

    def fake_get(url):
        if url not in payloads:
            raise SourceAccessError("not_available", f"HTTP 404 for {url}")
        return payloads[url]

    monkeypatch.setattr(repo, "_get_json", fake_get)
    return payloads


def test_zenodo_listing(api):
    api["https://zenodo.org/api/records/5"] = {"id": 5, "revision": 3, "files": [
        {"key": "data/study.csv", "size": len(CSV), "checksum": f"md5:{MD5}",
         "links": {"self": "https://zenodo.org/api/records/5/files/data/study.csv/content"}},
    ]}
    record = repo.zenodo_record("5")
    assert record.revision == "zenodo:5@3"
    assert record.files[0] == repo.RemoteFile(
        "data/study.csv", "https://zenodo.org/api/records/5/files/data/study.csv/content", len(CSV), ("md5", MD5)
    )


def test_zenodo_listing_invenio_entries(api):
    api["https://zenodo.org/api/records/6"] = {"id": 6, "files": {"entries": {"a.csv": {
        "key": "a.csv", "size": 3, "checksum": "md5:abc", "links": {"content": "https://z/a.csv/content"}}}}}
    assert repo.zenodo_record("6").files[0].url == "https://z/a.csv/content"


def test_fourtu_falls_back_to_files_endpoint(api):
    uuid = "0b9f2b9e-1c45-4bd6-9b8a-0cb1a3d1f3ab"
    api[f"https://data.4tu.nl/v2/articles/{uuid}"] = {"uuid": uuid, "version": 2, "files": []}
    api[f"https://data.4tu.nl/v2/articles/{uuid}/files"] = [
        {"name": "x.csv", "download_url": "https://data.4tu.nl/file/x", "size": 1, "computed_md5": "ff"},
    ]
    record = repo.fourtu_record(uuid)
    assert record.revision == f"4tu:{uuid}@v2"
    assert record.files[0].checksum == ("md5", "ff")


def test_dataverse_requests_original_uploads(api):
    api["https://dv.example/api/datasets/:persistentId/?persistentId=doi:10.1/X"] = {"data": {"latestVersion": {
        "versionNumber": 2, "versionMinorNumber": 1,
        "files": [
            {"directoryLabel": "raw", "dataFile": {
                "id": 11, "filename": "survey.tab", "originalFileName": "survey.csv", "originalFileSize": 99,
                "checksum": {"type": "MD5", "value": "aa"}}},
            {"dataFile": {"id": 12, "filename": "codebook.yaml", "filesize": 7, "md5": "bb"}},
        ],
    }}}
    record = repo.dataverse_record("https://dv.example", "doi:10.1/X")
    assert record.revision == "doi:10.1/X@v2.1"
    survey, codebook = record.files
    assert survey.path == "raw/survey.csv"
    assert survey.url == "https://dv.example/api/access/datafile/11?format=original"
    assert survey.checksum is None and survey.size == 99  # checksum belongs to the .tab conversion
    assert codebook.checksum == ("md5", "bb")


def test_http_errors_become_access_statuses(monkeypatch):
    def raise_status(code):
        def _read(req, timeout):
            request = httpx.Request("GET", "https://x")
            raise httpx.HTTPStatusError("err", request=request, response=httpx.Response(code, request=request))
        return _read

    for code, status in ((404, "not_available"), (403, "access_restricted"), (401, "login_required")):
        monkeypatch.setattr(repo, "_read_url_bytes", raise_status(code))
        with pytest.raises(SourceAccessError) as info:
            repo._get_json("https://x")
        assert info.value.status == status
    monkeypatch.setattr(repo, "_read_url_bytes", raise_status(503))
    with pytest.raises(httpx.HTTPStatusError):  # outages are reported by the runner
        repo._get_json("https://x")


# ── Download ─────────────────────────────────────────────────────────────────

def _fake_downloader(bodies: dict[str, bytes]):
    def download(url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bodies[url])
    return download


def test_download_filters_verifies_and_guards(tmp_path):
    record = repo.RemoteRecord((
        repo.RemoteFile("data/study.csv", "u1", len(CSV), ("md5", MD5)),
        repo.RemoteFile("paper.pdf", "u2", 5, None),
        repo.RemoteFile("../escape.csv", "u3", 3, None),
        repo.RemoteFile("big.csv", "u4", 10_000_000, None),
    ), "rev")
    written = repo.download_record(record, tmp_path, {".csv"}, max_file_bytes=1_000_000,
                                   downloader=_fake_downloader({"u1": CSV}))
    assert written == [(tmp_path / "data" / "study.csv").resolve()]
    assert not (tmp_path.parent / "escape.csv").exists()


def test_checksum_mismatch_deletes_the_file(tmp_path):
    record = repo.RemoteRecord((repo.RemoteFile("a.csv", "u", len(CSV), ("md5", "0" * 32)),), "rev")
    with pytest.raises(ValueError, match="checksum mismatch"):
        repo.download_record(record, tmp_path, {".csv"}, downloader=_fake_downloader({"u": CSV}))
    assert not (tmp_path / "a.csv").exists()


def test_size_mismatch_is_rejected(tmp_path):
    record = repo.RemoteRecord((repo.RemoteFile("a.csv", "u", len(CSV) + 1, None),), "rev")
    with pytest.raises(ValueError, match="size mismatch"):
        repo.download_record(record, tmp_path, {".csv"}, downloader=_fake_downloader({"u": CSV}))


# ── Batch runner integration ─────────────────────────────────────────────────

def test_zenodo_source_runs_through_the_batch(tmp_path, monkeypatch):
    calls = []

    def fake_record(record_id):
        calls.append(record_id)
        return repo.RemoteRecord((repo.RemoteFile("study.csv", "u1", len(CSV), ("md5", MD5)),), "zenodo:42@7")

    monkeypatch.setattr(repo, "zenodo_record", fake_record)
    monkeypatch.setattr(repo, "_download", _fake_downloader({"u1": CSV}))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": [{
        "source_id": "zen", "source_type": "zenodo_record", "location": "https://zenodo.org/records/42",
        "use_llm_deduction": False,
    }]}), encoding="utf-8")
    for run in ("a", "b"):
        summary = rbs.run_batch(manifest, tmp_path / run, SCHEMA, llm_deduction_enabled=False,
                                cache_root=tmp_path / "cache", fair_export=False)
        assert summary["successful_sources"] == 1 and not summary["failed_sources"], summary["results"]
    assert calls == ["42"]  # the second run reuses the verified cache
    meta = pd.read_csv(tmp_path / "b" / "meta_view.csv")
    assert set(meta["commit"]) == {"zenodo:42@7"}
    assert "mental_demand" in set(meta["canonical_dv"])


def test_unknown_record_is_reported_not_failed(tmp_path, monkeypatch):
    def missing(record_id):
        raise SourceAccessError("not_available", "HTTP 404")

    monkeypatch.setattr(repo, "zenodo_record", missing)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": [{
        "source_id": "gone", "source_type": "zenodo_record", "location": "1",
    }]}), encoding="utf-8")
    summary = rbs.run_batch(manifest, tmp_path / "out", SCHEMA, llm_deduction_enabled=False,
                            cache_root=tmp_path / "cache", fair_export=False)
    assert summary["failed_sources"] == []
    assert json.dumps(summary["unavailable_sources"]).count("gone") == 1


@pytest.mark.skipif(os.environ.get("OPENDV_NETWORK_TESTS") != "1", reason="set OPENDV_NETWORK_TESTS=1")
def test_live_zenodo_metadata():
    record = repo.zenodo_record("18344381")  # Running into Traffic (CHI 2026); metadata only
    assert any(f.path == "RunIntoTraffic.zip" and f.checksum for f in record.files)
