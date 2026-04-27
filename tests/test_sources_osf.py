"""Direct unit tests for scripts.sources.osf.

Validates the OSF JSON-API helpers as a standalone import (not via the
``scripts.run_batch_standardization`` re-export). The orchestrator's
end-to-end use of ``_iter_osf_file_entries`` is covered by the existing
batch-discovery tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts.sources.osf import (
    OSF_API_BASE,
    _download_osf_file,
    _extract_osf_project_id,
    _iter_osf_child_node_ids,
    _iter_osf_file_entries,
    _iter_osf_node_ids,
    _iter_osf_provider_urls,
)


class ExtractOsfProjectIdTests(unittest.TestCase):
    def test_accepts_lowercase_id(self):
        self.assertEqual(_extract_osf_project_id("cwd6h"), "cwd6h")

    def test_accepts_mixed_case_id(self):
        self.assertEqual(_extract_osf_project_id("CWd6H"), "cwd6h")

    def test_accepts_full_url(self):
        self.assertEqual(
            _extract_osf_project_id("https://osf.io/cwd6h/overview"),
            "cwd6h",
        )

    def test_accepts_url_with_trailing_slash(self):
        self.assertEqual(_extract_osf_project_id("https://osf.io/cwd6h/"), "cwd6h")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _extract_osf_project_id("")
        self.assertIn("non-empty", str(ctx.exception))

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _extract_osf_project_id("not_a_valid_id")
        self.assertIn("Invalid OSF location", str(ctx.exception))


class OsfApiBaseTests(unittest.TestCase):
    def test_api_base_points_to_v2(self):
        # The traversal helpers concatenate this constant — guard against drift.
        self.assertEqual(OSF_API_BASE, "https://api.osf.io/v2")


class IterOsfChildNodeIdsTests(unittest.TestCase):
    def test_walks_pagination(self):
        page1 = {
            "data": [{"id": "node_a"}, {"id": "node_b"}],
            "links": {"next": "https://api.osf.io/v2/nodes/parent/children/?page=2"},
        }
        page2 = {
            "data": [{"id": "node_c"}],
            "links": {"next": None},
        }

        with mock.patch(
            "scripts.sources.osf._osf_json_get",
            side_effect=[page1, page2],
        ) as mocked:
            ids = _iter_osf_child_node_ids("parent")

        self.assertEqual(ids, ["node_a", "node_b", "node_c"])
        self.assertEqual(mocked.call_count, 2)

    def test_skips_blank_ids(self):
        with mock.patch(
            "scripts.sources.osf._osf_json_get",
            return_value={"data": [{"id": ""}, {"id": "x"}], "links": {}},
        ):
            self.assertEqual(_iter_osf_child_node_ids("parent"), ["x"])


class IterOsfNodeIdsTests(unittest.TestCase):
    def test_bfs_includes_root_then_children(self):
        with mock.patch(
            "scripts.sources.osf._iter_osf_child_node_ids",
            side_effect=[["child_a", "child_b"], [], []],
        ):
            ids = _iter_osf_node_ids("root")
        # Root first, then children in discovery order.
        self.assertEqual(ids[0], "root")
        self.assertEqual(set(ids[1:]), {"child_a", "child_b"})

    def test_dedupes_repeated_ids(self):
        with mock.patch(
            "scripts.sources.osf._iter_osf_child_node_ids",
            side_effect=[["root"], []],
        ):
            ids = _iter_osf_node_ids("root")
        self.assertEqual(ids, ["root"])


class IterOsfProviderUrlsTests(unittest.TestCase):
    def test_extracts_provider_links_from_relationships(self):
        page = {
            "data": [
                {
                    "relationships": {
                        "files": {
                            "links": {
                                "related": {"href": "https://api.osf.io/v2/nodes/x/files/osfstorage/"}
                            }
                        }
                    }
                }
            ],
            "links": {"next": None},
        }
        with mock.patch("scripts.sources.osf._osf_json_get", return_value=page):
            urls = _iter_osf_provider_urls("x")
        self.assertEqual(urls, ["https://api.osf.io/v2/nodes/x/files/osfstorage/"])

    def test_falls_back_to_top_level_related_link(self):
        page = {
            "data": [{"links": {"related": {"href": "https://example.com/foo"}}}],
            "links": {"next": None},
        }
        with mock.patch("scripts.sources.osf._osf_json_get", return_value=page):
            urls = _iter_osf_provider_urls("x")
        self.assertEqual(urls, ["https://example.com/foo"])


class IterOsfFileEntriesTests(unittest.TestCase):
    def test_recurses_into_folders(self):
        provider_page = {
            "data": [
                {
                    "relationships": {
                        "files": {"links": {"related": {"href": "https://api.osf.io/v2/x/files/osfstorage/"}}}
                    }
                }
            ],
            "links": {"next": None},
        }
        osfstorage_page = {
            "data": [
                {
                    "attributes": {"kind": "file", "name": "a.csv"},
                },
                {
                    "attributes": {"kind": "folder", "name": "sub"},
                    "relationships": {
                        "files": {"links": {"related": {"href": "https://api.osf.io/v2/x/files/osfstorage/sub/"}}}
                    },
                },
            ],
            "links": {"next": None},
        }
        sub_page = {
            "data": [{"attributes": {"kind": "file", "name": "b.csv"}}],
            "links": {"next": None},
        }
        with mock.patch(
            "scripts.sources.osf._osf_json_get",
            side_effect=[provider_page, osfstorage_page, sub_page],
        ):
            entries = _iter_osf_file_entries("x")
        names = [e["attributes"]["name"] for e in entries]
        self.assertEqual(names, ["a.csv", "b.csv"])


class DownloadOsfFileTests(unittest.TestCase):
    def test_writes_payload_and_creates_parent(self):
        with TemporaryDirectory() as tmp:
            destination = Path(tmp) / "deeply" / "nested" / "file.csv"
            with mock.patch(
                "scripts.sources.osf._read_url_bytes",
                return_value=b"a,b\n1,2\n",
            ):
                _download_osf_file("https://x/file.csv", destination)
            self.assertTrue(destination.exists())
            self.assertEqual(destination.read_bytes(), b"a,b\n1,2\n")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
