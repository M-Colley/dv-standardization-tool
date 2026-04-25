"""Direct unit tests for the extracted scripts.http_utils module.

These tests import from ``scripts.http_utils`` rather than via the
``scripts.run_batch_standardization`` re-export, so they fail loudly if
the standalone module ever stops being importable on its own. The
broader behavioural coverage (transient retry, 429 retry, permanent 404,
challenge detection, web access errors) lives in
``test_run_batch_standardization``::ReadUrlResponseTests and
WebDatasetChallengeTests and exercises the same code through the
orchestrator.
"""

from __future__ import annotations

import socket
import unittest
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from urllib import error as urlerror

from scripts.http_utils import (
    HTTP_USER_AGENT,
    NETWORK_TIMEOUTS,
    SourceAccessError,
    WEB_CHALLENGE_BODY_MARKERS,
    WEB_CHALLENGE_HOSTS,
    WEB_CHALLENGE_TITLE_MARKERS,
    WEB_LOGIN_MARKERS,
    WEB_NO_DATA_MARKERS,
    _LandingPageParser,
    _build_unique_destination,
    _build_web_dataset_access_error,
    _decode_http_text,
    _extract_content_disposition_filename,
    _extract_html_download_urls,
    _infer_extension_from_content_type,
    _is_supported_download_response,
    _looks_like_challenge_response,
    _looks_like_supported_download_url,
    _normalize_remote_filename,
    _read_url_response,
    _write_download_payload,
)


class HttpUtilsModuleSurfaceTests(unittest.TestCase):
    """Lock in the public surface of scripts.http_utils."""

    def test_user_agent_string_is_open_dv(self):
        # Server-side log filters and user-agent allow-lists rely on this token.
        self.assertEqual(HTTP_USER_AGENT, "OpenDV-HCI/1.0")

    def test_network_timeouts_keys_present(self):
        for key in ("download_seconds", "landing_seconds", "api_seconds", "max_retry_attempts"):
            self.assertIn(key, NETWORK_TIMEOUTS)
        self.assertGreaterEqual(int(NETWORK_TIMEOUTS["max_retry_attempts"]), 1)

    def test_no_data_markers_default_to_ascii_variant(self):
        # The literal CJK fallback was a copy-paste duplicate; the canonical
        # exposed tuple is the unicode-escaped one.
        self.assertGreater(len(WEB_NO_DATA_MARKERS), 0)
        self.assertIn("no data", WEB_NO_DATA_MARKERS)

    def test_challenge_marker_sets_non_empty(self):
        self.assertIn("dl.acm.org", WEB_CHALLENGE_HOSTS)
        self.assertTrue(any("just a moment" in m for m in WEB_CHALLENGE_TITLE_MARKERS))
        self.assertTrue(any("cloudflare" in m for m in WEB_CHALLENGE_BODY_MARKERS))


class SourceAccessErrorTests(unittest.TestCase):
    def test_status_attribute_is_preserved(self):
        err = SourceAccessError("login_required", "boom")
        self.assertEqual(err.status, "login_required")
        self.assertEqual(str(err), "boom")
        self.assertIsInstance(err, RuntimeError)


class LandingPageParserTests(unittest.TestCase):
    def test_captures_title_links_and_ldjson(self):
        parser = _LandingPageParser()
        parser.feed(
            "<html><head><title>Dataset X</title>"
            '<script type="application/ld+json">{"contentUrl":"a.csv"}</script>'
            '</head><body><a href="b.csv">b</a></body></html>'
        )
        self.assertEqual(parser.title, "Dataset X")
        self.assertIn("b.csv", parser.links)
        self.assertEqual(len(parser.ldjson_blocks), 1)
        self.assertIn("a.csv", parser.ldjson_blocks[0])


class FilenameHelpersTests(unittest.TestCase):
    def test_extract_content_disposition_returns_none_for_empty(self):
        self.assertIsNone(_extract_content_disposition_filename(None))
        self.assertIsNone(_extract_content_disposition_filename(""))

    def test_extract_content_disposition_handles_utf8_filename_star(self):
        result = _extract_content_disposition_filename(
            "attachment; filename*=UTF-8''my%20study.csv"
        )
        self.assertEqual(result, "my study.csv")

    def test_extract_content_disposition_handles_quoted_filename(self):
        result = _extract_content_disposition_filename(
            'attachment; filename="study.csv"'
        )
        self.assertEqual(result, "study.csv")

    def test_infer_extension_from_known_content_type(self):
        self.assertEqual(_infer_extension_from_content_type("application/zip"), ".zip")
        self.assertEqual(_infer_extension_from_content_type("text/csv; charset=utf-8"), ".csv")
        self.assertEqual(_infer_extension_from_content_type("application/octet-stream"), "")

    def test_normalize_remote_filename_strips_traversal_chars(self):
        # Path-traversal attempt should be stripped to a flat filename.
        result = _normalize_remote_filename(
            raw_name="../../etc/passwd",
            fallback_prefix="dataset",
            final_url="https://example.com/x.csv",
            content_type="text/csv",
        )
        # Result must contain no path separators or traversal sequences.
        self.assertNotIn("/", result)
        self.assertNotIn("..", result)
        self.assertNotEqual(result, "")

    def test_normalize_remote_filename_falls_back_to_url_basename(self):
        result = _normalize_remote_filename(
            raw_name=None,
            fallback_prefix="dataset",
            final_url="https://example.com/foo/bar.tsv",
            content_type="text/tab-separated-values",
        )
        self.assertEqual(result, "bar.tsv")

    def test_normalize_remote_filename_uses_fallback_when_url_empty(self):
        result = _normalize_remote_filename(
            raw_name=None,
            fallback_prefix="dataset",
            final_url="https://example.com/",
            content_type="application/zip",
        )
        self.assertEqual(result, "dataset.zip")

    def test_build_unique_destination_avoids_collision(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "a.csv").write_text("first")
            dest = _build_unique_destination(target, "a.csv")
            self.assertEqual(dest.name, "a_2.csv")


class SupportedDownloadDetectionTests(unittest.TestCase):
    def test_recognises_csv_url(self):
        self.assertTrue(_looks_like_supported_download_url("https://x/data.csv"))

    def test_recognises_osf_ndownloader_path(self):
        self.assertTrue(
            _looks_like_supported_download_url("https://osf.io/abc/ndownloader/files/123")
        )

    def test_rejects_html_url(self):
        self.assertFalse(_looks_like_supported_download_url("https://x/index.html"))

    def test_supported_download_response_via_content_type_when_url_is_extensionless(self):
        headers = Message()
        headers["Content-Type"] = "application/zip"
        self.assertTrue(_is_supported_download_response("https://x/download", headers))

    def test_decode_http_text_uses_charset_when_present(self):
        headers = Message()
        headers["Content-Type"] = "text/html; charset=latin-1"
        text = _decode_http_text("café".encode("latin-1"), headers)
        self.assertEqual(text, "café")


class ChallengeDetectionTests(unittest.TestCase):
    def test_known_challenge_host_is_flagged(self):
        self.assertTrue(_looks_like_challenge_response("dl.acm.org", "", ""))

    def test_title_marker_triggers_challenge(self):
        self.assertTrue(_looks_like_challenge_response("example.com", "Just a moment...", ""))

    def test_body_marker_triggers_challenge(self):
        self.assertTrue(
            _looks_like_challenge_response(
                "example.com",
                "Welcome",
                "<p>please enable javascript and cookies to continue</p>",
            )
        )

    def test_clean_response_is_not_a_challenge(self):
        self.assertFalse(_looks_like_challenge_response("example.com", "Dataset X", "<p>data here</p>"))


class WebAccessErrorBuilderTests(unittest.TestCase):
    def test_classifies_cloudflare_challenge_as_access_restricted(self):
        err = _build_web_dataset_access_error(
            location="https://dl.acm.org/abs/123",
            final_url="https://dl.acm.org/abs/123",
            html_text="enable javascript and cookies to continue",
            title="Just a moment",
        )
        self.assertEqual(err.status, "access_restricted")

    def test_classifies_ieee_login_required(self):
        err = _build_web_dataset_access_error(
            location="https://ieee-dataport.org/x",
            final_url="https://ieee-dataport.org/x",
            html_text="<p>" + WEB_LOGIN_MARKERS[0] + "</p>",
            title="X",
        )
        self.assertEqual(err.status, "login_required")

    def test_classifies_4tu_with_no_links_as_not_available(self):
        err = _build_web_dataset_access_error(
            location="https://data.4tu.nl/x",
            final_url="https://data.4tu.nl/x",
            html_text="<html></html>",
            title="X",
        )
        self.assertEqual(err.status, "not_available")


class ReadUrlResponseDirectImportTests(unittest.TestCase):
    """One smoke test for the retry path against the directly-imported
    helper. Comprehensive scenarios live in
    test_run_batch_standardization::ReadUrlResponseTests."""

    def test_max_attempts_default_picks_up_network_timeouts(self):
        # When max_attempts is None, the function should fall back to
        # NETWORK_TIMEOUTS["max_retry_attempts"] retries before re-raising.
        attempts: list[int] = []

        def boom(*args, **kwargs):
            attempts.append(1)
            raise socket.timeout("simulated")

        with mock.patch("scripts.http_utils.request.urlopen", side_effect=boom):
            with self.assertRaises(socket.timeout):
                _read_url_response("https://x", timeout=1, max_attempts=None)

        self.assertEqual(len(attempts), int(NETWORK_TIMEOUTS["max_retry_attempts"]))


class WriteDownloadPayloadTests(unittest.TestCase):
    def test_writes_payload_using_url_basename(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp)
            headers = Message()
            headers["Content-Type"] = "text/csv"
            written = _write_download_payload(
                target,
                payload=b"a,b\n1,2\n",
                final_url="https://example.com/study.csv",
                headers=headers,
                fallback_prefix="dataset",
            )
            self.assertEqual(written.name, "study.csv")
            self.assertEqual(written.read_bytes(), b"a,b\n1,2\n")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
