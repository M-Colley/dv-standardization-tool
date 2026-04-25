"""Direct unit tests for the extracted scripts.archive_utils module.

These tests import from ``scripts.archive_utils`` rather than via the
``scripts.run_batch_standardization`` re-export, so they fail loudly if
the standalone module ever stops being importable on its own. The
broader behavioural coverage (zip-slip, max_depth, nested archives,
MACOSX/hidden skip, unsupported suffix skip, __extracted_archives skip)
already lives in test_run_batch_standardization::ArchiveExtractionTests
and exercises the same code through the orchestrator.
"""

from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.archive_utils import (
    ARCHIVE_SUFFIXES,
    DATA_FILE_SUFFIXES,
    DEFAULT_ARCHIVE_MAX_DEPTH,
    MAPPING_SUFFIXES,
    _extract_archives_in_tree,
    _extract_zip_files_recursive,
    _is_within_directory,
    _resolve_archive_max_depth,
    _should_skip_archive_member,
)


class ArchiveUtilsModuleSurfaceTests(unittest.TestCase):
    """Lock in the public surface of scripts.archive_utils."""

    def test_default_archive_max_depth_is_three(self):
        # The default has been 3 since the original implementation; tests
        # downstream rely on it not silently moving.
        self.assertEqual(DEFAULT_ARCHIVE_MAX_DEPTH, 3)

    def test_archive_suffix_set_only_contains_zip(self):
        # We have not yet added .tar/.tar.gz support — guard against drift.
        self.assertEqual(ARCHIVE_SUFFIXES, {".zip"})

    def test_mapping_suffix_set_contains_yaml_variants(self):
        self.assertEqual(MAPPING_SUFFIXES, {".yaml", ".yml"})

    def test_data_file_suffixes_includes_common_tabular_formats(self):
        # Sanity: at minimum csv/tsv/xlsx must be discoverable. Full set
        # is derived from convert_dv._FORMAT_REGISTRY at import time.
        for suffix in (".csv", ".tsv", ".xlsx"):
            self.assertIn(suffix, DATA_FILE_SUFFIXES)
        # Ambiguous extensions should be excluded from auto-discovery.
        self.assertNotIn(".txt", DATA_FILE_SUFFIXES)
        self.assertNotIn(".dat", DATA_FILE_SUFFIXES)


class IsWithinDirectoryTests(unittest.TestCase):
    """Zip-slip guard primitive."""

    def test_returns_true_when_path_is_inside_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inside = (root / "sub" / "file.csv").resolve()
            self.assertTrue(_is_within_directory(inside, root))

    def test_returns_false_when_path_escapes_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            outside = (root.parent / "sibling" / "file.csv").resolve()
            self.assertFalse(_is_within_directory(outside, root))

    def test_returns_true_for_root_itself(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertTrue(_is_within_directory(root, root))


class ResolveArchiveMaxDepthTests(unittest.TestCase):
    def test_uses_default_when_field_absent(self):
        self.assertEqual(_resolve_archive_max_depth({}), DEFAULT_ARCHIVE_MAX_DEPTH)

    def test_accepts_integer_string(self):
        self.assertEqual(_resolve_archive_max_depth({"archive_max_depth": "5"}), 5)

    def test_rejects_non_numeric_value(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve_archive_max_depth({"archive_max_depth": "deep"})
        self.assertIn("integer", str(ctx.exception))

    def test_rejects_zero_or_negative(self):
        with self.assertRaises(ValueError):
            _resolve_archive_max_depth({"archive_max_depth": 0})
        with self.assertRaises(ValueError):
            _resolve_archive_max_depth({"archive_max_depth": -3})


class ShouldSkipArchiveMemberTests(unittest.TestCase):
    def test_skips_macosx_directory(self):
        self.assertTrue(_should_skip_archive_member(Path("__MACOSX/foo/bar.csv")))

    def test_skips_ds_store(self):
        self.assertTrue(_should_skip_archive_member(Path("data/.DS_Store")))

    def test_skips_apple_double_files(self):
        self.assertTrue(_should_skip_archive_member(Path("data/._hidden.csv")))

    def test_does_not_skip_normal_csv(self):
        self.assertFalse(_should_skip_archive_member(Path("data/study.csv")))


class ExtractZipDirectImportTests(unittest.TestCase):
    """A single end-to-end smoke test against the directly-imported helpers
    (the broader scenarios live in ArchiveExtractionTests)."""

    def test_extracts_csv_from_simple_zip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "data.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("study.csv", "a,b\n1,2\n")

            _extract_zip_files_recursive(zip_path, root, depth=1, max_depth=3)

            extracted = root / "__extracted_archives" / "data" / "study.csv"
            self.assertTrue(extracted.exists())
            self.assertEqual(extracted.read_text(), "a,b\n1,2\n")

    def test_extract_archives_in_tree_processes_top_level_zip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("results/run1.csv", "x\n1\n")

            _extract_archives_in_tree(root, max_depth=DEFAULT_ARCHIVE_MAX_DEPTH)

            extracted = root / "__extracted_archives" / "bundle" / "results" / "run1.csv"
            self.assertTrue(extracted.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
