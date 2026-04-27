"""Direct unit tests for scripts.survey_parser_base.

Validates the encoding/delimiter sniffers, the slug helpers, and the
SurveyParser ABC contract as a standalone import — not via the
``scripts.survey_parsers`` re-export. Behavioural coverage of the three
concrete parsers (Qualtrics, LimeSurvey, REDCap) lives in the existing
``tests/test_extensibility.py`` and continues to exercise this base
module through the re-export.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from scripts.survey_parser_base import (
    SurveyParser,
    _detect_delimiter,
    _detect_encoding,
    _ensure_unique_columns,
    _read_file_bytes,
    _slugify,
)


class ReadFileBytesTests(unittest.TestCase):
    def test_reads_full_file_when_under_limit(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.bin"
            p.write_bytes(b"hello world")
            self.assertEqual(_read_file_bytes(p), b"hello world")

    def test_truncates_to_max_bytes(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.bin"
            p.write_bytes(b"a" * 1000)
            self.assertEqual(len(_read_file_bytes(p, max_bytes=128)), 128)


class DetectEncodingTests(unittest.TestCase):
    def test_utf8_bom(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_bytes(b"\xef\xbb\xbfhello")
            self.assertEqual(_detect_encoding(p), "utf-8-sig")

    def test_utf16_le_bom(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_bytes(b"\xff\xfehello")
            self.assertEqual(_detect_encoding(p), "utf-16-le")

    def test_utf16_be_bom(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_bytes(b"\xfe\xffhello")
            self.assertEqual(_detect_encoding(p), "utf-16-be")

    def test_falls_back_to_utf8_for_ascii(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_text("a,b\n1,2\n", encoding="ascii")
            self.assertEqual(_detect_encoding(p), "utf-8")


class DetectDelimiterTests(unittest.TestCase):
    def test_csv_comma(self):
        self.assertEqual(_detect_delimiter("a,b,c\n1,2,3\n4,5,6\n"), ",")

    def test_tsv_tab(self):
        self.assertEqual(
            _detect_delimiter("a\tb\tc\n1\t2\t3\n4\t5\t6\n", prefer="\t"),
            "\t",
        )

    def test_semicolon(self):
        self.assertEqual(_detect_delimiter("a;b;c\n1;2;3\n4;5;6\n"), ";")

    def test_pipe(self):
        self.assertEqual(_detect_delimiter("a|b|c\n1|2|3\n4|5|6\n"), "|")

    def test_empty_sample_returns_prefer(self):
        self.assertEqual(_detect_delimiter("", prefer=";"), ";")
        self.assertEqual(_detect_delimiter("", prefer=None), ",")


class SlugifyTests(unittest.TestCase):
    def test_replaces_non_alphanumeric_runs_with_underscore(self):
        self.assertEqual(_slugify("Hello, World!"), "Hello_World")

    def test_collapses_consecutive_underscores(self):
        self.assertEqual(_slugify("hello___world"), "hello_world")

    def test_strips_leading_and_trailing_underscores(self):
        self.assertEqual(_slugify("___task time___"), "task_time")

    def test_unicode_characters_become_underscores(self):
        # Non-ASCII alphanumerics also get squashed to underscores.
        self.assertEqual(_slugify("café — bistro"), "caf_bistro")

    def test_truncates_to_max_len(self):
        self.assertEqual(len(_slugify("a" * 200, max_len=50)), 50)

    def test_returns_unnamed_for_pure_garbage(self):
        self.assertEqual(_slugify("---!!!"), "unnamed")
        self.assertEqual(_slugify(""), "unnamed")


class EnsureUniqueColumnsTests(unittest.TestCase):
    def test_unique_columns_pass_through(self):
        self.assertEqual(_ensure_unique_columns(["a", "b", "c"]), ["a", "b", "c"])

    def test_duplicates_get_numeric_suffix_starting_at_2(self):
        self.assertEqual(
            _ensure_unique_columns(["a", "a", "a"]),
            ["a", "a_2", "a_3"],
        )

    def test_preserves_original_order(self):
        self.assertEqual(
            _ensure_unique_columns(["a", "b", "a", "c", "b"]),
            ["a", "b", "a_2", "c", "b_2"],
        )

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(_ensure_unique_columns([]), [])


class SurveyParserAbcTests(unittest.TestCase):
    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            SurveyParser()  # type: ignore[abstract]

    def test_subclass_must_implement_parse(self):
        class _Incomplete(SurveyParser):
            pass

        with self.assertRaises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_subclass_with_parse_can_be_instantiated(self):
        class _Concrete(SurveyParser):
            def parse(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
                return pd.DataFrame({"x": [1, 2]})

        parser = _Concrete()
        df = parser.parse("anything")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df["x"].tolist(), [1, 2])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
