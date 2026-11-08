"""Per-file table options from the manifest: sheet, header row, long-to-wide pivot and rename."""

from __future__ import annotations

import codecs
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opendv.data_loaders import load_input_file, table_options_for  # noqa: E402
from opendv.encoding_utils import detect_text_encoding  # noqa: E402
from opendv.run_batch_standardization import run_batch  # noqa: E402

SCHEMA = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"


def _workbook(path: Path) -> None:
    """A raw sheet with more cells than the analysed one, and a long TLX sheet under a title row."""
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({f"raw{i}": range(6) for i in range(8)}).to_excel(writer, sheet_name="original", index=False)
        pd.DataFrame({"UserID": [1, 2], "tlx_mental": [40, 60]}).to_excel(writer, sheet_name="Results", index=False)
        long = pd.DataFrame({
            "subject_id": ["P1", "P1", "P2", "P2"], "curvature": ["Flat"] * 4,
            "subscale": ["Mental", "Physical"] * 2, "score": [10, 4, 12, 6],
        })
        pd.DataFrame([["NASA-TLX Condition-Level Data"]]).to_excel(writer, sheet_name="TLX", index=False, header=False)
        long.to_excel(writer, sheet_name="TLX", index=False, startrow=1)


class TableOptionsTests(unittest.TestCase):
    def test_options_match_exact_paths_before_globs(self):
        options = {"data/*.xlsx": {"sheet": "A"}, "data/main.xlsx": {"sheet": "B"}}
        self.assertEqual(table_options_for(options, "data/main.xlsx"), {"sheet": "B"})
        self.assertEqual(table_options_for(options, "data/other.xlsx"), {"sheet": "A"})
        self.assertEqual(table_options_for(options, "notes.csv"), {})
        self.assertEqual(table_options_for(None, "x.csv"), {})

    def test_sheet_header_and_pivot(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "book.xlsx"
            _workbook(book)
            self.assertEqual(list(load_input_file(str(book)).columns)[:2], ["raw0", "raw1"])  # most cells wins
            self.assertEqual(list(load_input_file(str(book), {"sheet": "Results"}).columns), ["UserID", "tlx_mental"])
            wide = load_input_file(str(book), {
                "sheet": "TLX", "header": 1,
                "pivot": {"index": ["subject_id", "curvature"], "columns": "subscale", "values": "score"},
            })
            self.assertEqual(list(wide.columns), ["subject_id", "curvature", "Mental", "Physical"])
            self.assertEqual(wide["Physical"].tolist(), [4, 6])

    def test_invalid_options_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "t.csv"
            csv.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_input_file(str(csv), {"sheet": "x"})
            with self.assertRaises(ValueError):
                load_input_file(str(csv), {"skip": 2})

    def test_batch_reads_the_declared_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "src").mkdir()
            _workbook(tmp / "src" / "book.xlsx")
            (tmp / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
                "source_id": "s", "source_type": "local_path", "location": str(tmp / "src"),
                "include_globs": ["*.xlsx"], "table_options": {"book.xlsx": {"sheet": "Results"}},
            }]}), encoding="utf-8")
            run_batch(tmp / "manifest.yaml", tmp / "out", SCHEMA, llm_deduction_enabled=False, cache_root=tmp / "cache")
            meta = pd.read_csv(tmp / "out" / "meta_view.csv")
            self.assertIn("tlx_mental", meta["original_alias"].tolist())
            self.assertNotIn("raw0", meta["original_alias"].tolist())


class RenameOptionTests(unittest.TestCase):
    def test_rename_runs_after_pivot_and_drop_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "book.xlsx"
            _workbook(book)
            wide = load_input_file(str(book), {
                "sheet": "TLX", "header": 1,
                "pivot": {"index": ["subject_id", "curvature"], "columns": "subscale", "values": "score"},
                "drop_columns": ["curvature"],
                "rename": {"Mental": "tlx_mental", " Physical ": "tlx_physical"},  # headers match stripped
            })
            self.assertEqual(list(wide.columns), ["subject_id", "tlx_mental", "tlx_physical"])
            self.assertEqual(wide["tlx_physical"].tolist(), [4, 6])

    def test_rename_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "toast.csv"
            csv.write_text("pid,item_1,item_2\n1,5,6\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not found"):
                load_input_file(str(csv), {"rename": {"item_3": "toast_3"}})
            with self.assertRaisesRegex(ValueError, "not found"):  # dropped first, then renamed
                load_input_file(str(csv), {"drop_columns": ["item_1"], "rename": {"item_1": "toast_1"}})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_input_file(str(csv), {"rename": {"item_1": "item_2"}})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_input_file(str(csv), {"rename": {"item_1": "t", "item_2": "t"}})
            with self.assertRaisesRegex(ValueError, "must map"):
                load_input_file(str(csv), {"rename": ["item_1"]})
            # A swap is not a clash.
            swapped = load_input_file(str(csv), {"rename": {"item_1": "item_2", "item_2": "item_1"}})
            self.assertEqual(swapped["item_1"].tolist(), [6])

    def test_two_files_of_one_source_keep_same_raw_names_apart(self):
        # NL2TAP: sus_responses.csv and toast_responses.csv both name their items item_1..
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "src").mkdir()
            pd.DataFrame({"participant_id": [1, 2, 3], "item_1": [4, 5, 3]}).to_csv(tmp / "src" / "sus.csv", index=False)
            pd.DataFrame({"participant_id": [1, 2, 3], "item_1": [6, 7, 5]}).to_csv(tmp / "src" / "toast.csv", index=False)
            (tmp / "map.yaml").write_text(yaml.safe_dump({"item_1": "SUS1", "toast_1": "trust_rating"}), encoding="utf-8")
            (tmp / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
                "source_id": "s", "source_type": "local_path", "location": str(tmp / "src"),
                "include_globs": ["*.csv"], "mapping_path": str(tmp / "map.yaml"),
                "table_options": {"toast.csv": {"rename": {"item_1": "toast_1"}}},
            }]}), encoding="utf-8")
            run_batch(tmp / "manifest.yaml", tmp / "out", SCHEMA, llm_deduction_enabled=False, cache_root=tmp / "cache")
            meta = pd.read_csv(tmp / "out" / "meta_view.csv").set_index("canonical_dv")
            self.assertEqual(meta.loc["SUS1", "dataset_id"], "sus.csv")
            self.assertAlmostEqual(meta.loc["SUS1", "mean"], 4.0)
            self.assertEqual(meta.loc["trust_rating", "dataset_id"], "toast.csv")
            self.assertEqual(meta.loc["trust_rating", "original_alias"], "toast_1")
            self.assertAlmostEqual(meta.loc["trust_rating", "mean"], 6.0)


def _qualtrics_rows() -> list[list[object]]:
    """A Qualtrics export: variable names, question texts, import ids, then answers."""
    return [
        ["Q7", "Q4", "Q5_1"],
        ["Enter Participant ID (number)", "Which Condition did you experience?", "Rate the following"],
        ['{"ImportId":"QID7_TEXT"}', '{"ImportId":"QID4"}', '{"ImportId":"QID5_1"}'],
        [1, "Baseline", 5],
        [1, "Medieval", 6],
        [2, "Baseline", 3],
    ]


class SkipRowsOptionTests(unittest.TestCase):
    def test_qualtrics_rows_are_skipped_and_columns_stay_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            book, csv = Path(tmp) / "export.xlsx", Path(tmp) / "export.csv"
            rows = _qualtrics_rows()
            pd.DataFrame(rows[1:], columns=rows[0]).to_excel(book, index=False)
            pd.DataFrame(rows[1:], columns=rows[0]).to_csv(csv, index=False)
            for path in (book, csv):
                plain = load_input_file(str(path))
                self.assertFalse(pd.api.types.is_numeric_dtype(plain["Q5_1"]))  # the defect
                df = load_input_file(str(path), {"skip_rows": [0, 1]})
                self.assertEqual(len(df), 3, path.name)
                self.assertTrue(pd.api.types.is_numeric_dtype(df["Q7"]), path.name)
                self.assertTrue(pd.api.types.is_numeric_dtype(df["Q5_1"]), path.name)
                self.assertEqual(df["Q5_1"].tolist(), [5, 6, 3])
                self.assertEqual(df["Q4"].tolist(), ["Baseline", "Medieval", "Baseline"])

    def test_rows_count_from_the_header_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "book.xlsx"
            rows = _qualtrics_rows()
            with pd.ExcelWriter(book) as writer:
                pd.DataFrame([["Study 2 questionnaire"]]).to_excel(writer, sheet_name="Data", index=False, header=False)
                pd.DataFrame(rows[1:], columns=rows[0]).to_excel(writer, sheet_name="Data", index=False, startrow=1)
                pd.DataFrame({f"raw{i}": range(9) for i in range(8)}).to_excel(writer, sheet_name="raw", index=False)
            df = load_input_file(str(book), {"sheet": "Data", "header": 1, "skip_rows": [0, 1]})
            self.assertEqual(list(df.columns), ["Q7", "Q4", "Q5_1"])
            self.assertEqual(df["Q5_1"].tolist(), [5, 6, 3])
            # Without a sheet the most data-rich sheet is still chosen.
            self.assertEqual(list(load_input_file(str(book), {"skip_rows": [0]}).columns)[:2], ["raw0", "raw1"])

    def test_skip_rows_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "t.csv"
            csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"skip_rows \[2\] not found"):
                load_input_file(str(csv), {"skip_rows": [0, 2]})
            for bad in (1, "0", [-1], [True], [0.0]):
                with self.assertRaisesRegex(ValueError, "must be a list"):
                    load_input_file(str(csv), {"skip_rows": bad})
            parquet = Path(tmp) / "t.parquet"
            pd.DataFrame({"a": [1, 2]}).to_parquet(parquet)
            with self.assertRaisesRegex(ValueError, "spreadsheets and delimited text"):
                load_input_file(str(parquet), {"skip_rows": [0]})
            self.assertEqual(len(load_input_file(str(csv), {"skip_rows": []})), 2)

    def test_batch_skips_the_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "src").mkdir()
            rows = _qualtrics_rows()
            pd.DataFrame(rows[1:], columns=rows[0]).to_excel(tmp / "src" / "export.xlsx", index=False)
            (tmp / "map.yaml").write_text(yaml.safe_dump(
                {"Q7": "meta_participant_id", "Q4": "meta_condition", "Q5_1": "engagement_level"}
            ), encoding="utf-8")
            (tmp / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
                "source_id": "s", "source_type": "local_path", "location": str(tmp / "src"),
                "include_globs": ["*.xlsx"], "mapping_path": str(tmp / "map.yaml"),
                "table_options": {"export.xlsx": {"skip_rows": [0, 1]}},
            }]}), encoding="utf-8")
            run_batch(tmp / "manifest.yaml", tmp / "out", SCHEMA, llm_deduction_enabled=False, cache_root=tmp / "cache")
            out = pd.read_excel(next((tmp / "out" / "standardized" / "s").glob("*-standardized.xlsx")))
            self.assertEqual(out["meta_participant_id"].tolist(), [1, 1, 2])
            self.assertEqual(out["meta_condition"].tolist(), ["Baseline", "Medieval", "Baseline"])
            meta = pd.read_csv(tmp / "out" / "meta_view.csv").set_index("canonical_dv")
            self.assertAlmostEqual(meta.loc["engagement_level", "mean"], 14 / 3)


class EncodingTests(unittest.TestCase):
    def test_stray_cp1252_byte_is_not_read_as_utf8(self):
        # charset-normalizer calls this sample "ascii"; UTF-8 cannot decode 0x92.
        sample = b"pid,comment\n" + b"1,fine\n" * 400 + b"2,I\x92ve never used it\n"
        encoding = detect_text_encoding(sample)
        self.assertEqual(sample.decode(encoding).splitlines()[-1], "2,I’ve never used it")

    def test_stray_byte_beyond_the_sample_is_reread(self):
        # The FOV/snap-turning data: a cp1252 apostrophe at byte 104,213 of 108,810.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fov.csv"
            path.write_bytes(b"ID,FMS0,comment\n" + b"1,3,none\n" * 5000 + b"2,4,I\x92ve also never used it\n")
            df = load_input_file(str(path))
            self.assertEqual(df["comment"].iloc[-1], "I’ve also never used it")
            self.assertEqual(len(df), 5001)

    def test_utf8_cut_at_the_sample_boundary_stays_utf8(self):
        # The 32 KB sample ends inside the two-byte "ü"; the file is valid UTF-8.
        head = b"a,b\n" + b"1,x\n" * ((32767 - 4) // 4)
        text = head + b"x" * (32767 - len(head)) + "ü,Grüße\n".encode("utf-8")
        self.assertEqual(codecs.lookup(detect_text_encoding(text)).name, "utf-8")


if __name__ == "__main__":
    unittest.main()
