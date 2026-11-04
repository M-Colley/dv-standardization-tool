"""Per-file table options from the manifest: sheet, header row and long-to-wide pivot."""

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
