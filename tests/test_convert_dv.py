import importlib.util
import json
import tempfile
import warnings
import unittest
from pathlib import Path

import pandas as pd

from scripts.convert_dv import (
    build_original_column_lookup,
    detect_single_file,
    export_with_metadata,
    load_schema,
    resolve_io_paths,
    standardize_columns,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"


class ConvertDVTests(unittest.TestCase):
    def test_case_insensitive_mapping_works(self):
        schema_data = load_schema(str(SCHEMA_PATH))
        mapping = schema_data["mapping"]

        df = pd.DataFrame({"Task_Completion_Time": [1.0], "foo": [2]})
        standardized = standardize_columns(df.copy(), mapping)

        self.assertEqual(standardized.columns.tolist()[0], "task_completion_time")
        self.assertEqual(standardized.columns.tolist()[1], "foo")

    def test_original_column_lookup_tracks_source_aliases(self):
        mapping = {
            "task_time": "task_completion_time",
            "tasktime": "task_completion_time",
        }
        df = pd.DataFrame({"task_time": [1], "tasktime": [2], "other": [3]})

        lookup = build_original_column_lookup(df, mapping)

        self.assertEqual(
            lookup["task_completion_time"],
            ["task_time", "tasktime"],
        )
        self.assertEqual(lookup["other"], ["other"])

    def test_detect_single_file_returns_none_when_multiple_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "a.csv").write_text("x\n1\n")
            (folder / "b.csv").write_text("x\n2\n")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = detect_single_file(folder, (".csv",), "input data")

            self.assertIsNone(result)
            self.assertTrue(any("Multiple input data files" in str(w.message) for w in caught))

    def test_resolve_io_paths_infers_files_and_default_output_in_folder_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "study.csv").write_text("Task_Completion_Time\n1\n")
            (folder / "mapping.yaml").write_text("dvs: []\n")

            resolved_input, resolved_output, resolved_schema = resolve_io_paths(
                str(folder),
                output_arg=None,
                schema_arg=None,
            )

            self.assertEqual(resolved_input.name, "study.csv")
            self.assertEqual(resolved_schema.name, "mapping.yaml")
            self.assertEqual(resolved_output.name, "study-standardized.csv")


    def test_resolve_io_paths_raises_on_multiple_input_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "study.csv").write_text("a\n1\n")
            (folder / "study2.xlsx").write_text("placeholder")

            with self.assertRaises(ValueError):
                resolve_io_paths(str(folder), output_arg=None, schema_arg=None)

    def test_resolve_io_paths_raises_on_multiple_schema_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "study.csv").write_text("a\n1\n")
            (folder / "s1.yaml").write_text("dvs: []\n")
            (folder / "s2.yml").write_text("dvs: []\n")

            with self.assertRaises(ValueError):
                resolve_io_paths(str(folder), output_arg=None, schema_arg=None)

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl not installed")
    def test_export_with_metadata_honors_xlsx_extension(self):
        df = pd.DataFrame({"task_completion_time": [1.2, 2.3]})
        meta = {
            "task_completion_time": {
                "category": "continuous",
                "needs_review": False,
                "original_name": ["Task_Completion_Time"],
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "standardized.xlsx"
            export_with_metadata(df, meta, str(output_path), schema_version="2.1")

            self.assertTrue(output_path.exists())
            sidecar = output_path.with_suffix("")
            sidecar_json = Path(str(sidecar) + "_metadata.json")
            self.assertTrue(sidecar_json.exists())

            parsed = json.loads(sidecar_json.read_text())
            self.assertEqual(parsed["summary"]["total_columns"], 1)


if __name__ == "__main__":
    unittest.main()
