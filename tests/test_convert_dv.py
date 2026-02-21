import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.convert_dv import (
    build_original_column_lookup,
    export_with_metadata,
    load_schema,
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
