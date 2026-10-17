"""Tests for declarative text-label recoding (scripts/value_labels.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_dv import load_schema  # noqa: E402
from scripts.run_batch_standardization import run_batch  # noqa: E402
from scripts.value_labels import apply_value_labels  # noqa: E402

SCHEMA = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"
LABELS = [{
    "columns": ["Q1", "Q2"],
    "labels": {"Do not agree at all": 1, "Neither": 3, "Fully agree": 5},
}]


class ApplyValueLabelsTests(unittest.TestCase):
    def test_labels_become_codes_case_and_space_insensitively(self):
        df = pd.DataFrame({
            "Q1": ["Fully agree", "  neither ", "DO NOT AGREE AT ALL", None],
            "Q2": [5, "Neither", "Maybe", 1],
            "other": ["x", "y", "z", "w"],
        })
        out, report = apply_value_labels(df, LABELS)
        self.assertEqual(out["Q1"].tolist()[:3], [5.0, 3.0, 1.0])
        self.assertTrue(pd.isna(out["Q1"].iloc[3]))
        self.assertEqual(out["Q2"].tolist()[:2], [5.0, 3.0])
        self.assertTrue(pd.isna(out["Q2"].iloc[2]))  # undeclared label -> missing, counted
        self.assertEqual(report["Q1"], {"recoded": 3, "kept_numeric": 0, "unmatched": 0})
        self.assertEqual(report["Q2"], {"recoded": 1, "kept_numeric": 2, "unmatched": 1})
        self.assertEqual(out["other"].tolist(), ["x", "y", "z", "w"])
        self.assertEqual(df["Q1"].iloc[0], "Fully agree")  # input left untouched

    def test_no_labels_returns_the_same_frame(self):
        df = pd.DataFrame({"Q1": ["a"]})
        out, report = apply_value_labels(df, [])
        self.assertIs(out, df)
        self.assertEqual(report, {})

    def test_value_labels_key_is_not_treated_as_an_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "m.yaml"
            path.write_text(yaml.safe_dump({"Q1": "trust_rating", "value_labels": LABELS}), encoding="utf-8")
            mapping = load_schema(str(path))["mapping"]
        self.assertNotIn("value_labels", mapping)
        self.assertEqual(mapping["Q1"], "trust_rating")


class BatchValueLabelTests(unittest.TestCase):
    def test_declared_labels_make_text_likert_items_numeric_dvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data = tmp / "src"
            data.mkdir()
            pd.DataFrame({
                "ID": [1, 2, 3, 4],
                "Q1": ["Fully agree", "Neither", "Do not agree at all", "Fully agree"],
            }).to_csv(data / "survey.csv", index=False)
            (data / "survey_mapping.yaml").write_text(
                yaml.safe_dump({"Q1": "trust_rating", "value_labels": LABELS}), encoding="utf-8"
            )
            (tmp / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
                "source_id": "survey", "source_type": "local_path", "location": str(data),
                "include_globs": ["*.csv"],
            }]}), encoding="utf-8")
            run_batch(tmp / "manifest.yaml", tmp / "out", SCHEMA, llm_deduction_enabled=False,
                      cache_root=tmp / "cache")
            meta = pd.read_csv(tmp / "out" / "meta_view.csv")
            trust = meta[meta["canonical_dv"] == "trust_rating"].iloc[0]
            self.assertEqual(trust["n"], 4)
            self.assertAlmostEqual(trust["mean"], 3.5)
            quality = json.loads((tmp / "out" / "standardized" / "survey" / "survey_csv-quality.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(quality["value_labels_applied"]["Q1"]["recoded"], 4)


if __name__ == "__main__":
    unittest.main()
