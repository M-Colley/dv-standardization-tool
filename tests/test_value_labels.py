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



class NegativeAnswerCodeTests(unittest.TestCase):
    def test_limesurvey_negative_codes_become_negative_numbers(self):
        from scripts.value_labels import recode_negative_answer_codes

        df = pd.DataFrame({
            "aoa1": ["1", "neg2", "0", None, "NEG1"],
            "comment": ["neg1 is bad", "fine", "ok", "x", "y"],
        })
        out, report = recode_negative_answer_codes(df)
        self.assertEqual(out["aoa1"].tolist()[:3], [1.0, -2.0, 0.0])
        self.assertEqual(out["aoa1"].iloc[4], -1.0)
        self.assertEqual(report, {"aoa1": 2})
        self.assertEqual(out["comment"].tolist(), df["comment"].tolist())

    def test_suffix_negative_codes_become_negative_numbers(self):
        """eHMI-for-All exports -3..+3 items as "1neg", "2neg", "3neg"."""
        from scripts.value_labels import recode_negative_answer_codes

        df = pd.DataFrame({
            "PerSafe01": ["2", "1neg", "0", "3NEG", None, "2neg"],
            "id": ["a1", "b2", "c", "d", "e", "f"],
        })
        out, report = recode_negative_answer_codes(df)
        self.assertEqual(out["PerSafe01"].tolist()[:4], [2.0, -1.0, 0.0, -3.0])
        self.assertEqual(out["PerSafe01"].tolist()[5], -2.0)
        self.assertEqual(report, {"PerSafe01": 3})
        self.assertEqual(out["id"].tolist(), df["id"].tolist())

    def test_batch_keeps_negative_answers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data = tmp / "src"
            data.mkdir()
            pd.DataFrame({"trust_rating": ["2", "neg2", "neg1", "1"]}).to_csv(data / "s.csv", index=False)
            (tmp / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
                "source_id": "s", "source_type": "local_path", "location": str(data), "include_globs": ["*.csv"],
            }]}), encoding="utf-8")
            run_batch(tmp / "manifest.yaml", tmp / "out", SCHEMA, llm_deduction_enabled=False,
                      cache_root=tmp / "cache")
            meta = pd.read_csv(tmp / "out" / "meta_view.csv")
            row = meta[meta["canonical_dv"] == "trust_rating"].iloc[0]
            self.assertEqual(row["n"], 4)
            self.assertAlmostEqual(row["mean"], 0.0)


class AoaCodingTests(unittest.TestCase):
    def test_aoa_scores_do_not_depend_on_position_coding(self):
        sys.path.insert(0, str(REPO_ROOT))
        from analyses.multi_study_analysis import add_derived_scale_scores

        positions = pd.DataFrame({f"aoa{i}": [1, 2, 3, 4, 5, 2] for i in range(1, 10)})
        centred = positions - 3  # the same answers exported as -2..+2
        a = add_derived_scale_scores(positions)
        b = add_derived_scale_scores(centred)
        pd.testing.assert_series_equal(a["aoa_usefulness"], b["aoa_usefulness"])
        pd.testing.assert_series_equal(a["aoa_satisfying"], b["aoa_satisfying"])
        self.assertTrue(a["aoa_usefulness"].between(-2, 2).all())


if __name__ == "__main__":
    unittest.main()


class ReverseKeyingTests(unittest.TestCase):
    def test_declared_items_are_reversed_within_their_range(self):
        from scripts.value_labels import apply_reverse_keying

        df = pd.DataFrame({"C2": [1, 5, 3, 9, None], "ST2": [1, 7, 4, 2, 6], "other": [1, 2, 3, 4, 5]})
        out, report = apply_reverse_keying(df, [
            {"columns": ["C2"], "scale_range": [1, 5], "reason": "C2_inv"},
            {"columns": ["ST2", "missing"], "scale_range": [1, 7]},
        ])
        self.assertEqual(out["C2"].tolist()[:4], [5, 1, 3, 9])  # 9 is outside 1..5: kept and counted
        self.assertEqual(out["ST2"].tolist(), [7, 1, 4, 6, 2])
        self.assertEqual(out["other"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(report["C2"]["reversed"], 3)
        self.assertEqual(report["C2"]["out_of_range"], 1)
        self.assertNotIn("missing", report)
        self.assertEqual(df["C2"].tolist()[0], 1)  # input untouched

    def test_no_specs_returns_the_same_frame(self):
        from scripts.value_labels import apply_reverse_keying

        df = pd.DataFrame({"a": [1]})
        out, report = apply_reverse_keying(df, [])
        self.assertIs(out, df)
        self.assertEqual(report, {})

