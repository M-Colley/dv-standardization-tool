"""Tests for the LLM plausibility check (scripts/llm_plausibility.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import llm_plausibility as lp  # noqa: E402
from scripts.run_batch_standardization import run_batch  # noqa: E402

SCHEMA = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"


class NameRuleTests(unittest.TestCase):
    def test_tokenizer_splits_camel_case_acronyms_and_digits(self):
        self.assertEqual(lp.name_tokens("probandenIDTime"), ["probanden", "id", "time"])
        self.assertEqual(lp.name_tokens("PId"), ["p", "id"])
        self.assertEqual(lp.name_tokens("attention_check001"), ["attention", "check", "001"])
        self.assertEqual(lp.name_tokens("TimeOfTouchReal"), ["time", "of", "touch", "real"])
        self.assertEqual(lp.name_tokens("Start-Sprache"), ["start", "sprache"])

    def test_non_dv_columns_are_rejected_with_a_reason(self):
        cases = {
            "age": "demographic",
            "Gender": "demographic",
            "Start-Sprache": "demographic",
            "PId": "identifier",
            "Antwort ID": "identifier",
            "Zufallsgeneratorstartwert": "randomization",
            "seed": "randomization",
            "attention_check001": "quality_control",
            "Datum gestartet": "timestamp",
            "Time": "event_time",
            "TimeOfTouch": "event_time",
            "interviewtime": "survey_timing",
            "Fragenzeit: feedbackPositive": "survey_timing",
            "TouchX": "coordinate",
            "SpeedCondition": "experimental_condition",
            "IsBumpVirtual": "experimental_condition",
            "ranking2": "ranking",
        }
        for column, reason in cases.items():
            with self.subTest(column=column):
                self.assertEqual(lp.name_rejection_reason(column), reason)

    def test_plausible_dv_names_pass(self):
        for column in ["Trust1", "PerceivedSafety3", "tlx1_sq001", "ReactionTime", "DurationX",
                       "perceivedSafety_score", "usage", "update_rate", "SART2", "valid_trials",
                       "facilitating_conditions"]:
            with self.subTest(column=column):
                self.assertIsNone(lp.name_rejection_reason(column))

    def test_limesurvey_timing_columns_need_a_limesurvey_export(self):
        survey = ["lastpage", "tlx1_sq001", "TLX1Time", "trustiaunderp_sq001", "TrustiAUnderPTime"]
        self.assertEqual(lp.name_rejection_reason("TLX1Time", survey), "survey_timing")
        self.assertEqual(lp.name_rejection_reason("TrustiAUnderPTime", survey), "survey_timing")
        # Same naming outside a LimeSurvey export is a plausible DV.
        self.assertIsNone(lp.name_rejection_reason("ReactionTime", ["ReactionCorrect", "ReactionTime"]))


class ProfileAndRangeTests(unittest.TestCase):
    def test_profiles_detect_text_decimal_commas_and_empty_columns(self):
        df = pd.DataFrame({
            "feedback": ["gut", "zu schnell", "ok", "5"],
            "decimal_comma": ["3,5", "4,0", "2,5", None],
            "empty": [np.nan] * 4,
            "score": [1, 2, 3, 4],
        })
        profiles = lp.profile_columns(df, df.columns)
        self.assertEqual(lp.data_rejection_reason(profiles["feedback"]), "non_numeric")
        self.assertIsNone(lp.data_rejection_reason(profiles["decimal_comma"]))
        self.assertEqual(profiles["decimal_comma"].maximum, 4.0)
        self.assertEqual(lp.data_rejection_reason(profiles["empty"]), "no_values")
        self.assertEqual((profiles["score"].minimum, profiles["score"].maximum), (1.0, 4.0))

    def test_suggestions_outside_every_plausible_scale_are_rejected(self):
        ranges = {"trust_rating": (1.0, 7.0)}
        likert = lp.profile_column(pd.Series([1, 4, 7]))
        vas = lp.profile_column(pd.Series([0, 55, 100]))
        seed = lp.profile_column(pd.Series([854012000, 912345678]))
        self.assertIsNone(lp.check_llm_suggestion("t", "trust_rating", likert, ranges))
        self.assertIsNone(lp.check_llm_suggestion("t", "trust_rating", vas, ranges))
        rejection = lp.check_llm_suggestion("seed", "trust_rating", seed, ranges)
        self.assertEqual(rejection["reason"], "out_of_range")
        self.assertEqual(rejection["stage"], lp.STAGE_POST_LLM)
        # DVs without a canonical range (e.g. times) cannot be range-checked.
        self.assertIsNone(lp.check_llm_suggestion("x", "task_completion_time", seed, ranges))

    def test_canonical_ranges_come_from_the_dv_schema(self):
        ranges = lp.canonical_ranges_from_schema(yaml.safe_load(SCHEMA.read_text(encoding="utf-8")))
        self.assertEqual(ranges["mental_demand"], (0.0, 20.0))
        self.assertIn("trust_rating", ranges)
        self.assertNotIn("task_completion_time", ranges)


class BatchIntegrationTests(unittest.TestCase):
    def _run(self, tmp: Path, name: str, **kwargs):
        input_dir = tmp / "input"
        if not input_dir.exists():
            input_dir.mkdir()
            n = 12
            rng = np.random.default_rng(1)
            pd.DataFrame({
                "age": rng.integers(19, 60, n),
                "Zufallsgeneratorstartwert": rng.integers(10**8, 10**9, n),
                "comment": [f"answer {i}" for i in range(n)],
                "MysteryLikert": rng.integers(1, 8, n),
                "MysteryHuge": rng.integers(10**5, 10**6, n),
            }).to_csv(input_dir / "survey.csv", index=False)
            (tmp / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
                "source_id": "survey", "source_type": "local_path",
                "location": str(input_dir), "include_globs": ["*.csv"],
            }]}), encoding="utf-8")
        calls: list[str] = []

        def fake_llm(raw_column_name, canonical_candidates, **_kwargs):
            calls.append(raw_column_name)
            return "trust_rating"

        with mock.patch("scripts.llm_coordinator.deduce_standard_name_with_local_llm", side_effect=fake_llm), \
                mock.patch("scripts.run_batch_standardization.collect_repository_context", return_value=""), \
                mock.patch("scripts.llm_coordinator._select_llm_candidate_shortlist",
                           return_value=(["trust_rating"], 100.0)):
            summary = run_batch(tmp / "manifest.yaml", tmp / name, SCHEMA, cache_root=tmp / "cache", **kwargs)
        return summary, calls

    def test_check_filters_llm_input_and_output_and_logs_rejections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary, calls = self._run(tmp, "checked")
            # Demographic, seed and free-text columns never reach the model.
            self.assertEqual(sorted(calls), ["MysteryHuge", "MysteryLikert"])
            deductions = json.loads((tmp / "checked" / "llm_deductions.json").read_text(encoding="utf-8"))
            self.assertEqual([d["alias"] for d in deductions], ["MysteryLikert"])
            rejected = json.loads(
                (tmp / "checked" / "llm_rejected_suggestions.json").read_text(encoding="utf-8")
            )
            reasons = {r["alias"]: (r["stage"], r["reason"]) for r in rejected}
            self.assertEqual(reasons, {
                "age": ("pre_llm", "demographic"),
                "Zufallsgeneratorstartwert": ("pre_llm", "randomization"),
                "comment": ("pre_llm", "non_numeric"),
                "MysteryHuge": ("post_llm", "out_of_range"),
            })
            self.assertEqual(summary["llm_rejected_count"], 4)
            self.assertTrue(summary["llm_plausibility_check"])
            meta = pd.read_csv(tmp / "checked" / "meta_view.csv")
            self.assertEqual(set(meta.loc[meta["mapping_method"] == "llm", "original_alias"]), {"MysteryLikert"})
            log_text = (tmp / "checked" / "llm_deductions.log").read_text(encoding="utf-8")
            self.assertIn("Rejected by the LLM plausibility check (4)", log_text)

    def test_check_can_be_disabled_for_ablation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary, calls = self._run(Path(tmpdir), "unchecked", llm_plausibility_check=False)
        self.assertEqual(len(calls), 5)
        self.assertEqual(summary["llm_deductions_count"], 5)
        self.assertEqual(summary["llm_rejected_count"], 0)


if __name__ == "__main__":
    unittest.main()
