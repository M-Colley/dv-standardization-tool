"""Direct unit tests for the extracted scripts.llm_coordinator module.

These tests import from ``scripts.llm_coordinator`` directly so they
fail loudly if the standalone module ever stops being importable on its
own. Behavioural coverage of ``_augment_mapping_with_llm_deductions``
through the orchestrator already lives in
test_run_batch_standardization (the LLM-augmentation tests there
exercise the same code via the run_batch entry point).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.llm_coordinator import (
    _augment_mapping_with_llm_deductions,
    _build_llm_deduction_log_lines,
    _collect_llm_deduction,
    _finalize_llm_deductions,
    _score_alias_match,
    _select_llm_candidate_shortlist,
)


class ScoreAliasMatchTests(unittest.TestCase):
    def test_returns_zero_for_empty_input(self):
        self.assertEqual(_score_alias_match("", "duration"), 0.0)
        self.assertEqual(_score_alias_match("duration", ""), 0.0)

    def test_returns_zero_for_short_alias_not_in_raw(self):
        # "u1" is shorter than 4 chars and is not a substring of
        # "completion_time" → no fuzzy match attempted.
        self.assertEqual(_score_alias_match("completion_time", "u1"), 0.0)

    def test_short_alias_substring_still_scores(self):
        # "eda" appears in "eda_signal" → fuzzy match runs.
        self.assertGreater(_score_alias_match("eda_signal", "eda"), 0.0)

    def test_high_score_for_identical_strings(self):
        self.assertGreaterEqual(_score_alias_match("task_time", "task_time"), 99.0)

    def test_falls_back_to_difflib_when_rapidfuzz_raises(self):
        # Force the rapidfuzz import to raise; difflib fallback must still
        # produce a numeric score.
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "rapidfuzz":
                raise ImportError("simulated")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            score = _score_alias_match("task_time", "task_completion_time")
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)


class SelectLlmCandidateShortlistTests(unittest.TestCase):
    def test_returns_empty_for_empty_mapping(self):
        candidates, top_score = _select_llm_candidate_shortlist({}, "anything")
        self.assertEqual(candidates, [])
        self.assertEqual(top_score, 0.0)

    def test_caps_at_max_candidates(self):
        # Build a mapping with 12 distinct canonicals; max=3 should trim to 3.
        mapping = {f"alias_{i}": f"canon_{i}" for i in range(12)}
        candidates, _ = _select_llm_candidate_shortlist(
            mapping, raw_column_name="alias_3", max_candidates=3
        )
        self.assertEqual(len(candidates), 3)

    def test_skips_non_string_entries(self):
        # rogue non-string keys/values must be silently ignored.
        mapping: dict = {"valid_alias": "valid_canonical", 42: "x", "y": 99}
        candidates, _ = _select_llm_candidate_shortlist(
            mapping, raw_column_name="valid_alias"
        )
        self.assertIn("valid_canonical", candidates)


class AugmentMappingTests(unittest.TestCase):
    def test_inferred_alias_is_added_in_both_cases(self):
        mapping = {
            "task_time": "task_completion_time",
            "duration": "task_completion_time",
            "task_completion_time": "task_completion_time",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "scripts.llm_coordinator.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                augmented = _augment_mapping_with_llm_deductions(
                    mapping,
                    columns=["task_time", "NovelDuration"],
                    source_root=Path(tmp),
                )
        # Original key + lowercase variant both get the inferred canonical.
        self.assertEqual(augmented["NovelDuration"], "task_completion_time")
        self.assertEqual(augmented["novelduration"], "task_completion_time")
        # Already-mapped column does not retrigger the LLM call.
        mocked.assert_called_once()

    def test_inference_cache_prevents_duplicate_llm_calls(self):
        mapping = {
            "task_time": "task_completion_time",
            "duration": "task_completion_time",
            "task_completion_time": "task_completion_time",
        }
        cache: dict[str, str | None] = {}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "scripts.llm_coordinator.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                _augment_mapping_with_llm_deductions(
                    mapping,
                    columns=["NovelDuration"],
                    source_root=Path(tmp),
                    inference_cache=cache,
                )
                _augment_mapping_with_llm_deductions(
                    mapping,
                    columns=["NovelDuration"],
                    source_root=Path(tmp),
                    inference_cache=cache,
                )
        # Second call should be served from cache, not from the LLM.
        self.assertEqual(mocked.call_count, 1)
        self.assertIn("novelduration", cache)

    def test_low_score_skips_llm_call(self):
        # When no candidate scores above min_attempt_score, the LLM is
        # never invoked for that alias.
        mapping = {"foo": "alpha"}  # nothing remotely close to "z9"
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "scripts.llm_coordinator.deduce_standard_name_with_local_llm",
                return_value="alpha",
            ) as mocked:
                _augment_mapping_with_llm_deductions(
                    mapping,
                    columns=["z9"],
                    source_root=Path(tmp),
                    min_attempt_score=99.0,
                )
        mocked.assert_not_called()


class CollectAndFinalizeLlmDeductionTests(unittest.TestCase):
    def test_dedupes_same_alias_across_datasets(self):
        store: dict[tuple[str, str], dict] = {}
        _collect_llm_deduction(store, "src", "ds1", "TaskTime", "task_completion_time")
        _collect_llm_deduction(store, "src", "ds2", "tasktime", "task_completion_time")
        # Same alias (case-insensitive) within the same source → one entry,
        # two datasets.
        self.assertEqual(len(store), 1)
        entry = next(iter(store.values()))
        self.assertEqual(sorted(entry["datasets"]), ["ds1", "ds2"])

    def test_skips_blank_alias(self):
        store: dict[tuple[str, str], dict] = {}
        _collect_llm_deduction(store, "src", "ds1", "   ", "x")
        self.assertEqual(store, {})

    def test_finalize_sorts_datasets_and_entries(self):
        store: dict[tuple[str, str], dict] = {}
        _collect_llm_deduction(store, "src_b", "z", "BAlias", "x")
        _collect_llm_deduction(store, "src_a", "y", "AAlias", "x")
        _collect_llm_deduction(store, "src_a", "x", "AAlias", "x")  # add second dataset
        finalized = _finalize_llm_deductions(store)
        # Outer order: by source_id then alias.
        self.assertEqual(finalized[0]["source_id"], "src_a")
        self.assertEqual(finalized[1]["source_id"], "src_b")
        # Inner datasets sorted alphabetically.
        self.assertEqual(finalized[0]["datasets"], ["x", "y"])


class BuildLlmDeductionLogLinesTests(unittest.TestCase):
    def test_empty_returns_no_op_message(self):
        lines = _build_llm_deduction_log_lines([])
        self.assertEqual(lines, ["No LLM-derived mappings were applied in this run."])

    def test_renders_one_line_per_entry(self):
        lines = _build_llm_deduction_log_lines(
            [
                {
                    "source_id": "src",
                    "alias": "TaskTime",
                    "canonical_dv": "task_completion_time",
                    "datasets": ["ds1", "ds2"],
                }
            ]
        )
        self.assertEqual(lines[0], "LLM-derived mappings applied:")
        self.assertEqual(
            lines[1],
            "[src] TaskTime -> task_completion_time (datasets: ds1, ds2)",
        )

    def test_handles_missing_datasets_field(self):
        lines = _build_llm_deduction_log_lines(
            [{"source_id": "src", "alias": "X", "canonical_dv": "y"}]
        )
        self.assertIn("datasets: n/a", lines[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
