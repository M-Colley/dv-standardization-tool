"""Regression tests for previously untested failure modes.

These pin the gaps found in the 2026-06 code review:

* the shipped schemas must pass their own validator (CI ran the validator,
  but no local test did — the standard schema was red for weeks);
* ``get_measurement_from_schema`` must work for EVERY schema DV (85/126
  entries used to raise ``KeyError`` via a missing ``SCORE`` enum member and
  a hard ``allowed_units`` subscript);
* the batch runner must be able to WRITE every format it DISCOVERS
  (``.json``/``.parquet``/... inputs used to fail at save time);
* assorted edge cases: trim-and-fill with no asymmetric studies,
  schema-suggestion merging mutating its input, silent long-to-wide
  aggregation, ``.git`` traversal during discovery, manifest validation
  raising ``SystemExit`` from library code.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.multi_study_analysis import (  # noqa: E402 — needs sys.path setup above
    _canonicalize_studies,
    compute_overlap,
    harmonized_summary,
    trim_and_fill,
)
from scripts.dv_inference import get_measurement_from_schema  # noqa: E402
from scripts.measurement_types import MeasurementCategory, MeasurementMeta  # noqa: E402
from scripts.reshape_utils import long_to_wide  # noqa: E402
from scripts.run_batch_standardization import (  # noqa: E402
    _find_repository_mapping_candidates,
    _match_files,
    _resolve_output_suffix,
    load_manifest,
    run_batch,
)
from scripts.schema_utils import update_schema_with_suggestions  # noqa: E402
from scripts.validate_schema import validate  # noqa: E402

SCHEMAS_DIR = REPO_ROOT / "schemas"
STANDARD_SCHEMA_PATH = SCHEMAS_DIR / "standard_dv_mapping.yaml"
SHIPPED_SCHEMAS = [
    "standard_dv_mapping.yaml",
    "standard_sensor_mapping.yaml",
    "standard_detection_mapping.yaml",
    "standard_metadata_mapping.yaml",
]


class ShippedSchemaIntegrityTests(unittest.TestCase):
    """Every schema we ship must pass the same validator CI runs."""

    def test_shipped_schemas_pass_validator(self):
        for name in SHIPPED_SCHEMAS:
            with self.subTest(schema=name):
                schema = yaml.safe_load(
                    (SCHEMAS_DIR / name).read_text(encoding="utf-8")
                )
                issues = validate(schema)
                self.assertEqual(
                    issues, [],
                    f"{name} failed its own validator: {issues[:5]}",
                )

    def test_every_standard_dv_resolves_measurement_metadata(self):
        schema = yaml.safe_load(STANDARD_SCHEMA_PATH.read_text(encoding="utf-8"))
        for dv in schema["dvs"]:
            if not isinstance(dv, dict) or not dv.get("measurement"):
                continue
            with self.subTest(dv=dv["id"]):
                meta = get_measurement_from_schema(dv["id"])
                self.assertIsInstance(meta, MeasurementMeta)
                self.assertIsInstance(meta.category, MeasurementCategory)
                self.assertIsInstance(meta.allowed_units, list)
                self.assertTrue(meta.primary_unit)

    def test_every_schema_category_is_a_known_enum_value(self):
        schema = yaml.safe_load(STANDARD_SCHEMA_PATH.read_text(encoding="utf-8"))
        known_values = {category.value for category in MeasurementCategory}
        for dv in schema["dvs"]:
            measurement = dv.get("measurement") if isinstance(dv, dict) else None
            if not measurement:
                continue
            with self.subTest(dv=dv["id"]):
                self.assertIn(measurement.get("category"), known_values)


class BatchOutputFormatTests(unittest.TestCase):
    """Discovered formats the writer cannot serialize fall back to CSV."""

    def test_resolve_output_suffix_passthrough_and_fallbacks(self):
        self.assertEqual(_resolve_output_suffix(".csv"), ".csv")
        self.assertEqual(_resolve_output_suffix(".TSV"), ".tsv")
        self.assertEqual(_resolve_output_suffix(".xlsx"), ".xlsx")
        self.assertEqual(_resolve_output_suffix(".pkl"), ".pkl")
        # pandas 2.x cannot write legacy .xls — upgraded to .xlsx
        self.assertEqual(_resolve_output_suffix(".xls"), ".xlsx")
        for unwritable in (".json", ".jsonl", ".parquet", ".sav", ".dta", ".feather", ".ods", ".xlsm"):
            self.assertEqual(_resolve_output_suffix(unwritable), ".csv", unwritable)

    def test_run_batch_standardizes_json_dataset_as_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "study"
            input_dir.mkdir()
            records = [
                {"UserID": i, "TLX1": i % 5, "taskTime": 10.0 + i}
                for i in range(6)
            ]
            (input_dir / "results.json").write_text(json.dumps(records), encoding="utf-8")

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "json_study",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["**/*.json"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "out"
            summary = run_batch(
                manifest_path,
                output_dir,
                STANDARD_SCHEMA_PATH,
                llm_deduction_enabled=False,
            )

            result = summary["results"][0]
            self.assertEqual(result["status"], "completed", result.get("message"))
            self.assertEqual(result["processed_files"], 1)
            standardized = list((output_dir / "standardized" / "json_study").glob("*-standardized.csv"))
            self.assertEqual(len(standardized), 1, "JSON input must be written back as CSV")
            out_df = pd.read_csv(standardized[0])
            # taskTime is a schema alias of task_completion_time
            self.assertIn("task_completion_time", out_df.columns)


class ManifestValidationTests(unittest.TestCase):
    def test_invalid_manifest_raises_value_error_not_system_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "bad id with spaces",
                                "source_type": "local_path",
                                "location": tmpdir,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_manifest(manifest_path)


class DiscoveryPruningTests(unittest.TestCase):
    def test_match_files_prunes_git_and_matches_top_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "top.csv").write_text("a\n1\n", encoding="utf-8")
            nested = root / "sub" / "deeper"
            nested.mkdir(parents=True)
            (nested / "inner.csv").write_text("a\n1\n", encoding="utf-8")
            git_dir = root / ".git" / "objects"
            git_dir.mkdir(parents=True)
            (git_dir / "tracked.csv").write_text("a\n1\n", encoding="utf-8")

            matched = _match_files(root, ["**/*.csv"], None)

        names = sorted(p.relative_to(root).as_posix() for p in matched)
        self.assertEqual(names, ["sub/deeper/inner.csv", "top.csv"])

    def test_mapping_candidates_skip_git_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "study_mapping.yaml").write_text("a: b\n", encoding="utf-8")
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "stale_mapping.yaml").write_text("a: b\n", encoding="utf-8")

            candidates = _find_repository_mapping_candidates(root)

        self.assertEqual([c.name for c in candidates], ["study_mapping.yaml"])


class TrimAndFillEdgeTests(unittest.TestCase):
    def test_identical_means_do_not_crash_and_impute_nothing(self):
        summary = pd.DataFrame(
            {
                "study": ["a", "b", "c"],
                "dv": ["x", "x", "x"],
                "n": [20, 20, 20],
                "mean": [5.0, 5.0, 5.0],
                "sd": [1.0, 1.0, 1.0],
            }
        )
        result = trim_and_fill(summary, "x")
        self.assertIsNotNone(result)
        self.assertEqual(result["k_imputed"], 0)
        self.assertAlmostEqual(result["adjusted_mean"], result["original_mean"])
        self.assertTrue(np.isfinite(result["adjusted_se"]))


class CanonicalStudiesParameterTests(unittest.TestCase):
    """Passing precomputed canonical frames must not change results."""

    def _make_studies(self) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(7)
        return {
            "study_a": pd.DataFrame(
                {
                    "mental_demand": rng.integers(0, 20, 30),
                    "task_completion_time": rng.normal(12, 2, 30),
                }
            ),
            "study_b": pd.DataFrame(
                {
                    "mental_demand": rng.integers(0, 20, 25),
                    "error_rate": rng.uniform(0, 30, 25),
                }
            ),
        }

    def test_harmonized_summary_equivalence(self):
        studies = self._make_studies()
        canonical = _canonicalize_studies(studies)
        on_demand = harmonized_summary(studies)
        precomputed = harmonized_summary(studies, canonical_studies=canonical)
        pd.testing.assert_frame_equal(
            on_demand.reset_index(drop=True),
            precomputed.reset_index(drop=True),
        )

    def test_compute_overlap_equivalence(self):
        studies = self._make_studies()
        canonical = _canonicalize_studies(studies)
        pd.testing.assert_frame_equal(
            compute_overlap(studies),
            compute_overlap(studies, canonical_studies=canonical),
        )


class SchemaSuggestionMergeTests(unittest.TestCase):
    def test_update_schema_with_suggestions_does_not_mutate_input(self):
        existing = {
            "dvs": [
                {"id": "trust_rating", "aliases": ["trust"]},
            ]
        }
        snapshot = json.loads(json.dumps(existing))

        updated = update_schema_with_suggestions(
            existing,
            {"trust_rating": ["trust_score"], "new_dv": ["brand_new"]},
        )

        self.assertEqual(existing, snapshot, "input schema must not be mutated")
        self.assertIn("trust_score", updated["dvs"][0]["aliases"])
        self.assertEqual(len(updated["dvs"]), 2)


class LongToWideAggregationWarningTests(unittest.TestCase):
    def test_duplicate_observations_emit_user_warning(self):
        df = pd.DataFrame(
            {
                "pid": [1, 1, 2],
                "measure": ["trust", "trust", "trust"],
                "score": [4.0, 6.0, 3.0],
            }
        )
        with self.assertWarns(UserWarning):
            wide = long_to_wide(df, id_col="pid", variable_col="measure", value_col="score")
        # Duplicates are mean-aggregated
        self.assertAlmostEqual(float(wide.loc[wide["pid"] == 1, "trust"].iloc[0]), 5.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
