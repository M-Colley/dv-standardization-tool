"""Regression tests for the mapping-coverage / analysis-honesty fixes.

Locks in behaviors introduced while lifting schema coverage on the example
catalog (ROADS / eHMI-for-All):

- structured delimiters beat whitespace, and "; "-delimited logs are tidied
  (no phantom ``Unnamed: N`` columns, stripped names) — scripts/data_loaders
- simulator/VR telemetry routes to ``sensor_stream``; crossing-event summaries
  stay ``results_table`` — scripts/batch_profiles + dataset_type_profiles.yaml
- junk trees/manifests are excluded from discovery, and ambiguous suffixes
  (.json family) need an explicit include glob — run_batch_standardization
- shared never-map blocklist YAML is wired into both pipeline layers
- PerSafe/TiA item batteries derive perceived_safety / trust_rating composites
  without overwriting native columns — analyses/multi_study_analysis
- the PCA composite excludes studies that observe none of the shared DVs
- LiNGAM refuses the synthetic-Cholesky fallback by default
- load_studies header-prefilters DV-less files and keeps empty study frames
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.multi_study_analysis import (
    ID_LIKE_COLUMNS,
    add_derived_scale_scores,
    build_composite_index,
    discover_causal_structure,
    load_studies,
)
from scripts.batch_profiles import classify_dataset_type
from scripts.data_loaders import _detect_delimiter, load_input_file
from scripts.run_batch_standardization import (
    NEVER_MAP_NORMALIZED_COLUMNS,
    _match_files,
)


class DelimiterPreferenceTests(unittest.TestCase):
    def test_semicolon_space_files_split_on_semicolon(self):
        sample = "a; b; c\n1; 2; 3\n4; 5; 6\n"
        self.assertEqual(_detect_delimiter(sample), ";")

    def test_whitespace_only_files_still_fall_back_to_space(self):
        sample = "a b c\n1 2 3\n4 5 6\n"
        self.assertEqual(_detect_delimiter(sample), " ")


class TextTableTidyTests(unittest.TestCase):
    def test_trailing_delimiter_phantom_column_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "log.csv"
            p.write_text("UserID;ScenarioID;vehicleSpeed;\n1;2;13.4;\n1;2;13.9;\n", encoding="utf-8")
            df = load_input_file(p)
        self.assertEqual(list(df.columns), ["UserID", "ScenarioID", "vehicleSpeed"])

    def test_semicolon_space_column_names_are_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "eye.csv"
            p.write_text("X Filtered Pixel; Y Filtered Pixel; Blinking\n1; 2; 0\n", encoding="utf-8")
            df = load_input_file(p)
        self.assertEqual(list(df.columns), ["X Filtered Pixel", "Y Filtered Pixel", "Blinking"])

    def test_real_unnamed_index_column_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "idx.csv"
            p.write_text("Unnamed: 0,score\n0,5\n1,7\n", encoding="utf-8")
            df = load_input_file(p)
        self.assertIn("Unnamed: 0", df.columns)


class TelemetryRoutingTests(unittest.TestCase):
    def test_roads_simulator_log_routes_to_sensor_stream(self):
        cols = [
            "UserID", "ScenarioID", "controlMode", "elapsedTimeSinceAccess",
            "vehiclePosition", "vehicleSpeed", "currentLaneDeviation",
            "distanceToEnd", "blindTimeSum",
        ]
        self.assertEqual(classify_dataset_type(Path("logs/log_AS21_1.csv"), cols), "sensor_stream")

    def test_ehmi_vr_trajectory_routes_to_sensor_stream(self):
        cols = [
            "user_id", "scenario", "distraction", "ehmi", "timestamp",
            "userPositionX", "userPositionY", "userPositionZ",
            "playerPositon", "time", "focused object",
        ]
        self.assertEqual(classify_dataset_type(Path("data/p01.csv"), cols), "sensor_stream")

    def test_ehmi_crossing_summary_stays_results_table(self):
        cols = [
            "user_id", "scenario", "distraction", "ehmi",
            "collision with cars", "time before", "time on intersection",
            "time after", "total time",
        ]
        self.assertEqual(classify_dataset_type(Path("data/p01.csv"), cols), "results_table")


class DiscoveryFilterTests(unittest.TestCase):
    def _make_tree(self, tmp: str) -> Path:
        base = Path(tmp)
        (base / "data").mkdir()
        (base / "data" / "results.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (base / "Library" / "PackageCache" / "pkg").mkdir(parents=True)
        (base / "Library" / "PackageCache" / "pkg" / "package.json").write_text("{}", encoding="utf-8")
        (base / "catboost_info").mkdir()
        (base / "catboost_info" / "training.json").write_text("{}", encoding="utf-8")
        (base / "records.json").write_text('[{"a": 1}]', encoding="utf-8")
        (base / "notes.txt").write_text("hello", encoding="utf-8")
        return base

    def test_default_discovery_excludes_json_and_junk_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._make_tree(tmp)
            files = _match_files(base, None, None)
        self.assertEqual([f.name for f in files], ["results.csv"])

    def test_explicit_include_glob_revives_json_but_not_excluded_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._make_tree(tmp)
            files = _match_files(base, ["**/*.json"], None)
        self.assertEqual([f.name for f in files], ["records.json"])


class SharedBlocklistTests(unittest.TestCase):
    def test_ehmi_independent_variables_are_never_mapped(self):
        for alias in ("scenario", "distraction", "ehmi", "scenarioid", "controlmode"):
            self.assertIn(alias, NEVER_MAP_NORMALIZED_COLUMNS)

    def test_analysis_id_like_columns_loaded_from_shared_yaml(self):
        self.assertIn("participant_id", ID_LIKE_COLUMNS)
        self.assertIn("user_id", ID_LIKE_COLUMNS)


class DerivedCompositeTests(unittest.TestCase):
    def test_tia_and_persafe_items_derive_composites(self):
        df = pd.DataFrame({
            "tia_item_01": [3, 4], "tia_item_02": [4, 4], "tia_item_03": [2, 5],
            "tia_item_04": [3, 3], "tia_item_05": [4, 2], "tia_item_06": [5, 4],
            "perceived_safety_item_01": [1, -2], "perceived_safety_item_02": [2, 0],
            "perceived_safety_item_03": [0, 1], "perceived_safety_item_04": [-1, 2],
        })
        out = add_derived_scale_scores(df)
        self.assertAlmostEqual(out["trust_rating"].iloc[0], 3.5, places=6)
        self.assertAlmostEqual(out["perceived_safety"].iloc[0], 0.5, places=6)

    def test_native_column_is_not_overwritten_by_derived_mean(self):
        df = pd.DataFrame({
            "trust_rating": [9.0],
            "tia_item_01": [1], "tia_item_02": [1], "tia_item_03": [1],
            "tia_item_04": [1], "tia_item_05": [1], "tia_item_06": [1],
        })
        out = add_derived_scale_scores(df)
        self.assertEqual(out["trust_rating"].tolist(), [9.0])


class CompositeExclusionTests(unittest.TestCase):
    def test_studies_without_shared_dv_observations_are_excluded(self):
        rng = np.random.default_rng(3)
        studies = {
            "study_a": pd.DataFrame({
                "mental_demand": rng.normal(10, 2, 40),
                "effort": rng.normal(8, 2, 40),
            }),
            "study_b": pd.DataFrame({
                "mental_demand": rng.normal(12, 2, 40),
                "effort": rng.normal(9, 2, 40),
            }),
            # Sensor-only study: no canonical DV columns at all.
            "study_c": pd.DataFrame({"telemetry_channel": rng.normal(0, 1, 40)}),
        }
        summary = build_composite_index(studies)
        self.assertEqual(set(summary["study"]), {"study_a", "study_b"})


class SyntheticCausalDefaultTests(unittest.TestCase):
    def test_synthetic_cholesky_fallback_is_refused_by_default(self):
        rng = np.random.default_rng(11)
        # Shared DV names that never co-occur within a study -> the only way
        # to fit LiNGAM would be the synthetic path, which must now be off
        # unless explicitly allowed.
        studies = {
            "study_a": pd.DataFrame({
                "mental_demand": rng.normal(size=25), "effort": np.nan,
            }),
            "study_b": pd.DataFrame({
                "mental_demand": np.nan, "effort": rng.normal(size=25),
            }),
        }
        self.assertIsNone(discover_causal_structure(studies, min_rows=10))


class LoadStudiesPrefilterTests(unittest.TestCase):
    def test_dv_less_files_are_skipped_and_dv_files_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "mixed_study"
            study.mkdir()
            pd.DataFrame({
                "user_id": [1, 2],
                "userPositionX": [0.1, 0.2],
                "userPositionY": [1.1, 1.2],
            }).to_csv(study / "telemetry.csv", index=False)
            pd.DataFrame({
                "participant_id": [1, 2],
                "trust_rating": [3.0, 4.0],
            }).to_csv(study / "survey.csv", index=False)

            studies = load_studies(Path(tmp))
            sources = set(studies["mixed_study"].get("_source_file", pd.Series(dtype=str)))
        self.assertEqual(sources, {"survey.csv"})

    def test_studies_with_no_dv_files_are_kept_as_empty_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "sensor_only"
            study.mkdir()
            pd.DataFrame({"userPositionX": [0.1]}).to_csv(study / "telemetry.csv", index=False)
            studies = load_studies(Path(tmp))
        self.assertIn("sensor_only", studies)
        self.assertTrue(studies["sensor_only"].empty)

    def test_prefilter_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "sensor_only"
            study.mkdir()
            pd.DataFrame({"userPositionX": [0.1, 0.2]}).to_csv(study / "telemetry.csv", index=False)
            studies = load_studies(Path(tmp), prefilter_headers=False)
        self.assertEqual(len(studies["sensor_only"]), 2)


if __name__ == "__main__":
    unittest.main()
