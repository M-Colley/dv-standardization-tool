"""Tests for the parallel batch runner, its extension points, and the
analysis-side loading / composite / causal changes that accompany it."""

from __future__ import annotations

import filecmp
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses import multi_study_analysis as msa  # noqa: E402
from scripts import batch_profiles  # noqa: E402
from scripts import run_batch_standardization as rbs  # noqa: E402
from scripts.data_loaders import _decode_text_sample, _load_text_table  # noqa: E402

SCHEMA = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"


def _write_study_tree(root: Path) -> None:
    """A small source with questionnaire items, a results table and telemetry."""
    rng = np.random.default_rng(7)
    n = 40
    base = rng.normal(4, 1, n)
    pd.DataFrame({
        "participant_id": range(n),
        "Trust1": base + rng.normal(0, 0.3, n),
        "Trust2": base + rng.normal(0, 0.3, n),
        "Trust3": base + rng.normal(0, 0.3, n),
        "TLX1": rng.integers(0, 21, n),
        "mystery_score": rng.normal(0, 1, n),
    }).to_csv(root / "questionnaire.csv", index=False)
    pd.DataFrame({
        "participant_id": range(n),
        "sus_score": rng.normal(70, 10, n),
        "completion_time": rng.normal(30, 5, n),
    }).to_csv(root / "results.csv", index=False)
    logs = root / "logs"
    logs.mkdir()
    for i in range(3):
        pd.DataFrame({
            "Timestamp": np.arange(50) * 0.1,
            "GazeForward": rng.normal(0, 1, 50),
            "GazeOrigin": rng.normal(0, 1, 50),
            "LeftPupilDiameterInMM": rng.normal(3, 0.2, 50),
            "RightPupilDiameterInMM": rng.normal(3, 0.2, 50),
        }).to_csv(logs / f"eye_{i}.csv", index=False)


def _write_manifest(path: Path, location: Path, **extra) -> Path:
    source = {
        "source_id": "toy",
        "source_type": "local_path",
        "location": str(location),
        "include_globs": ["**/*.csv"],
        **extra,
    }
    path.write_text(yaml.safe_dump({"sources": [source]}), encoding="utf-8")
    return path


def _comparable_meta_view(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "meta_view.csv").drop(columns=["run_timestamp"])


class ParallelBatchTests(unittest.TestCase):
    def _run(self, tmp: Path, name: str, workers: int, **kwargs) -> tuple[Path, dict]:
        out = tmp / name
        summary = rbs.run_batch(
            tmp / "manifest.yaml", out, SCHEMA, cache_root=tmp / "cache", workers=workers, **kwargs
        )
        return out, summary

    def test_parallel_run_matches_sequential_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "data").mkdir()
            _write_study_tree(tmp / "data")
            _write_manifest(tmp / "manifest.yaml", tmp / "data")

            seq_out, seq = self._run(tmp, "seq", workers=1, llm_deduction_enabled=False)
            par_out, par = self._run(tmp, "par", workers=2, llm_deduction_enabled=False)

            pd.testing.assert_frame_equal(_comparable_meta_view(seq_out), _comparable_meta_view(par_out))
            self.assertEqual(seq["mapping_metrics"], par["mapping_metrics"])
            self.assertEqual(seq["composites_composed"], par["composites_composed"])
            self.assertEqual(par["workers"], 2)
            seq_files = sorted(p.name for p in (seq_out / "standardized" / "toy").iterdir())
            par_files = sorted(p.name for p in (par_out / "standardized" / "toy").iterdir())
            self.assertEqual(seq_files, par_files)
            for name in seq_files:
                if name.endswith(".csv"):
                    self.assertTrue(
                        filecmp.cmp(seq_out / "standardized" / "toy" / name,
                                    par_out / "standardized" / "toy" / name, shallow=False),
                        name,
                    )

    def test_llm_deduction_runs_in_parent_and_matches_sequential(self):
        """Pool mode hands LLM-eligible datasets back to the parent, so mocks
        applied here (and the real model) only ever run in this process."""
        calls: list[str] = []

        def fake_llm(raw_column_name, canonical_candidates, **_kwargs):
            calls.append(raw_column_name)
            return "trust_rating" if raw_column_name == "mystery_score" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "data").mkdir()
            _write_study_tree(tmp / "data")
            _write_manifest(tmp / "manifest.yaml", tmp / "data")
            outputs = {}
            for workers in (1, 2):
                calls.clear()
                with mock.patch("scripts.llm_coordinator.deduce_standard_name_with_local_llm", side_effect=fake_llm), \
                        mock.patch("scripts.run_batch_standardization.collect_repository_context", return_value=""), \
                        mock.patch("scripts.llm_coordinator._select_llm_candidate_shortlist",
                                   return_value=(["trust_rating"], 100.0)):
                    out, summary = self._run(tmp, f"w{workers}", workers=workers, llm_deduction_enabled=True)
                outputs[workers] = (out, summary, list(calls))

            (out1, sum1, calls1), (out2, sum2, calls2) = outputs[1], outputs[2]
            self.assertEqual(calls1, calls2)
            self.assertIn("mystery_score", calls1)
            self.assertEqual(sum1["llm_deductions_count"], sum2["llm_deductions_count"])
            self.assertGreaterEqual(sum1["llm_deductions_count"], 1)
            self.assertEqual(
                json.loads((out1 / "llm_deductions.json").read_text(encoding="utf-8")),
                json.loads((out2 / "llm_deductions.json").read_text(encoding="utf-8")),
            )
            pd.testing.assert_frame_equal(_comparable_meta_view(out1), _comparable_meta_view(out2))

    def test_llm_failure_marks_only_that_dataset_failed_in_both_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "data").mkdir()
            _write_study_tree(tmp / "data")
            _write_manifest(tmp / "manifest.yaml", tmp / "data")
            for workers in (1, 2):
                with self.subTest(workers=workers), \
                        mock.patch("scripts.llm_coordinator.deduce_standard_name_with_local_llm",
                                   side_effect=RuntimeError("model crashed")), \
                        mock.patch("scripts.run_batch_standardization.collect_repository_context", return_value=""), \
                        mock.patch("scripts.llm_coordinator._select_llm_candidate_shortlist",
                                   return_value=(["trust_rating"], 100.0)):
                    _, summary = self._run(tmp, f"fail_w{workers}", workers=workers, llm_deduction_enabled=True)
                    result = summary["results"][0]
                    self.assertEqual(result["status"], "partial")
                    self.assertGreaterEqual(result["failed_files"], 1)
                    self.assertGreater(result["processed_files"], 0)
                    self.assertIn("model crashed", result["message"])

    def test_transient_http_outage_is_reported_as_temporarily_unavailable(self):
        request = httpx.Request("GET", "https://api.osf.io/v2/nodes/abcde/children/")
        error = httpx.HTTPStatusError(
            "503", request=request, response=httpx.Response(503, request=request)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_manifest(tmp / "manifest.yaml", tmp)
            with mock.patch("scripts.run_batch_standardization.discover_source_files", side_effect=error):
                _, summary = self._run(tmp, "out", workers=1, llm_deduction_enabled=False)
        self.assertEqual(summary["results"][0]["status"], "temporarily_unavailable")
        self.assertEqual(summary["unavailable_sources"], ["toy"])
        self.assertEqual(summary["failed_sources"], [])


class SourceRegistryTests(unittest.TestCase):
    def tearDown(self):
        rbs._SOURCE_MATERIALIZERS.pop("unit_test_copy", None)
        rbs.SUPPORTED_SOURCE_TYPES.discard("unit_test_copy")

    def test_registered_source_type_is_validated_and_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "upstream").mkdir()
            _write_study_tree(tmp / "upstream")

            @rbs.register_source_type("unit_test_copy")
            def _copy(source, target, refresh):  # noqa: ARG001
                import shutil
                shutil.copytree(source["location"], target, dirs_exist_ok=True)
                return target, "copied"

            (tmp / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
                "source_id": "copied_source",
                "source_type": "unit_test_copy",
                "location": str(tmp / "upstream"),
                "include_globs": ["**/*.csv"],
                "exclude_globs": ["**/logs/**"],
            }]}), encoding="utf-8")
            summary = rbs.run_batch(tmp / "manifest.yaml", tmp / "out", SCHEMA,
                                    llm_deduction_enabled=False, cache_root=tmp / "cache")
            result = summary["results"][0]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["processed_files"], 2)
            meta = pd.read_csv(tmp / "out" / "meta_view.csv")
            self.assertEqual(set(meta["commit"]), {"copied"})

    def test_unknown_source_type_is_rejected(self):
        with self.assertRaises(ValueError):
            rbs.discover_source_files({"source_type": "nope", "source_id": "x", "location": "."}, Path("."))


class DatasetProfileTests(unittest.TestCase):
    def test_builtin_profiles_keep_llm_and_composition_on_dv_tables_only(self):
        expected = {batch_profiles.DATASET_TYPE_QUESTIONNAIRE, batch_profiles.DATASET_TYPE_RESULTS}
        self.assertEqual(set(batch_profiles.llm_eligible_dataset_types()), expected)
        self.assertEqual(set(batch_profiles.item_composition_dataset_types()), expected)

    def test_profile_switches_override_schema_default(self):
        profiles = [
            {"id": "experience_sampling", "schema": "standard_dv_mapping.yaml", "llm_deduction": False},
            {"id": "physio", "schema": "standard_sensor_mapping.yaml", "item_composition": True},
        ]
        with mock.patch("scripts.batch_profiles.load_dataset_type_profiles", return_value=profiles):
            self.assertEqual(batch_profiles.llm_eligible_dataset_types(), frozenset())
            self.assertEqual(
                set(batch_profiles.item_composition_dataset_types()), {"experience_sampling", "physio"}
            )

    def test_schema_family_selection_matches_previous_hard_coded_rule(self):
        families = rbs.SchemaFamilies.load(SCHEMA)
        metadata = families.mappings["standard_metadata_mapping.yaml"]
        dv = families.mappings["standard_dv_mapping.yaml"]
        expected = {
            batch_profiles.DATASET_TYPE_RESULTS: {**metadata, **dv},
            batch_profiles.DATASET_TYPE_QUESTIONNAIRE: {**metadata, **dv},
            batch_profiles.DATASET_TYPE_SENSOR: {**metadata, **families.mappings["standard_sensor_mapping.yaml"]},
            batch_profiles.DATASET_TYPE_DETECTION: {
                **metadata, **families.mappings["standard_detection_mapping.yaml"]
            },
            batch_profiles.DATASET_TYPE_PROCESS: dict(metadata),
            "some_unknown_type": {**metadata, **dv},
        }
        for dataset_type, mapping in expected.items():
            with self.subTest(dataset_type=dataset_type):
                self.assertEqual(families.standard_mapping_for(dataset_type), mapping)


class TextSampleTests(unittest.TestCase):
    def _assert_matches_text_read(self, path: Path, encoding: str) -> None:
        head = path.read_bytes()[:32768]
        with path.open("r", encoding=encoding, errors="replace") as fh:
            expected = fh.read(16384)
        self.assertEqual(_decode_text_sample(head, encoding, path), expected)

    def test_matches_text_mode_read_for_crlf_and_multibyte_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rows = "".join(f"{i};Zufriedenheit ä ö ü € {i * 1.5}\r\n" for i in range(4000))
            (tmp / "long.csv").write_bytes(("id;label;value\r\n" + rows).encode("utf-8"))
            (tmp / "short.csv").write_bytes(b"a,b\r\n1,2\r\n")
            (tmp / "utf16.tsv").write_bytes(("x\ty\n" + "1\té\n" * 9000).encode("utf-16-le"))
            self._assert_matches_text_read(tmp / "long.csv", "utf-8")
            self._assert_matches_text_read(tmp / "short.csv", "utf-8")
            self._assert_matches_text_read(tmp / "utf16.tsv", "utf-16-le")
            df = _load_text_table(tmp / "long.csv")
            self.assertEqual(list(df.columns), ["id", "label", "value"])
            self.assertEqual(len(df), 4000)


class AnalysisLoadingTests(unittest.TestCase):
    def test_skipping_dv_free_files_keeps_results_and_study_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "study_a").mkdir()
            (root / "study_b").mkdir()
            pd.DataFrame({"trust_rating": [3.0, 4.0, 5.0], "mental_demand": [10, 12, 14]}).to_csv(
                root / "study_a" / "questionnaire.csv", index=False
            )
            pd.DataFrame({"Timestamp": range(100), "GazeForward": np.zeros(100)}).to_csv(
                root / "study_a" / "telemetry.csv", index=False
            )
            pd.DataFrame({"Timestamp": range(10)}).to_csv(root / "study_b" / "log.csv", index=False)

            full = msa.load_studies(root)
            fast = msa.load_studies(root, skip_files_without_dvs=True)

        self.assertEqual(set(full), set(fast))
        self.assertEqual(len(full["study_a"]), 103)
        self.assertEqual(len(fast["study_a"]), 3)
        self.assertTrue(fast["study_b"].empty)
        full_summary = msa.harmonized_summary(full)
        fast_summary = msa.harmonized_summary(fast)
        pd.testing.assert_frame_equal(
            full_summary.reset_index(drop=True), fast_summary.reset_index(drop=True)
        )

    def test_mapping_provenance_keeps_most_cautious_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta_view.csv"
            pd.DataFrame({
                "source_id": ["s1", "s1", "s1", "s2", "s2"],
                "canonical_dv": ["trust_rating", "trust_rating", "usability", "usability", None],
                "mapping_source": [
                    "schemas/standard_dv_mapping.yaml", "llm_deduction", "repo/mapping.yaml", None, "x",
                ],
            }).to_csv(path, index=False)
            provenance = msa.load_mapping_provenance(path)
        self.assertEqual(provenance, {
            ("s1", "trust_rating"): "llm_deduced",
            ("s1", "usability"): "repo_mapping",
            ("s2", "usability"): "unknown",
        })


class CompositeIndexTests(unittest.TestCase):
    def test_rows_without_any_shared_dv_do_not_change_the_composite(self):
        rng = np.random.default_rng(3)
        studies = {
            "a": pd.DataFrame({"trust_rating": rng.normal(4, 1, 30), "mental_demand": rng.normal(10, 3, 30)}),
            "b": pd.DataFrame({"trust_rating": rng.normal(3, 1, 25), "mental_demand": rng.normal(12, 3, 25)}),
        }
        padded = {
            "a": pd.concat([studies["a"], pd.DataFrame(np.nan, index=range(500), columns=studies["a"].columns)],
                           ignore_index=True),
            "b": studies["b"],
        }
        clean = msa.build_composite_index(studies)
        with_padding = msa.build_composite_index(padded)
        pd.testing.assert_frame_equal(clean, with_padding)
        self.assertEqual(int(clean.loc[clean["study"] == "a", "n"].iloc[0]), 30)

    def test_scores_can_be_shared_between_index_and_plot(self):
        rng = np.random.default_rng(4)
        studies = {
            s: pd.DataFrame({"trust_rating": rng.normal(4, 1, 20), "mental_demand": rng.normal(10, 3, 20)})
            for s in ("a", "b")
        }
        scores = msa.compute_composite_scores(studies)
        pd.testing.assert_frame_equal(
            msa.build_composite_index(studies), msa.build_composite_index(studies, scores=scores)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            msa.save_composite_plot(studies, Path(tmpdir), scores=scores)
            self.assertTrue((Path(tmpdir) / "cross_study_composite_distribution.png").exists())


class CausalSubitemTests(unittest.TestCase):
    def test_schema_item_ids_are_grouped_under_their_construct(self):
        cs = msa.CONSTRUCT_SUBITEMS
        self.assertEqual(
            cs["perceived_safety"],
            [f"perceived_safety_item_0{i}" for i in range(1, 5)],
        )
        # Declared explicitly via `subitem_of` (naming differs from the construct id).
        self.assertEqual(len(cs["perceived_conflict"]), 5)
        self.assertEqual(len(cs["sts_ad_trust"]), 6)
        # Items without a matching construct id stay ungrouped.
        self.assertFalse(any("tia_item_01" in items for items in cs.values()))

    def test_items_are_dropped_from_the_dag_only_when_the_construct_is_present(self):
        dvs = ["perceived_safety", "perceived_safety_item_01", "trust_rating"]
        self.assertEqual(msa._filter_subitems_when_construct_present(dvs), ["perceived_safety", "trust_rating"])
        self.assertEqual(
            msa._filter_subitems_when_construct_present(["perceived_safety_item_01", "trust_rating"]),
            ["perceived_safety_item_01", "trust_rating"],
        )


class CausalStabilityTests(unittest.TestCase):
    def _studies(self):
        rng = np.random.default_rng(11)

        def block(n):
            x = rng.uniform(-1, 1, n)
            y = 0.8 * x + rng.uniform(-0.5, 0.5, n)
            z = 0.5 * y + rng.uniform(-0.5, 0.5, n)
            return pd.DataFrame({"trust_rating": x, "usability": y, "frustration": z})

        return {"s1": block(250), "s2": block(250)}

    def test_bootstrap_on_dag_data_is_reproducible_and_leaves_global_rng_alone(self):
        studies = self._studies()
        causal = msa.discover_causal_structure(studies, min_rows=100)
        self.assertIsNotNone(causal)
        data = causal["data"]
        self.assertEqual(len(data), causal["n_observations"])

        np.random.seed(123)
        expected_next = np.random.random()
        np.random.seed(123)
        first = msa.bootstrap_causal_stability(studies, n_bootstrap=10, data=data)
        self.assertEqual(np.random.random(), expected_next)
        second = msa.bootstrap_causal_stability(studies, n_bootstrap=10, data=data)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first.empty)
        self.assertTrue(((first["appearance_rate"] > 0) & (first["appearance_rate"] <= 1)).all())

    def test_pc_skeleton_accepts_dag_data(self):
        studies = self._studies()
        causal = msa.discover_causal_structure(studies, min_rows=100)
        result = msa.discover_causal_structure_pc(studies, data=causal["data"])
        self.assertEqual(result["n_observations"], causal["n_observations"])
        self.assertEqual(sorted(result["dvs_used"]), sorted(causal["dvs_used"]))


if __name__ == "__main__":
    unittest.main()
