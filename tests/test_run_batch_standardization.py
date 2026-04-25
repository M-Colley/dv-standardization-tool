import io
import json
import os
import socket
import stat
import zipfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from email.message import Message
from urllib import error as urlerror

import pandas as pd
import yaml

from scripts.run_batch_standardization import (
    SourceAccessError,
    _augment_mapping_with_llm_deductions,
    _build_artifact_prefix,
    _extract_osf_project_id,
    _iter_osf_file_entries,
    _load_repository_mapping,
    _match_files,
    _safe_rmtree,
    discover_source_files,
    load_manifest,
    run_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"
EXAMPLE_MANIFEST_PATH = REPO_ROOT / "sources_manifest_example.yaml"


def _paths_equal(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return os.path.normcase(os.path.realpath(os.fspath(left))) == os.path.normcase(
            os.path.realpath(os.fspath(right))
        )


class BatchStandardizationTests(unittest.TestCase):
    def test_load_repository_mapping_uses_standard_schema_precedence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            (source_root / "custom_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {
                                "id": "custom_task_time",
                                "aliases": ["duration", "local_alias_only"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            merged_mapping, mapping_path = _load_repository_mapping(source_root)

            self.assertIsNotNone(mapping_path)
            self.assertTrue(mapping_path.endswith("custom_mapping.yaml"))
            self.assertEqual(merged_mapping["duration"], "task_completion_time")
            self.assertEqual(merged_mapping["local_alias_only"], "custom_task_time")

    def test_load_repository_mapping_detects_nested_mapping_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            nested = source_root / "nested" / "config"
            nested.mkdir(parents=True, exist_ok=True)
            mapping_file = nested / "custom_mapping.yaml"
            mapping_file.write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {"id": "custom_task_time", "aliases": ["local_alias_only"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            merged_mapping, mapping_path = _load_repository_mapping(source_root)

            self.assertIsNotNone(mapping_path)
            self.assertTrue(_paths_equal(mapping_path, mapping_file))
            self.assertEqual(merged_mapping["local_alias_only"], "custom_task_time")

    def test_load_repository_mapping_tolerates_null_dvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            (source_root / "custom_mapping.yaml").write_text(
                "version: '2.1'\ndvs: null\n",
                encoding="utf-8",
            )

            merged_mapping, mapping_path = _load_repository_mapping(source_root)

            self.assertIn("task_time", merged_mapping)
            self.assertIsNotNone(mapping_path)
            self.assertTrue(mapping_path.endswith("custom_mapping.yaml"))

    def test_load_repository_mapping_ignores_ambiguous_mapping_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            (source_root / "a_mapping.yaml").write_text("dvs: []", encoding="utf-8")
            (source_root / "b_mapping.yaml").write_text("dvs: []", encoding="utf-8")

            merged_mapping, mapping_path = _load_repository_mapping(source_root)

            self.assertEqual(merged_mapping, {})
            self.assertIsNone(mapping_path)

    def test_load_repository_mapping_uses_nearest_dataset_mapping_when_multiple_candidates_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            dataset_dir = source_root / "study_a"
            other_dir = source_root / "study_b"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            other_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "results.csv"
            dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
            (dataset_dir / "source_mapping.yaml").write_text(
                yaml.safe_dump({"dvs": [{"id": "trust_rating", "aliases": ["TrustA"]}]}),
                encoding="utf-8",
            )
            (other_dir / "source_mapping.yaml").write_text(
                yaml.safe_dump({"dvs": [{"id": "mental_demand", "aliases": ["TrustA"]}]}),
                encoding="utf-8",
            )

            merged_mapping, mapping_path = _load_repository_mapping(
                source_root,
                dataset_path=dataset_path,
            )

            self.assertTrue(_paths_equal(mapping_path, dataset_dir / "source_mapping.yaml"))
            self.assertEqual(merged_mapping["TrustA"], "trust_rating")

    def test_load_repository_mapping_uses_requested_mapping_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            nested = source_root / "config"
            nested.mkdir(parents=True, exist_ok=True)
            (source_root / "a_mapping.yaml").write_text("dvs: []", encoding="utf-8")
            requested_path = nested / "custom_mapping.yaml"
            requested_path.write_text(
                yaml.safe_dump({"dvs": [{"id": "trust_rating", "aliases": ["TrustA"]}]}),
                encoding="utf-8",
            )

            merged_mapping, mapping_path = _load_repository_mapping(
                source_root,
                requested_mapping_path="config/custom_mapping.yaml",
            )

            self.assertTrue(_paths_equal(mapping_path, requested_path))
            self.assertEqual(merged_mapping["TrustA"], "trust_rating")

    def test_safe_rmtree_handles_read_only_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "readonly_tree"
            root.mkdir(parents=True, exist_ok=True)
            target = root / "pack.idx"
            target.write_text("x", encoding="utf-8")
            os.chmod(target, stat.S_IREAD)

            _safe_rmtree(root)

            self.assertFalse(root.exists())

    def test_match_files_skips_macos_metadata_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "usable.csv").write_text("a\n1\n", encoding="utf-8")
            macosx_dir = root / "__MACOSX" / "nested"
            macosx_dir.mkdir(parents=True, exist_ok=True)
            (macosx_dir / "ignored.csv").write_text("a\n1\n", encoding="utf-8")
            (root / "._ignored.csv").write_text("a\n1\n", encoding="utf-8")

            matched = _match_files(root, ["**/*.csv"], None)

        self.assertEqual([path.name for path in matched], ["usable.csv"])

    def test_match_files_respects_directory_style_exclude_globs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "usable"
            excluded_dir = root / "ClearedLogs" / "nested"
            keep_dir.mkdir(parents=True, exist_ok=True)
            excluded_dir.mkdir(parents=True, exist_ok=True)
            (keep_dir / "results.csv").write_text("a\n1\n", encoding="utf-8")
            (excluded_dir / "ignored.csv").write_text("a\n1\n", encoding="utf-8")

            matched = _match_files(root, ["**/*.csv"], ["**/ClearedLogs/**"])

        self.assertEqual(
            [path.relative_to(root).as_posix() for path in matched],
            ["usable/results.csv"],
        )

    def test_iter_osf_file_entries_reads_all_provider_roots(self):
        provider_page = {
            "data": [
                {
                    "relationships": {
                        "files": {
                            "links": {
                                "related": {"href": "https://api.osf.io/v2/nodes/cwd6h/files/osfstorage/"}
                            }
                        }
                    }
                },
                {"links": {"related": {"href": "https://api.osf.io/v2/nodes/cwd6h/files/dropbox/"}}},
            ],
            "links": {"next": None},
        }
        osfstorage_page = {
            "data": [
                {
                    "attributes": {"kind": "file", "path": "/study_a.csv"},
                    "links": {"download": "https://files.osf.io/study_a.csv"},
                }
            ],
            "links": {"next": None},
        }
        dropbox_page = {
            "data": [
                {
                    "attributes": {"kind": "file", "path": "/study_b.csv"},
                    "links": {"download": "https://files.osf.io/study_b.csv"},
                }
            ],
            "links": {"next": None},
        }

        with mock.patch(
            "scripts.run_batch_standardization._osf_json_get",
            side_effect=[provider_page, osfstorage_page, dropbox_page],
        ) as mocked_get:
            entries = _iter_osf_file_entries("cwd6h")

        self.assertEqual(len(entries), 2)
        self.assertEqual(mocked_get.call_count, 3)

    def test_augment_mapping_with_llm_deductions_adds_inferred_aliases(self):
        mapping = {
            "task_time": "task_completion_time",
            "duration": "task_completion_time",
            "task_completion_time": "task_completion_time",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                augmented = _augment_mapping_with_llm_deductions(
                    mapping,
                    ["task_time", "NovelDuration"],
                    Path(tmpdir),
                )

        self.assertEqual(augmented["NovelDuration"], "task_completion_time")
        self.assertEqual(augmented["novelduration"], "task_completion_time")
        mocked.assert_called_once()

    def test_augment_mapping_with_llm_deductions_reuses_inference_cache(self):
        mapping = {
            "task_time": "task_completion_time",
            "duration": "task_completion_time",
            "task_completion_time": "task_completion_time",
        }
        inference_cache: dict[str, str | None] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                _augment_mapping_with_llm_deductions(
                    mapping,
                    ["NovelDuration"],
                    Path(tmpdir),
                    inference_cache=inference_cache,
                )
                _augment_mapping_with_llm_deductions(
                    mapping,
                    ["NovelDuration"],
                    Path(tmpdir),
                    inference_cache=inference_cache,
                )

        self.assertEqual(mocked.call_count, 1)

    def test_augment_mapping_with_llm_deductions_skips_low_similarity_aliases(self):
        mapping = {"task_time": "task_completion_time", "trust": "trust_rating"}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                augmented = _augment_mapping_with_llm_deductions(
                    mapping,
                    ["TotallyNovelSensorBlob"],
                    Path(tmpdir),
                )

        mocked.assert_not_called()
        self.assertNotIn("TotallyNovelSensorBlob", augmented)

    def test_run_batch_uses_llm_deduction_when_repo_has_no_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            pd.DataFrame({"DurationX": [1.0, 2.0]}).to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ):
                summary = run_batch(manifest_path, output_dir, SCHEMA_PATH)

            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("task_completion_time", standardized_df.columns)
            self.assertEqual(summary["llm_deductions_count"], 1)
            llm_log_path = Path(summary["llm_deductions_log"])
            llm_json_path = Path(summary["llm_deductions_json"])
            self.assertTrue(llm_log_path.exists())
            self.assertTrue(llm_json_path.exists())
            self.assertIn("DurationX -> task_completion_time", llm_log_path.read_text(encoding="utf-8"))
            llm_records = json.loads(llm_json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(llm_records), 1)
            self.assertEqual(llm_records[0]["alias"], "DurationX")
            self.assertEqual(llm_records[0]["canonical_dv"], "task_completion_time")

    def test_run_batch_marks_sources_with_no_supported_files_as_not_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()
            (input_dir / "notes.txt").write_text("no tabular data here", encoding="utf-8")

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "empty_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["**/*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = run_batch(manifest_path, tmp / "output", SCHEMA_PATH, llm_deduction_enabled=False)

        self.assertEqual(summary["results"][0]["status"], "not_available")
        self.assertIn("No supported dataset files", summary["results"][0]["message"])

    def test_run_batch_records_login_required_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "ieee_source",
                                "source_type": "web_dataset",
                                "location": "https://ieee-dataport.org/open-access/usyd-campus-dataset",
                                "include_globs": ["**/*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "scripts.run_batch_standardization.discover_source_files",
                side_effect=SourceAccessError("login_required", "Dataset files require login."),
            ):
                summary = run_batch(manifest_path, tmp / "output", SCHEMA_PATH, llm_deduction_enabled=False)

        self.assertEqual(summary["results"][0]["status"], "login_required")
        self.assertEqual(summary["results"][0]["message"], "Dataset files require login.")

    def test_run_batch_standardizes_pickle_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()
            pd.DataFrame({"DurationX": [1.0, 2.0]}).to_pickle(input_dir / "study.pkl")
            (input_dir / "source_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {"id": "task_completion_time", "aliases": ["DurationX"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "pickle_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.pkl"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = run_batch(manifest_path, tmp / "output", SCHEMA_PATH, llm_deduction_enabled=False)

            standardized_path = tmp / "output" / "standardized" / "pickle_source" / "study_pkl-standardized.pkl"
            standardized_df = pd.read_pickle(standardized_path)

        self.assertEqual(summary["results"][0]["status"], "completed")
        self.assertIn("task_completion_time", standardized_df.columns)

    def test_run_batch_passes_publication_hints_into_llm_context_collection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            pd.DataFrame({"DurationX": [1.0, 2.0]}).to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                                "publication_doi": "10.1145/3706598.3713762",
                                "publication_pdf_url": "https://example.com/paper.pdf",
                                "llm_context": "Study focuses on trust and workload.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.collect_repository_context",
                return_value="[CONTEXT_SUMMARY]",
            ) as mocked_context:
                with mock.patch(
                    "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                    return_value="task_completion_time",
                ):
                    run_batch(manifest_path, output_dir, SCHEMA_PATH)

            mocked_context.assert_called_once()
            self.assertEqual(
                mocked_context.call_args.kwargs["explicit_dois"],
                ["10.1145/3706598.3713762"],
            )
            self.assertEqual(
                mocked_context.call_args.kwargs["explicit_pdf_urls"],
                ["https://example.com/paper.pdf"],
            )
            self.assertEqual(
                mocked_context.call_args.kwargs["extra_context"],
                ["Study focuses on trust and workload."],
            )

    def test_load_manifest_rejects_duplicate_source_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {"source_id": "dup", "source_type": "local_path", "location": "/tmp/a"},
                            {"source_id": "dup", "source_type": "local_path", "location": "/tmp/b"},
                        ]
                    }
                )
            )

            with self.assertRaises(ValueError):
                load_manifest(manifest_path)

    def test_example_manifest_points_to_requested_github_repos(self):
        sources = load_manifest(EXAMPLE_MANIFEST_PATH)

        # Manifest may grow over time — check that it has at least 5 sources
        self.assertGreaterEqual(len(sources), 5)

        # Build a lookup by source_id so the test is order-independent
        by_id = {s["source_id"]: s for s in sources}

        self.assertIn("roads_chi25", by_id)
        self.assertEqual(
            by_id["roads_chi25"]["location"],
            "https://github.com/M-Colley/roads-chi25-data",
        )
        self.assertIn("ehmi_optimization_chi25", by_id)
        self.assertEqual(
            by_id["ehmi_optimization_chi25"]["location"],
            "https://github.com/M-Colley/ehmi-optimization-chi25-data",
        )
        self.assertIn("osf_cwd6h", by_id)
        self.assertEqual(
            by_id["osf_cwd6h"]["location"],
            "https://osf.io/cwd6h/overview",
        )
        self.assertIn("road_bumps_touch", by_id)
        self.assertEqual(
            by_id["road_bumps_touch"]["location"],
            "https://github.com/interactionlab/Touch-Interaction-with-Road-Bumps/tree/master/data",
        )
        self.assertIn("fourtu_critical_ehmi", by_id)
        self.assertEqual(
            by_id["fourtu_critical_ehmi"]["location"],
            "https://data.4tu.nl/articles/_/20224281",
        )

    def test_run_batch_maps_semicolon_csv_columns_including_mental_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            (input_dir / "ObservationsPerEvaluation.csv").write_text(
                "User_ID;Trust;MentalLoad\n1;4.0;2.0\n2;3.5;1.5\n",
                encoding="utf-8",
            )
            (input_dir / "ehmi_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {"id": "trust_rating", "aliases": ["Trust"]},
                            {"id": "mental_effort", "aliases": ["MentalLoad"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "semicolon_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            run_batch(manifest_path, output_dir, SCHEMA_PATH)

            standardized_path = output_dir / "standardized" / "semicolon_source" / "ObservationsPerEvaluation_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)

            self.assertIn("mental_demand", standardized_df.columns)
            self.assertIn("trust_rating", standardized_df.columns)

    def test_run_batch_generates_meta_view_and_summary_for_local_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            df = pd.DataFrame(
                {
                    "Task_Completion_Time": [10.0, 12.0, 11.0],
                    "Unknown Alias": [1, 2, 3],
                }
            )
            df.to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                )
            )

            output_dir = tmp / "output"
            summary = run_batch(
                manifest_path,
                output_dir,
                SCHEMA_PATH,
                llm_deduction_enabled=False,
            )

            self.assertEqual(summary["total_sources"], 1)
            self.assertEqual(summary["successful_sources"], 1)
            self.assertIn("mapped_ratio", summary["results"][0])
            self.assertGreaterEqual(summary["results"][0]["mapped_ratio"], 0.0)
            self.assertIn("mapping_metrics", summary)
            self.assertIn("mapping_metrics_by_source", summary)
            self.assertIn("mapping_metrics_by_domain", summary)
            self.assertIn("unknown_alias_summary", summary)
            self.assertIn("mapping_debug_summary", summary)
            self.assertEqual(
                summary["mapping_metrics"]["total_columns_seen"],
                summary["mapping_metrics"]["mapping"]
                + summary["mapping_metrics"]["llm"]
                + summary["mapping_metrics"]["blocked"]
                + summary["mapping_metrics"]["unmapped"],
            )

            meta_view_path = output_dir / "meta_view.csv"
            run_summary_path = output_dir / "run_summary.json"
            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            quality_path = output_dir / "standardized" / "study_source" / "study_csv-quality.json"

            self.assertTrue(meta_view_path.exists())
            self.assertTrue(run_summary_path.exists())
            self.assertTrue(standardized_path.exists())
            self.assertTrue(quality_path.exists())
            self.assertTrue(Path(summary["unknown_alias_summary"]).exists())
            self.assertTrue(Path(summary["mapping_debug_summary"]).exists())

            meta_df = pd.read_csv(meta_view_path)
            self.assertIn("canonical_dv", meta_df.columns)
            self.assertIn("source_id", meta_df.columns)
            self.assertIn("dataset_type", meta_df.columns)
            self.assertIn("mapping_domain", meta_df.columns)
            self.assertIn("task_completion_time", meta_df["canonical_dv"].values)
            self.assertEqual(meta_df.loc[meta_df["canonical_dv"] == "task_completion_time", "dataset_type"].iat[0], "results_table")
            self.assertEqual(meta_df.loc[meta_df["canonical_dv"] == "task_completion_time", "mapping_domain"].iat[0], "dv")

            quality = json.loads(quality_path.read_text())
            self.assertEqual(quality["unknown_columns"], 1)
            self.assertIn("Unknown Alias", quality["unknown_aliases"])
            unknown_summary = json.loads(Path(summary["unknown_alias_summary"]).read_text(encoding="utf-8"))
            self.assertEqual(unknown_summary["total_unknown_alias_events"], 1)
            self.assertEqual(unknown_summary["top_unknown_aliases"][0]["alias"], "Unknown Alias")

    def test_run_batch_applies_repo_mapping_without_overriding_standard_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            (input_dir / "source_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {
                                "id": "custom_task_time",
                                "aliases": ["duration", "LocalOnlyAlias"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pd.DataFrame(
                {
                    "duration": [10.0, 12.0],
                    "LocalOnlyAlias": [2.0, 4.0],
                    "Unmapped": [7.0, 8.0],
                }
            ).to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            # The "Unmapped" column triggers _augment_mapping_with_llm_deductions
            # for any candidate above the heuristic similarity threshold, which
            # otherwise loads a real local LLM (gemma-4) and segfaults on RAM-
            # constrained CI. The intent of *this* test is to verify that
            # standard-schema aliases beat repo-mapping aliases — not LLM
            # inference behaviour — so we stub the LLM out entirely.
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value=None,
            ):
                run_batch(manifest_path, output_dir, SCHEMA_PATH)

            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            meta_view_path = output_dir / "meta_view.csv"

            standardized_df = pd.read_csv(standardized_path)
            meta_df = pd.read_csv(meta_view_path)

            self.assertIn("task_completion_time", standardized_df.columns)
            self.assertIn("custom_task_time", standardized_df.columns)
            self.assertNotIn("duration", standardized_df.columns)
            self.assertTrue(
                _paths_equal(
                    meta_df.loc[meta_df["canonical_dv"] == "task_completion_time", "source_mapping"].iat[0],
                    input_dir / "source_mapping.yaml",
                )
            )
            self.assertIn("Unmapped", standardized_df.columns)

    def test_run_batch_uses_llm_after_repo_mapping_for_unmapped_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            (input_dir / "source_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {
                                "id": "trust_rating",
                                "aliases": ["KnownAlias"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame({"KnownAlias": [1.0, 2.0], "NovelDuration": [3.0, 4.0]}).to_csv(
                input_dir / "study.csv",
                index=False,
            )

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                run_batch(manifest_path, output_dir, SCHEMA_PATH)

            mocked.assert_called_once()
            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("trust_rating", standardized_df.columns)
            self.assertIn("task_completion_time", standardized_df.columns)

    def test_run_batch_never_maps_identifier_or_condition_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            (input_dir / "source_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {
                                "id": "trust_rating",
                                "aliases": ["ConditionID", "UserID"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "ConditionID": [1, 2],
                    "UserID": [101, 102],
                    "id": [11, 12],
                    "lastpage": [5, 6],
                    "seed": [123, 456],
                    "startlanguage": ["en", "en"],
                    "submitdate": ["2026-03-05", "2026-03-05"],
                    "DurationX": [3.0, 4.0],
                }
            ).to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization._select_llm_candidate_shortlist",
                return_value=(["task_completion_time"], 90.0),
            ):
                with mock.patch(
                    "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                    return_value="task_completion_time",
                ) as mocked:
                    summary = run_batch(manifest_path, output_dir, SCHEMA_PATH)

            mocked.assert_called_once()
            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)

            self.assertIn("ConditionID", standardized_df.columns)
            self.assertIn("UserID", standardized_df.columns)
            self.assertIn("id", standardized_df.columns)
            self.assertIn("lastpage", standardized_df.columns)
            self.assertIn("seed", standardized_df.columns)
            self.assertIn("startlanguage", standardized_df.columns)
            self.assertIn("submitdate", standardized_df.columns)
            self.assertNotIn("trust_rating", standardized_df.columns)
            self.assertIn("task_completion_time", standardized_df.columns)

            llm_log = Path(summary["llm_deductions_log"]).read_text(encoding="utf-8")
            self.assertIn("DurationX -> task_completion_time", llm_log)
            self.assertNotIn("ConditionID ->", llm_log)
            self.assertNotIn("UserID ->", llm_log)
            self.assertNotIn("id ->", llm_log)
            self.assertNotIn("lastpage ->", llm_log)
            self.assertNotIn("seed ->", llm_log)
            self.assertNotIn("startlanguage ->", llm_log)
            self.assertNotIn("submitdate ->", llm_log)

    def test_run_batch_does_not_send_metadata_columns_to_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            pd.DataFrame(
                {
                    "Timestamp": [1.0, 2.0],
                    "Phase": ["intro", "main"],
                    "Trust": [4.0, 5.0],
                    "DurationX": [3.0, 4.0],
                }
            ).to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                summary = run_batch(manifest_path, output_dir, SCHEMA_PATH)

            mocked.assert_called_once()
            llm_log = Path(summary["llm_deductions_log"]).read_text(encoding="utf-8")
            self.assertIn("DurationX -> task_completion_time", llm_log)
            self.assertNotIn("Timestamp ->", llm_log)
            self.assertNotIn("Phase ->", llm_log)
            self.assertNotIn("Trust ->", llm_log)

            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("meta_timestamp", standardized_df.columns)
            self.assertIn("meta_phase", standardized_df.columns)
            self.assertIn("trust_rating", standardized_df.columns)
            self.assertIn("task_completion_time", standardized_df.columns)

            meta_df = pd.read_csv(output_dir / "meta_view.csv")
            metadata_rows = meta_df[meta_df["mapping_domain"] == "metadata"]
            self.assertGreaterEqual(len(metadata_rows), 2)
            self.assertTrue((metadata_rows["dataset_type"] == "results_table").all())

    def test_run_batch_debug_mappings_writes_per_dataset_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            pd.DataFrame(
                {
                    "Trust": [1.0, 2.0],
                    "DurationX": [3.0, 4.0],
                    "UserID": [10, 11],
                    "Unknown Alias": [7.0, 8.0],
                }
            ).to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                side_effect=lambda raw_column_name, **_: (
                    "task_completion_time" if raw_column_name == "DurationX" else None
                ),
            ) as mocked:
                summary = run_batch(
                    manifest_path,
                    output_dir,
                    SCHEMA_PATH,
                    debug_mappings=True,
                )

            # Both "DurationX" and "Unknown Alias" pass the heuristic shortlist
            # threshold and reach the LLM stub; only "DurationX" gets a non-None
            # mapping. Verify both calls happened so future regressions in the
            # candidate-shortlisting code path get caught.
            self.assertEqual(mocked.call_count, 2)
            llm_called_columns = sorted(
                call.kwargs["raw_column_name"] for call in mocked.call_args_list
            )
            self.assertEqual(llm_called_columns, ["DurationX", "Unknown Alias"])
            self.assertTrue(summary["debug_mappings_enabled"])

            debug_path = (
                output_dir
                / "standardized"
                / "study_source"
                / "study_csv-mapping-debug.json"
            )
            self.assertTrue(debug_path.exists())
            payload = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_type"], "results_table")
            rows = {row["original_column"]: row for row in payload["debug_mappings"]}

            self.assertEqual(rows["Trust"]["mapped_column"], "trust_rating")
            self.assertEqual(rows["Trust"]["mapping_origin"], "schema")
            self.assertEqual(rows["Trust"]["mapping_method"], "mapping")
            self.assertEqual(rows["Trust"]["mapping_source"], str(SCHEMA_PATH))
            self.assertEqual(rows["Trust"]["mapping_domain"], "dv")

            self.assertEqual(rows["DurationX"]["mapped_column"], "task_completion_time")
            self.assertEqual(rows["DurationX"]["mapping_origin"], "llm")
            self.assertEqual(rows["DurationX"]["mapping_method"], "llm")
            self.assertEqual(rows["DurationX"]["mapping_source"], "llm_deduction")
            self.assertEqual(rows["DurationX"]["mapping_domain"], "dv")

            self.assertEqual(rows["UserID"]["mapped_column"], "UserID")
            self.assertEqual(rows["UserID"]["mapping_origin"], "blocked_never_map")
            self.assertEqual(rows["UserID"]["mapping_status"], "blocked")
            self.assertEqual(rows["UserID"]["mapping_method"], "blocked")
            self.assertEqual(rows["UserID"]["mapping_source"], "never_map_blocklist")
            self.assertEqual(rows["UserID"]["mapping_domain"], "blocked")

            self.assertEqual(rows["Unknown Alias"]["mapped_column"], "Unknown Alias")
            self.assertEqual(rows["Unknown Alias"]["mapping_origin"], "none")
            self.assertEqual(rows["Unknown Alias"]["mapping_status"], "unmapped")
            self.assertEqual(rows["Unknown Alias"]["mapping_method"], "unmapped")
            self.assertIsNone(rows["Unknown Alias"]["mapping_source"])
            self.assertEqual(rows["Unknown Alias"]["mapping_domain"], "unmapped")

    def test_run_batch_surfaces_dataset_errors_in_source_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()
            (input_dir / "study.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization._load_any_table",
                side_effect=ValueError("broken dataset"),
            ):
                summary = run_batch(manifest_path, output_dir, SCHEMA_PATH)

            self.assertEqual(summary["results"][0]["status"], "failed")
            self.assertIn("study.csv: broken dataset", summary["results"][0]["message"])

    def test_run_batch_maps_sensor_stream_columns_with_sensor_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            pd.DataFrame(
                {
                    "gaze.forward": [0.1, 0.2],
                    "leftPupilDiameterInMM": [3.4, 3.5],
                    "rightIrisDiameterInMM": [11.1, 11.0],
                    "focusDistance": [2.2, 2.3],
                    "ArduinoData1": [12, 13],
                    "PixelX": [640, 641],
                }
            ).to_csv(input_dir / "study.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            sensor_schema_path = str(REPO_ROOT / "schemas" / "standard_sensor_mapping.yaml")
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                summary = run_batch(
                    manifest_path,
                    output_dir,
                    SCHEMA_PATH,
                    llm_deduction_enabled=False,
                    debug_mappings=True,
                )

            mocked.assert_not_called()
            self.assertEqual(summary["sensor_schema"], sensor_schema_path)
            self.assertGreater(summary["sensor_mapping_aliases"], 0)
            self.assertEqual(summary["results"][0]["unknown_columns"], 0)
            self.assertEqual(summary["mapping_metrics"]["llm"], 0)

            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("eye_gaze_forward", standardized_df.columns)
            self.assertIn("eye_left_pupil_diameter_mm", standardized_df.columns)
            self.assertIn("eye_right_iris_diameter_mm", standardized_df.columns)
            self.assertIn("eye_focus_distance", standardized_df.columns)
            self.assertIn("sensor_arduino_data_1", standardized_df.columns)
            self.assertIn("sensor_pixel_x", standardized_df.columns)

            debug_path = output_dir / "standardized" / "study_source" / "study_csv-mapping-debug.json"
            payload = json.loads(debug_path.read_text(encoding="utf-8"))
            rows = {row["original_column"]: row for row in payload["debug_mappings"]}
            self.assertEqual(rows["gaze.forward"]["mapping_method"], "mapping")
            self.assertEqual(rows["gaze.forward"]["mapping_source"], sensor_schema_path)
            self.assertEqual(rows["gaze.forward"]["mapping_domain"], "sensor")
            self.assertEqual(rows["ArduinoData1"]["mapping_method"], "mapping")
            self.assertEqual(rows["ArduinoData1"]["mapping_source"], sensor_schema_path)
            self.assertEqual(rows["ArduinoData1"]["mapping_domain"], "sensor")

    def test_run_batch_maps_detection_columns_with_detection_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input" / "YOLO_BoundingBoxes"
            input_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "video": ["a.mp4", "a.mp4"],
                    "frame": [1, 2],
                    "object_id": [1, 2],
                    "class": ["car", "pedestrian"],
                    "x": [10, 12],
                    "y": [20, 21],
                    "width": [40, 41],
                    "height": [30, 31],
                    "confidence": [0.9, 0.8],
                }
            ).to_csv(input_dir / "bounding_boxes.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(tmp / "input"),
                                "include_globs": ["**/*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            summary = run_batch(
                manifest_path,
                output_dir,
                SCHEMA_PATH,
                llm_deduction_enabled=False,
                debug_mappings=True,
            )

            detection_schema_path = str(REPO_ROOT / "schemas" / "standard_detection_mapping.yaml")
            self.assertEqual(summary["detection_schema"], detection_schema_path)
            self.assertGreater(summary["detection_mapping_aliases"], 0)

            standardized_path = (
                output_dir / "standardized" / "study_source" / "YOLO_BoundingBoxes_bounding_boxes_csv-standardized.csv"
            )
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("detection_video_id", standardized_df.columns)
            self.assertIn("detection_frame_index", standardized_df.columns)
            self.assertIn("detection_bbox_width", standardized_df.columns)
            self.assertIn("detection_confidence", standardized_df.columns)

            debug_path = (
                output_dir / "standardized" / "study_source" / "YOLO_BoundingBoxes_bounding_boxes_csv-mapping-debug.json"
            )
            payload = json.loads(debug_path.read_text(encoding="utf-8"))
            rows = {row["original_column"]: row for row in payload["debug_mappings"]}
            self.assertEqual(payload["dataset_type"], "object_detection")
            self.assertEqual(rows["video"]["mapping_domain"], "detection")
            self.assertEqual(rows["video"]["mapping_source"], detection_schema_path)
            self.assertEqual(rows["confidence"]["mapping_domain"], "detection")

    def test_run_batch_uses_relative_path_prefix_to_avoid_filename_collisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            (input_dir / "wave1").mkdir(parents=True)
            (input_dir / "wave2").mkdir(parents=True)

            pd.DataFrame({"Trust": [1.0, 2.0]}).to_csv(input_dir / "wave1" / "results.csv", index=False)
            pd.DataFrame({"Trust": [3.0, 4.0]}).to_csv(input_dir / "wave2" / "results.csv", index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["**/*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            summary = run_batch(manifest_path, output_dir, SCHEMA_PATH)

            source_output = output_dir / "standardized" / "study_source"
            standardized_files = sorted(p.name for p in source_output.glob("*-standardized.csv"))

            self.assertEqual(summary["results"][0]["discovered_files"], 2)
            self.assertEqual(summary["results"][0]["processed_files"], 2)
            self.assertEqual(
                standardized_files,
                [
                    "wave1_results_csv-standardized.csv",
                    "wave2_results_csv-standardized.csv",
                ],
            )

            meta_df = pd.read_csv(output_dir / "meta_view.csv")
            self.assertEqual(sorted(meta_df["dataset_id"].unique().tolist()), ["wave1/results.csv", "wave2/results.csv"])

    def test_build_artifact_prefix_shortens_long_names_with_hash(self):
        relative_path = Path(
            "__extracted_archives/extracted_archives_Supplementary_data_for_the_paper_Get_out_of_the_way_Examining_eHMIs_in_critical_driver-pedestrian_encounters_in_a_coupled_si__2_all_data/data/Session1/World Root-HostFixedTimeLog-Pedestrian-2019_11_26_14_37_25.csv"
        )
        output_dir = Path("C:/tmp/output/fourtu_critical_ehmi")

        raw_prefix = _build_artifact_prefix(relative_path)
        shortened_prefix = _build_artifact_prefix(
            relative_path,
            output_dir=output_dir,
            artifact_suffix_length=len("-mapping-debug.json"),
        )

        self.assertGreater(len(raw_prefix), len(shortened_prefix))
        self.assertEqual(
            shortened_prefix,
            _build_artifact_prefix(
                relative_path,
                output_dir=output_dir,
                artifact_suffix_length=len("-mapping-debug.json"),
            ),
        )
        self.assertIn("_", shortened_prefix)

    def test_run_batch_shortens_long_artifact_prefixes_for_deep_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            long_relative = Path(
                "__extracted_archives/extracted_archives_Supplementary_data_for_the_paper_Get_out_of_the_way_Examining_eHMIs_in_critical_driver-pedestrian_encounters_in_a_coupled_si__2_all_data/data/Session1/World Root-HostFixedTimeLog-Pedestrian-2019_11_26_14_37_25.csv"
            )
            dataset_path = input_dir / long_relative
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Trust": [1.0, 2.0]}).to_csv(dataset_path, index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["**/*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            summary = run_batch(manifest_path, output_dir, SCHEMA_PATH, llm_deduction_enabled=False)

            source_output = output_dir / "standardized" / "study_source"
            standardized_files = list(source_output.glob("*-standardized.csv"))
            quality_files = list(source_output.glob("*-quality.json"))
            raw_prefix = _build_artifact_prefix(long_relative)

            self.assertEqual(summary["results"][0]["status"], "completed")
            self.assertEqual(summary["results"][0]["processed_files"], 1)
            self.assertEqual(len(standardized_files), 1)
            self.assertEqual(len(quality_files), 1)
            self.assertLess(len(standardized_files[0].name), len(f"{raw_prefix}-standardized.csv"))
            self.assertIn(long_relative.as_posix(), pd.read_csv(output_dir / "meta_view.csv")["dataset_id"].tolist())

    def test_discover_source_files_local_path_extracts_zip_tabular_files_in_nested_subfolders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            archive_dir = input_dir / "Study Data"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / "Study_Logs_Raw_Data.zip"

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("Studie_Logs_Raw_Data/01_car/Car_Critical_1.csv", "a,b\n1,2\n")
                zf.writestr("Studie_Logs_Raw_Data/01_car/readme.txt", "notes")

            source = {
                "source_id": "local_zip_source",
                "source_type": "local_path",
                "location": str(input_dir),
                "include_globs": ["**/*.csv"],
            }

            workdir = tmp / "work"
            workdir.mkdir(parents=True, exist_ok=True)
            base_dir, files, commit = discover_source_files(source, workdir)

            self.assertIsNone(commit)
            self.assertTrue(_paths_equal(base_dir, input_dir.resolve()))
            self.assertEqual(len(files), 1)
            self.assertIn("Car_Critical_1.csv", files[0].name)
            self.assertIn("__extracted_archives", files[0].parts)

    def test_discover_source_files_local_path_skips_macos_resource_forks_in_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            archive_path = input_dir / "study_data.zip"

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("__MACOSX/Study/._Car_Critical_1.csv", "macos metadata")
                zf.writestr("Study/.DS_Store", "finder metadata")
                zf.writestr("Study/Car_Critical_1.csv", "a,b\n1,2\n")

            source = {
                "source_id": "local_zip_source_filtered",
                "source_type": "local_path",
                "location": str(input_dir),
                "include_globs": ["**/*.csv"],
            }

            _, files, _ = discover_source_files(source, tmp / "work")

            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "Car_Critical_1.csv")

    def test_discover_source_files_local_path_respects_archive_max_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            archive_path = input_dir / "outer.zip"

            inner_buffer = io.BytesIO()
            with zipfile.ZipFile(inner_buffer, "w") as inner_zip:
                inner_zip.writestr("nested/deep.csv", "a,b\n1,2\n")

            with zipfile.ZipFile(archive_path, "w") as outer_zip:
                outer_zip.writestr("nested/inner.zip", inner_buffer.getvalue())

            source_depth_1 = {
                "source_id": "local_nested_zip_depth1",
                "source_type": "local_path",
                "location": str(input_dir),
                "include_globs": ["**/*.csv"],
                "archive_max_depth": 1,
            }
            source_depth_2 = {
                "source_id": "local_nested_zip_depth2",
                "source_type": "local_path",
                "location": str(input_dir),
                "include_globs": ["**/*.csv"],
                "archive_max_depth": 2,
            }

            workdir = tmp / "work"
            workdir.mkdir(parents=True, exist_ok=True)

            _, files_depth_1, _ = discover_source_files(source_depth_1, workdir)
            self.assertEqual(files_depth_1, [])

            _, files_depth_2, _ = discover_source_files(source_depth_2, workdir)
            self.assertEqual(len(files_depth_2), 1)
            self.assertIn("deep.csv", files_depth_2[0].name)

    def test_discover_source_files_rejects_invalid_archive_max_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "study.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            source = {
                "source_id": "invalid_archive_depth",
                "source_type": "local_path",
                "location": str(input_dir),
                "archive_max_depth": 0,
            }

            with self.assertRaises(ValueError):
                discover_source_files(source, tmp / "work")

    @unittest.skipUnless(
        os.environ.get("RUN_GITHUB_BATCH_INTEGRATION") == "1",
        "Set RUN_GITHUB_BATCH_INTEGRATION=1 to run live GitHub discovery checks.",
    )
    def test_discover_source_files_with_requested_public_repos(self):
        sources = load_manifest(EXAMPLE_MANIFEST_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            for source in sources:
                base_dir, files, commit_sha = discover_source_files(source, workdir)
                self.assertTrue(base_dir.exists())
                self.assertIsNotNone(commit_sha)
                self.assertRegex(commit_sha, r"^[0-9a-f]{40}$")
                self.assertGreater(len(files), 0, f"Expected tabular files for {source['source_id']}")

    def test_load_manifest_accepts_osf_project_source_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "osf_source",
                                "source_type": "osf_project",
                                "location": "https://osf.io/cwd6h/overview",
                                "include_globs": ["**/*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sources = load_manifest(manifest_path)
            self.assertEqual(sources[0]["source_type"], "osf_project")

    def test_example_manifest_contains_acm_chi26_supplemental_entry(self):
        with open(EXAMPLE_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        source_ids = [entry["source_id"] for entry in manifest["sources"]]
        self.assertIn("acm_chi26_3790738", source_ids)
        entry = next(
            e for e in manifest["sources"] if e["source_id"] == "acm_chi26_3790738"
        )
        self.assertEqual(entry["source_type"], "web_dataset")
        self.assertEqual(entry["publication_doi"], "10.1145/3772318.3790738")
        self.assertTrue(
            entry["location"].startswith("https://dl.acm.org/doi/suppl/10.1145/3772318.3790738/")
        )
        self.assertTrue(entry["location"].endswith(".zip"))

    def test_load_manifest_accepts_web_dataset_source_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "fourtu_source",
                                "source_type": "web_dataset",
                                "location": "https://data.4tu.nl/articles/_/20224281",
                                "include_globs": ["**/*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sources = load_manifest(manifest_path)
            self.assertEqual(sources[0]["source_type"], "web_dataset")

    def test_discover_source_files_github_tree_url_scopes_to_subpath_and_pickle_files(self):
        source = {
            "source_id": "touch_bumps",
            "source_type": "github_repo",
            "location": "https://github.com/interactionlab/Touch-Interaction-with-Road-Bumps/tree/master/data",
            "include_globs": ["**/*.csv", "**/*.pkl"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            def _fake_git_run(command, check, capture_output, text):
                if command[:3] == ["git", "clone", "--depth"]:
                    target = Path(command[-1])
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "root.csv").write_text("a,b\n0,1\n", encoding="utf-8")
                    data_dir = target / "data"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    (data_dir / "study.csv").write_text("a,b\n1,2\n", encoding="utf-8")
                    pd.DataFrame({"signal": [0.1, 0.2]}).to_pickle(data_dir / "sensor.pkl")
                    return mock.Mock(stdout="")
                if command[:4] == ["git", "-C", str(workdir / "touch_bumps"), "checkout"]:
                    return mock.Mock(stdout="")
                if command[:4] == ["git", "-C", str(workdir / "touch_bumps"), "rev-parse"]:
                    return mock.Mock(stdout=("a" * 40) + "\n")
                raise AssertionError(f"Unexpected git command: {command}")

            with mock.patch("scripts.run_batch_standardization.subprocess.run", side_effect=_fake_git_run):
                base_dir, files, commit_sha = discover_source_files(source, workdir)

        self.assertTrue(_paths_equal(base_dir, workdir / "touch_bumps" / "data"))
        self.assertEqual(sorted(path.name for path in files), ["sensor.pkl", "study.csv"])
        self.assertEqual(commit_sha, "a" * 40)

    def test_discover_source_files_web_dataset_downloads_4tu_style_archive(self):
        source = {
            "source_id": "fourtu_source",
            "source_type": "web_dataset",
            "location": "https://data.4tu.nl/articles/_/20224281",
            "include_globs": ["**/*.csv"],
        }

        page_headers = Message()
        page_headers["Content-Type"] = "text/html; charset=utf-8"
        zip_headers = Message()
        zip_headers["Content-Type"] = "application/zip"
        zip_headers["Content-Disposition"] = 'attachment; filename="dataset.zip"'
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("Study/results.csv", "a,b\n1,2\n")

        html = (
            '<html><head><title>4TU dataset</title>'
            '<script type="application/ld+json">'
            '{"distribution":{"contentUrl":"https://data.4tu.nl/ndownloader/items/test/versions/2"}}'
            "</script></head><body></body></html>"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            with mock.patch(
                "scripts.run_batch_standardization._read_url_response",
                side_effect=[
                    (html.encode("utf-8"), page_headers, source["location"]),
                    (zip_buffer.getvalue(), zip_headers, "https://data.4tu.nl/ndownloader/items/test/versions/2"),
                ],
            ):
                base_dir, files, commit_sha = discover_source_files(source, workdir)

        self.assertTrue(_paths_equal(base_dir, workdir / "fourtu_source"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "results.csv")
        self.assertEqual(commit_sha, source["location"])

    def test_discover_source_files_web_dataset_reports_cloudflare_challenge(self):
        source = {
            "source_id": "acm_chi26_3790738",
            "source_type": "web_dataset",
            "location": (
                "https://dl.acm.org/doi/suppl/10.1145/3772318.3790738/"
                "suppl_file/3772318.3790738-supplemental-material-1.zip"
            ),
            "include_globs": ["**/*.csv"],
        }
        import urllib.error as urlerror

        challenge_html = (
            b"<!DOCTYPE html><html><head><title>Just a moment...</title>"
            b"</head><body>Enable JavaScript and cookies to continue. "
            b"cf-mitigated: challenge</body></html>"
        )

        def _raise_challenge(*_args, **_kwargs):
            raise urlerror.HTTPError(
                source["location"],
                403,
                "Forbidden",
                Message(),
                io.BytesIO(challenge_html),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            with mock.patch(
                "scripts.run_batch_standardization._read_url_response",
                side_effect=_raise_challenge,
            ):
                with self.assertRaises(SourceAccessError) as ctx:
                    discover_source_files(source, workdir)

        self.assertEqual(ctx.exception.status, "access_restricted")
        self.assertIn("dl.acm.org", str(ctx.exception))

    def test_discover_source_files_web_dataset_reports_login_required(self):
        source = {
            "source_id": "ieee_source",
            "source_type": "web_dataset",
            "location": "https://ieee-dataport.org/open-access/usyd-campus-dataset",
            "include_globs": ["**/*.csv"],
        }
        page_headers = Message()
        page_headers["Content-Type"] = "text/html; charset=utf-8"
        html = "<html><body>LOGIN TO ACCESS DATASET FILES <a href='/saml_login'>Login</a></body></html>"

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            with mock.patch(
                "scripts.run_batch_standardization._read_url_response",
                return_value=(html.encode("utf-8"), page_headers, source["location"]),
            ):
                with self.assertRaises(SourceAccessError) as ctx:
                    discover_source_files(source, workdir)

        self.assertEqual(ctx.exception.status, "login_required")

    def test_extract_osf_project_id_accepts_url_and_raw_id(self):
        self.assertEqual(_extract_osf_project_id("cwd6h"), "cwd6h")
        self.assertEqual(_extract_osf_project_id("https://osf.io/cwd6h/overview"), "cwd6h")

    def test_extract_osf_project_id_rejects_invalid_location(self):
        with self.assertRaises(ValueError):
            _extract_osf_project_id("https://example.com/not-osf")

    def test_discover_source_files_osf_project_downloads_tabular_files(self):
        source = {
            "source_id": "osf_no_mapping",
            "source_type": "osf_project",
            "location": "https://osf.io/cwd6h/overview",
            "include_globs": ["**/*.csv"],
        }

        file_entries = [
            {
                "attributes": {"path": "/nested/study.csv", "kind": "file"},
                "links": {"download": "https://files.osf.io/study.csv"},
            },
            {
                "attributes": {"path": "/notes/readme.txt", "kind": "file"},
                "links": {"download": "https://files.osf.io/readme.txt"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            downloaded: list[Path] = []

            def _fake_download(url: str, destination: Path):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("a,b\n1,2\n", encoding="utf-8")
                downloaded.append(destination)

            with mock.patch(
                "scripts.run_batch_standardization._iter_osf_node_ids",
                return_value=["cwd6h"],
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._iter_osf_file_entries",
                    return_value=file_entries,
                ):
                    with mock.patch(
                        "scripts.run_batch_standardization._download_osf_file",
                        side_effect=_fake_download,
                    ):
                        base_dir, files, commit = discover_source_files(source, workdir)

            self.assertEqual(commit, "cwd6h")
            self.assertTrue(_paths_equal(base_dir, workdir / "osf_no_mapping"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith("study.csv"))
            self.assertEqual(len(downloaded), 1)

    def test_discover_source_files_osf_project_extracts_zip_tabular_files(self):
        source = {
            "source_id": "osf_zip_source",
            "source_type": "osf_project",
            "location": "https://osf.io/cwd6h/overview",
            "include_globs": ["**/*.csv"],
        }

        file_entries = [
            {
                "attributes": {"path": "/Study_Logs_Raw_Data.zip", "kind": "file"},
                "links": {"download": "https://files.osf.io/Study_Logs_Raw_Data.zip"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            def _fake_download(url: str, destination: Path):
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(destination, "w") as zf:
                    zf.writestr("Study_Logs/Observations.csv", "a,b\n1,2\n")
                    zf.writestr("Study_Logs/readme.txt", "notes")

            with mock.patch(
                "scripts.run_batch_standardization._iter_osf_node_ids",
                return_value=["cwd6h"],
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._iter_osf_file_entries",
                    return_value=file_entries,
                ):
                    with mock.patch(
                        "scripts.run_batch_standardization._download_osf_file",
                        side_effect=_fake_download,
                    ):
                        base_dir, files, commit = discover_source_files(source, workdir)

            self.assertEqual(commit, "cwd6h")
            self.assertTrue(_paths_equal(base_dir, workdir / "osf_zip_source"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith("Observations.csv"))

    def test_discover_source_files_osf_project_reads_component_nodes(self):
        source = {
            "source_id": "osf_component_source",
            "source_type": "osf_project",
            "location": "https://osf.io/cwd6h/overview",
            "include_globs": ["**/*.csv"],
        }

        component_entries = [
            {
                "attributes": {"path": "/Studie_Logs_Raw_Data/01_car/Car_Critical_1.csv", "kind": "file"},
                "links": {"download": "https://files.osf.io/component.csv"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            def _fake_download(url: str, destination: Path):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("a,b\n1,2\n", encoding="utf-8")

            with mock.patch(
                "scripts.run_batch_standardization._iter_osf_node_ids",
                return_value=["cwd6h", "node123"],
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._iter_osf_file_entries",
                    side_effect=[[], component_entries],
                ):
                    with mock.patch(
                        "scripts.run_batch_standardization._download_osf_file",
                        side_effect=_fake_download,
                    ):
                        base_dir, files, commit = discover_source_files(source, workdir)

            self.assertEqual(commit, "cwd6h")
            self.assertTrue(_paths_equal(base_dir, workdir / "osf_component_source"))
            self.assertEqual(len(files), 1)
            self.assertIn("Car_Critical_1.csv", files[0].name)

    def test_discover_source_files_osf_project_extracts_zip_from_component_node(self):
        source = {
            "source_id": "osf_component_zip_source",
            "source_type": "osf_project",
            "location": "https://osf.io/cwd6h/overview",
            "include_globs": ["**/*.csv"],
        }

        component_entries = [
            {
                "attributes": {"path": "/Study Data/Study_Logs_Raw_Data.zip", "kind": "file"},
                "links": {"download": "https://files.osf.io/component-zip"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            def _fake_download(url: str, destination: Path):
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(destination, "w") as zf:
                    zf.writestr("Studie_Logs_Raw_Data/01_car/Car_Critical_1.csv", "a,b\n1,2\n")

            with mock.patch(
                "scripts.run_batch_standardization._iter_osf_node_ids",
                return_value=["cwd6h", "node123"],
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._iter_osf_file_entries",
                    side_effect=[[], component_entries],
                ):
                    with mock.patch(
                        "scripts.run_batch_standardization._download_osf_file",
                        side_effect=_fake_download,
                    ):
                        base_dir, files, commit = discover_source_files(source, workdir)

            self.assertEqual(commit, "cwd6h")
            self.assertTrue(_paths_equal(base_dir, workdir / "osf_component_zip_source"))
            self.assertEqual(len(files), 1)
            self.assertIn("Car_Critical_1.csv", files[0].name)

    def test_run_batch_disables_llm_when_flag_is_off(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            input_dir.mkdir()

            pd.DataFrame({"DurationX": [1.0, 2.0]}).to_csv(input_dir / "study.csv", index=False)
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "study_source",
                                "source_type": "local_path",
                                "location": str(input_dir),
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                return_value="task_completion_time",
            ) as mocked:
                summary = run_batch(
                    manifest_path,
                    output_dir,
                    SCHEMA_PATH,
                    llm_deduction_enabled=False,
                )

            mocked.assert_not_called()
            self.assertEqual(summary["llm_deductions_count"], 0)
            self.assertIn(
                "No LLM-derived mappings were applied",
                Path(summary["llm_deductions_log"]).read_text(encoding="utf-8"),
            )
            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("DurationX", standardized_df.columns)

    def test_run_batch_uses_llm_for_osf_sources_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            osf_cache = tmp / "osf_source"
            osf_cache.mkdir(parents=True, exist_ok=True)
            study_file = osf_cache / "study.csv"
            pd.DataFrame({"DurationX": [1.0, 2.0]}).to_csv(study_file, index=False)

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "osf_source",
                                "source_type": "osf_project",
                                "location": "https://osf.io/cwd6h/overview",
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.discover_source_files",
                return_value=(osf_cache, [study_file], "cwd6h"),
            ):
                with mock.patch(
                    "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                    return_value="task_completion_time",
                ) as mocked:
                    run_batch(manifest_path, output_dir, SCHEMA_PATH)

            mocked.assert_called()
            standardized_path = output_dir / "standardized" / "osf_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("task_completion_time", standardized_df.columns)

    def test_run_batch_osf_prefers_mapping_and_skips_llm_when_mapping_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            osf_cache = tmp / "osf_source"
            osf_cache.mkdir(parents=True, exist_ok=True)
            study_file = osf_cache / "study.csv"
            pd.DataFrame({"DurationX": [1.0, 2.0]}).to_csv(study_file, index=False)
            (osf_cache / "source_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {"id": "task_completion_time", "aliases": ["DurationX"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "osf_source",
                                "source_type": "osf_project",
                                "location": "https://osf.io/cwd6h/overview",
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.discover_source_files",
                return_value=(osf_cache, [study_file], "cwd6h"),
            ):
                with mock.patch(
                    "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                    return_value="task_completion_time",
                ) as mocked:
                    run_batch(manifest_path, output_dir, SCHEMA_PATH)

            mocked.assert_not_called()
            standardized_path = output_dir / "standardized" / "osf_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("task_completion_time", standardized_df.columns)

    def test_run_batch_osf_uses_llm_after_mapping_for_unmapped_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            osf_cache = tmp / "osf_source"
            osf_cache.mkdir(parents=True, exist_ok=True)
            study_file = osf_cache / "study.csv"
            pd.DataFrame({"DurationX": [1.0, 2.0], "NovelDuration": [3.0, 4.0]}).to_csv(study_file, index=False)
            (osf_cache / "source_mapping.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dvs": [
                            {"id": "trust_rating", "aliases": ["DurationX"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "source_id": "osf_source",
                                "source_type": "osf_project",
                                "location": "https://osf.io/cwd6h/overview",
                                "include_globs": ["*.csv"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = tmp / "output"
            with mock.patch(
                "scripts.run_batch_standardization.discover_source_files",
                return_value=(osf_cache, [study_file], "cwd6h"),
            ):
                with mock.patch(
                    "scripts.run_batch_standardization.deduce_standard_name_with_local_llm",
                    return_value="task_completion_time",
                ) as mocked:
                    run_batch(manifest_path, output_dir, SCHEMA_PATH)

            mocked.assert_called_once()
            standardized_path = output_dir / "standardized" / "osf_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("trust_rating", standardized_df.columns)
            self.assertIn("task_completion_time", standardized_df.columns)

    def test_discover_source_files_osf_project_uses_cache(self):
        source = {
            "source_id": "osf_no_mapping",
            "source_type": "osf_project",
            "location": "https://osf.io/cwd6h/overview",
            "include_globs": ["**/*.csv"],
        }
        file_entries = [
            {
                "attributes": {"path": "/nested/study.csv", "kind": "file"},
                "links": {"download": "https://files.osf.io/study.csv"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "work"
            cache_root = Path(tmpdir) / "cache"
            workdir.mkdir(parents=True, exist_ok=True)

            def _fake_download(url: str, destination: Path):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("a,b\n1,2\n", encoding="utf-8")

            with mock.patch(
                "scripts.run_batch_standardization._iter_osf_node_ids",
                return_value=["cwd6h"],
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._iter_osf_file_entries",
                    return_value=file_entries,
                ):
                    with mock.patch(
                        "scripts.run_batch_standardization._download_osf_file",
                        side_effect=_fake_download,
                    ) as mocked_download:
                        _, files_first, commit_first = discover_source_files(
                            source,
                            workdir,
                            cache_root=cache_root,
                        )

            self.assertEqual(commit_first, "cwd6h")
            self.assertEqual(len(files_first), 1)
            self.assertEqual(mocked_download.call_count, 1)

            with mock.patch(
                "scripts.run_batch_standardization._iter_osf_node_ids",
                side_effect=AssertionError("OSF node discovery should not run when cache exists"),
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._iter_osf_file_entries",
                    side_effect=AssertionError("OSF listing should not run when cache exists"),
                ):
                    with mock.patch(
                        "scripts.run_batch_standardization._download_osf_file",
                        side_effect=AssertionError("download should not run when cache exists"),
                    ):
                        _, files_second, commit_second = discover_source_files(
                            source,
                            workdir,
                            cache_root=cache_root,
                        )

            self.assertEqual(commit_second, "cwd6h")
            self.assertEqual(len(files_second), 1)

class ArchiveExtractionTests(unittest.TestCase):
    """Cover the zip extraction helpers — zip-slip protection, depth limits,
    nested archives, and skip rules. These wrap _extract_zip_files_recursive
    and _extract_archives_in_tree, which previously lacked unit coverage."""

    def _make_zip(self, zip_path: Path, members: dict[str, bytes]) -> None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in members.items():
                zf.writestr(name, content)

    def test_extract_zip_skips_zip_slip_paths(self):
        from scripts.run_batch_standardization import _extract_zip_files_recursive

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "evil.zip"
            # The "../escape.csv" entry would land outside the extract root if
            # the helper did not guard against zip-slip. The harmless sibling
            # entry confirms extraction still succeeds for legitimate members.
            self._make_zip(
                zip_path,
                {
                    "../escape.csv": b"col\nval",
                    "data.csv": b"col\nval",
                },
            )

            _extract_zip_files_recursive(zip_path, root, depth=1, max_depth=3)

            extracted_root = root / "__extracted_archives" / "evil"
            self.assertTrue((extracted_root / "data.csv").is_file())
            # The escape attempt must not produce a file outside the root
            # (which is what `..` would resolve to relative to extract_root).
            self.assertFalse((root / "escape.csv").exists())
            self.assertFalse(zip_path.parent.parent.joinpath("escape.csv").exists())

    def test_extract_zip_respects_max_depth_for_nested_archives(self):
        from scripts.run_batch_standardization import _extract_zip_files_recursive

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inner_zip_bytes = io.BytesIO()
            with zipfile.ZipFile(inner_zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("deep.csv", b"deep,col\n1,2")
            inner_payload = inner_zip_bytes.getvalue()

            outer_zip = root / "outer.zip"
            self._make_zip(
                outer_zip,
                {
                    "shallow.csv": b"shallow,col\n1,2",
                    "inner.zip": inner_payload,
                },
            )

            # max_depth=1 means the outer archive is processed but inner.zip
            # must not be recursively extracted (only copied as a file).
            _extract_zip_files_recursive(outer_zip, root, depth=1, max_depth=1)

            extracted = root / "__extracted_archives" / "outer"
            self.assertTrue((extracted / "shallow.csv").is_file())
            # inner.zip is a recognized archive suffix, so it gets copied out;
            # the depth limit prevents further recursion into deep.csv.
            self.assertTrue((extracted / "inner.zip").is_file())
            self.assertFalse(
                (root / "__extracted_archives" / "inner" / "deep.csv").exists(),
                "Inner archive should not have been recursed into at max_depth=1.",
            )

    def test_extract_zip_recurses_into_nested_archives_when_depth_allows(self):
        from scripts.run_batch_standardization import _extract_zip_files_recursive

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inner_zip_bytes = io.BytesIO()
            with zipfile.ZipFile(inner_zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("deep.csv", b"deep,col\n1,2")
            inner_payload = inner_zip_bytes.getvalue()

            outer_zip = root / "outer.zip"
            self._make_zip(outer_zip, {"inner.zip": inner_payload})

            _extract_zip_files_recursive(outer_zip, root, depth=1, max_depth=3)

            # The inner archive id is derived from the relative path
            # __extracted_archives/outer/inner.zip → that path becomes the
            # archive id (slashes/dots normalised to underscores).
            candidates = list(
                (root / "__extracted_archives").glob("*/deep.csv")
            )
            self.assertEqual(
                len(candidates), 1,
                f"Expected exactly one extracted deep.csv, found: {candidates}",
            )

    def test_extract_zip_skips_macosx_metadata_and_hidden_files(self):
        from scripts.run_batch_standardization import _extract_zip_files_recursive

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "with_macos.zip"
            self._make_zip(
                zip_path,
                {
                    "data.csv": b"col\nval",
                    "__MACOSX/._data.csv": b"junk",
                    ".DS_Store": b"junk",
                    "subdir/._junkfile.csv": b"junk",
                },
            )

            _extract_zip_files_recursive(zip_path, root, depth=1, max_depth=3)

            extracted = root / "__extracted_archives" / "with_macos"
            self.assertTrue((extracted / "data.csv").is_file())
            self.assertFalse((extracted / "__MACOSX" / "._data.csv").exists())
            self.assertFalse((extracted / ".DS_Store").exists())
            self.assertFalse((extracted / "subdir" / "._junkfile.csv").exists())

    def test_extract_zip_ignores_unsupported_suffixes(self):
        from scripts.run_batch_standardization import _extract_zip_files_recursive

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "mixed.zip"
            self._make_zip(
                zip_path,
                {
                    "kept.csv": b"col\nval",
                    "ignored.exe": b"\x4d\x5a",
                    "ignored.txt": b"text only",
                },
            )

            _extract_zip_files_recursive(zip_path, root, depth=1, max_depth=3)

            extracted = root / "__extracted_archives" / "mixed"
            self.assertTrue((extracted / "kept.csv").is_file())
            self.assertFalse((extracted / "ignored.exe").exists())
            self.assertFalse((extracted / "ignored.txt").exists())

    def test_extract_archives_in_tree_skips_extracted_root(self):
        """The helper must not recursively re-process the __extracted_archives
        folder it just created, otherwise nested zips would loop."""
        from scripts.run_batch_standardization import _extract_archives_in_tree

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_zip(
                root / "outer.zip",
                {"data.csv": b"col\nval"},
            )

            _extract_archives_in_tree(root, max_depth=3)
            # Drop a synthetic re-archive into the extracted tree to simulate
            # what would happen if the helper accidentally re-processed it.
            sentinel_zip = root / "__extracted_archives" / "outer" / "duplicate.zip"
            self._make_zip(sentinel_zip, {"never_seen.csv": b"col\nval"})

            _extract_archives_in_tree(root, max_depth=3)

            self.assertFalse(
                (root / "__extracted_archives" / "duplicate" / "never_seen.csv").exists(),
                "duplicate.zip inside __extracted_archives must not be reprocessed.",
            )


class ReadUrlResponseTests(unittest.TestCase):
    """Cover the retry/backoff behaviour of _read_url_response. The function
    used to silently swallow the difference between transient and permanent
    failures; these tests pin the retry contract so future refactors do not
    regress it."""

    def _make_http_error(self, code: int) -> "Exception":
        msg = Message()
        return urlerror.HTTPError(
            url="https://example.com",
            code=code,
            msg="boom",
            hdrs=msg,
            fp=io.BytesIO(b""),
        )

    def test_retries_on_transient_5xx_and_eventually_succeeds(self):
        from scripts.run_batch_standardization import _read_url_response

        responses = [
            self._make_http_error(503),
            self._make_http_error(502),
            mock.MagicMock(),  # success on third attempt
        ]
        # The success response needs to behave like urlopen()'s context manager.
        success_resp = responses[2]
        success_resp.__enter__ = mock.MagicMock(return_value=success_resp)
        success_resp.__exit__ = mock.MagicMock(return_value=False)
        success_resp.read.return_value = b"ok"
        success_resp.headers = {"X": "y"}
        success_resp.geturl.return_value = "https://example.com/final"

        call_args: list = []

        def _fake_urlopen(req, timeout):
            call_args.append((req, timeout))
            outcome = responses[len(call_args) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with mock.patch("scripts.run_batch_standardization.request.urlopen", side_effect=_fake_urlopen):
            with mock.patch("scripts.run_batch_standardization.time.sleep") as sleep_mock:
                payload, headers, final_url = _read_url_response(
                    "https://example.com", timeout=10, max_attempts=4,
                )

        self.assertEqual(payload, b"ok")
        self.assertEqual(final_url, "https://example.com/final")
        self.assertEqual(len(call_args), 3)
        # Backoff schedule is 0.75 * (attempt + 1) — two sleeps between the
        # three attempts.
        self.assertEqual(sleep_mock.call_count, 2)

    def test_retries_on_timeout_then_raises_after_max_attempts(self):
        from scripts.run_batch_standardization import _read_url_response

        with mock.patch(
            "scripts.run_batch_standardization.request.urlopen",
            side_effect=socket.timeout("slow"),
        ):
            with mock.patch("scripts.run_batch_standardization.time.sleep"):
                with self.assertRaises(socket.timeout):
                    _read_url_response("https://example.com", timeout=1, max_attempts=2)

    def test_does_not_retry_on_permanent_404(self):
        from scripts.run_batch_standardization import _read_url_response

        permanent_error = self._make_http_error(404)
        with mock.patch(
            "scripts.run_batch_standardization.request.urlopen",
            side_effect=permanent_error,
        ) as urlopen_mock:
            with mock.patch("scripts.run_batch_standardization.time.sleep") as sleep_mock:
                with self.assertRaises(urlerror.HTTPError) as ctx:
                    _read_url_response("https://example.com", timeout=1, max_attempts=4)

        self.assertEqual(ctx.exception.code, 404)
        # No retry: 4xx responses (other than 429) are permanent errors.
        self.assertEqual(urlopen_mock.call_count, 1)
        sleep_mock.assert_not_called()

    def test_retries_on_429_throttle(self):
        from scripts.run_batch_standardization import _read_url_response

        responses = [self._make_http_error(429), self._make_http_error(429)]
        with mock.patch(
            "scripts.run_batch_standardization.request.urlopen",
            side_effect=responses,
        ) as urlopen_mock:
            with mock.patch("scripts.run_batch_standardization.time.sleep"):
                with self.assertRaises(urlerror.HTTPError) as ctx:
                    _read_url_response("https://example.com", timeout=1, max_attempts=2)

        self.assertEqual(ctx.exception.code, 429)
        self.assertEqual(urlopen_mock.call_count, 2)

    def test_max_attempts_defaults_to_network_timeouts_setting(self):
        from scripts.run_batch_standardization import (
            NETWORK_TIMEOUTS,
            _read_url_response,
        )

        # When the caller does not pass max_attempts, we fall back to the
        # NETWORK_TIMEOUTS["max_retry_attempts"] value so all callers benefit
        # from the central tuning knob.
        original = NETWORK_TIMEOUTS["max_retry_attempts"]
        try:
            NETWORK_TIMEOUTS["max_retry_attempts"] = 2
            with mock.patch(
                "scripts.run_batch_standardization.request.urlopen",
                side_effect=socket.timeout("slow"),
            ) as urlopen_mock:
                with mock.patch("scripts.run_batch_standardization.time.sleep"):
                    with self.assertRaises(socket.timeout):
                        _read_url_response("https://example.com", timeout=1)
            self.assertEqual(urlopen_mock.call_count, 2)
        finally:
            NETWORK_TIMEOUTS["max_retry_attempts"] = original


class WebDatasetChallengeTests(unittest.TestCase):
    """Ensure that the Cloudflare/CAPTCHA detection path correctly rejects
    challenge responses with `access_restricted` rather than treating them as
    successful HTML landing pages or generic failures."""

    def test_looks_like_challenge_response_detects_cloudflare_host(self):
        from scripts.run_batch_standardization import _looks_like_challenge_response

        self.assertTrue(_looks_like_challenge_response("dl.acm.org", "", ""))
        self.assertTrue(_looks_like_challenge_response("WWW.DL.ACM.ORG".lower(), "", ""))

    def test_looks_like_challenge_response_detects_title_marker(self):
        from scripts.run_batch_standardization import _looks_like_challenge_response

        self.assertTrue(
            _looks_like_challenge_response(
                "example.com", "Just a moment...", "<html></html>",
            )
        )

    def test_looks_like_challenge_response_detects_body_marker(self):
        from scripts.run_batch_standardization import _looks_like_challenge_response

        body = "<html><body>Please enable javascript and cookies to continue</body></html>"
        self.assertTrue(_looks_like_challenge_response("example.com", "OK", body))

    def test_looks_like_challenge_response_negative_case(self):
        from scripts.run_batch_standardization import _looks_like_challenge_response

        self.assertFalse(
            _looks_like_challenge_response(
                "data.example.org", "Dataset", "<html><body><a href='/file.csv'>get</a></body></html>",
            )
        )

    def test_build_web_dataset_access_error_returns_access_restricted_for_challenge(self):
        from scripts.run_batch_standardization import _build_web_dataset_access_error

        err = _build_web_dataset_access_error(
            location="https://dl.acm.org/doi/10.1145/x",
            final_url="https://dl.acm.org/doi/10.1145/x",
            html_text="<html><title>Just a moment...</title></html>",
            title="Just a moment...",
        )
        self.assertEqual(err.status, "access_restricted")
        self.assertIn("Cloudflare", str(err))

    def test_materialize_web_dataset_raises_access_restricted_on_challenge_403(self):
        """End-to-end: when a 4xx with Cloudflare markers comes back, the
        helper must classify the source as access_restricted instead of
        failed, so the run loop reports it cleanly without a traceback."""
        from scripts.run_batch_standardization import (
            SourceAccessError,
            _materialize_web_dataset_source,
        )

        challenge_html = (
            b"<html><head><title>Just a moment...</title></head>"
            b"<body>cf-mitigated</body></html>"
        )
        http_error = urlerror.HTTPError(
            url="https://dl.acm.org/doi/10.1145/x",
            code=403,
            msg="Forbidden",
            hdrs=Message(),
            fp=io.BytesIO(challenge_html),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir)
            with mock.patch(
                "scripts.run_batch_standardization._read_url_response",
                side_effect=http_error,
            ):
                with self.assertRaises(SourceAccessError) as ctx:
                    _materialize_web_dataset_source(
                        "https://dl.acm.org/doi/10.1145/x", target,
                    )

        self.assertEqual(ctx.exception.status, "access_restricted")

    def test_materialize_web_dataset_raises_login_required_for_ieee_dataport(self):
        from scripts.run_batch_standardization import (
            SourceAccessError,
            _materialize_web_dataset_source,
        )

        # Page contains a login-required marker but exposes NO downloadable
        # links — that combination must surface as `login_required` rather
        # than producing a generic failure or attempting blind download.
        html = (
            b"<html><head><title>IEEE DataPort</title></head>"
            b"<body>Login to access dataset files. Sign in to download.</body></html>"
        )
        headers = Message()
        headers["Content-Type"] = "text/html"

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir)
            with mock.patch(
                "scripts.run_batch_standardization._read_url_response",
                return_value=(html, headers, "https://ieee-dataport.org/datasets/example/"),
            ):
                with self.assertRaises(SourceAccessError) as ctx:
                    _materialize_web_dataset_source(
                        "https://ieee-dataport.org/datasets/example/", target,
                    )

        self.assertEqual(ctx.exception.status, "login_required")


if __name__ == "__main__":
    unittest.main()
