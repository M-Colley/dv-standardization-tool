import json
import os
import zipfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_batch_standardization import (
    _augment_mapping_with_llm_deductions,
    _extract_osf_project_id,
    _load_repository_mapping,
    discover_source_files,
    load_manifest,
    run_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"
EXAMPLE_MANIFEST_PATH = REPO_ROOT / "sources_manifest_example.yaml"

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

    def test_augment_mapping_with_llm_deductions_adds_inferred_aliases(self):
        mapping = {"task_time": "task_completion_time", "task_completion_time": "task_completion_time"}

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
                run_batch(manifest_path, output_dir, SCHEMA_PATH)

            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)
            self.assertIn("task_completion_time", standardized_df.columns)

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

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["source_id"], "roads_chi25")
        self.assertEqual(
            sources[0]["location"],
            "https://github.com/M-Colley/roads-chi25-data",
        )
        self.assertEqual(sources[1]["source_id"], "ehmi_optimization_chi25")
        self.assertEqual(
            sources[1]["location"],
            "https://github.com/M-Colley/ehmi-optimization-chi25-data",
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
            summary = run_batch(manifest_path, output_dir, SCHEMA_PATH)

            self.assertEqual(summary["total_sources"], 1)
            self.assertEqual(summary["successful_sources"], 1)
            self.assertIn("mapped_ratio", summary["results"][0])
            self.assertGreaterEqual(summary["results"][0]["mapped_ratio"], 0.0)

            meta_view_path = output_dir / "meta_view.csv"
            run_summary_path = output_dir / "run_summary.json"
            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            quality_path = output_dir / "standardized" / "study_source" / "study_csv-quality.json"

            self.assertTrue(meta_view_path.exists())
            self.assertTrue(run_summary_path.exists())
            self.assertTrue(standardized_path.exists())
            self.assertTrue(quality_path.exists())

            meta_df = pd.read_csv(meta_view_path)
            self.assertIn("canonical_dv", meta_df.columns)
            self.assertIn("source_id", meta_df.columns)
            self.assertIn("task_completion_time", meta_df["canonical_dv"].values)

            quality = json.loads(quality_path.read_text())
            self.assertEqual(quality["unknown_columns"], 1)
            self.assertIn("Unknown Alias", quality["unknown_aliases"])

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
            run_batch(manifest_path, output_dir, SCHEMA_PATH)

            standardized_path = output_dir / "standardized" / "study_source" / "study_csv-standardized.csv"
            meta_view_path = output_dir / "meta_view.csv"

            standardized_df = pd.read_csv(standardized_path)
            meta_df = pd.read_csv(meta_view_path)

            self.assertIn("task_completion_time", standardized_df.columns)
            self.assertIn("custom_task_time", standardized_df.columns)
            self.assertNotIn("duration", standardized_df.columns)
            self.assertEqual(
                meta_df.loc[meta_df["canonical_dv"] == "task_completion_time", "source_mapping"].iat[0],
                str(input_dir / "source_mapping.yaml"),
            )
            self.assertIn("Unmapped", standardized_df.columns)

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
                "scripts.run_batch_standardization._iter_osf_file_entries",
                return_value=file_entries,
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._download_osf_file",
                    side_effect=_fake_download,
                ):
                    base_dir, files, commit = discover_source_files(source, workdir)

            self.assertEqual(commit, "cwd6h")
            self.assertEqual(base_dir, workdir / "osf_no_mapping")
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
                "scripts.run_batch_standardization._iter_osf_file_entries",
                return_value=file_entries,
            ):
                with mock.patch(
                    "scripts.run_batch_standardization._download_osf_file",
                    side_effect=_fake_download,
                ):
                    base_dir, files, commit = discover_source_files(source, workdir)

            self.assertEqual(commit, "cwd6h")
            self.assertEqual(base_dir, workdir / "osf_zip_source")
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith("Observations.csv"))

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
                run_batch(
                    manifest_path,
                    output_dir,
                    SCHEMA_PATH,
                    llm_deduction_enabled=False,
                )

            mocked.assert_not_called()
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

if __name__ == "__main__":
    unittest.main()

