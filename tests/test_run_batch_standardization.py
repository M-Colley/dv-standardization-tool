import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_batch_standardization import (
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

    def test_load_repository_mapping_ignores_ambiguous_mapping_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            (source_root / "a_mapping.yaml").write_text("dvs: []", encoding="utf-8")
            (source_root / "b_mapping.yaml").write_text("dvs: []", encoding="utf-8")

            merged_mapping, mapping_path = _load_repository_mapping(source_root)

            self.assertEqual(merged_mapping, {})
            self.assertIsNone(mapping_path)

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

            standardized_path = output_dir / "standardized" / "semicolon_source" / "ObservationsPerEvaluation-standardized.csv"
            standardized_df = pd.read_csv(standardized_path)

            self.assertIn("mental_effort", standardized_df.columns)
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

            meta_view_path = output_dir / "meta_view.csv"
            run_summary_path = output_dir / "run_summary.json"
            standardized_path = output_dir / "standardized" / "study_source" / "study-standardized.csv"
            quality_path = output_dir / "standardized" / "study_source" / "study-quality.json"

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

            standardized_path = output_dir / "standardized" / "study_source" / "study-standardized.csv"
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


if __name__ == "__main__":
    unittest.main()
