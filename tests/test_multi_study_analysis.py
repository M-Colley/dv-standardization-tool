import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyses.multi_study_analysis import (
    build_composite_index,
    load_studies,
    meta_analysis_summary,
)


class MultiStudyAnalysisTests(unittest.TestCase):
    def test_load_studies_treats_root_files_as_individual_studies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame({"task_success_rate": [0.8, 0.9]}).to_csv(root / "study_a.csv", index=False)
            pd.DataFrame({"task_success_rate": [0.6, 0.7]}).to_csv(root / "study_b.csv", index=False)

            studies = load_studies(root)

        self.assertEqual(set(studies.keys()), {"study_a", "study_b"})
        self.assertEqual(len(studies["study_a"]), 2)
        self.assertEqual(len(studies["study_b"]), 2)

    def test_meta_analysis_summary_returns_random_effects_metrics(self):
        summary = pd.DataFrame(
            [
                {"study": "a", "dv": "task_success_rate", "n": 100, "mean": 0.70, "sd": 0.10},
                {"study": "b", "dv": "task_success_rate", "n": 120, "mean": 0.80, "sd": 0.12},
                {"study": "c", "dv": "sus_score", "n": 50, "mean": 65.0, "sd": 10.0},
            ]
        )

        meta = meta_analysis_summary(summary)

        self.assertEqual(meta["dv"].tolist(), ["task_success_rate"])
        self.assertEqual(int(meta.loc[0, "k_studies"]), 2)
        self.assertGreater(meta.loc[0, "ci95_high"], meta.loc[0, "ci95_low"])
        self.assertGreaterEqual(meta.loc[0, "heterogeneity_i2_pct"], 0.0)

    def test_build_composite_index_handles_partially_shared_dvs(self):
        studies = {
            "study_a": pd.DataFrame(
                {
                    "task_success_rate": [0.8, 0.9, 0.85],
                    "sus_score": [70, 75, 80],
                }
            ),
            "study_b": pd.DataFrame(
                {
                    "task_success_rate": [0.6, 0.7, 0.65],
                    "sus_score": [60, 62, 64],
                }
            ),
            "study_c": pd.DataFrame(
                {
                    "task_success_rate": [0.75, 0.78, 0.8],
                    "nasa_tlx_score": [55, 50, 52],
                }
            ),
        }

        composite = build_composite_index(studies)

        self.assertEqual(set(composite["study"]), {"study_a", "study_b", "study_c"})
        self.assertEqual(set(composite["n_shared_dvs_used"]), {2})
        self.assertTrue(composite["mean"].notna().all())
        self.assertTrue(composite["explained_variance_ratio"].notna().all())

