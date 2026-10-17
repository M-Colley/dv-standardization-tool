"""Cross-validation of the meta-analysis estimators against R's metafor.

``tests/fixtures/metafor_reference.json`` holds effect sizes and metafor
(5.2.1) results for two published datasets bundled with metafor:

* ``dat.hackshaw1998`` — 37 studies on passive smoking and lung cancer
  (log odds ratios); the classic trim-and-fill example (k0 = 7, left).
* ``dat.bcg`` (Colditz et al., 1994) — 13 BCG vaccine trials (log risk
  ratios); highly heterogeneous, imputes on the right.

Regenerate the fixture with ``Rscript`` + metafor if the estimators change:
``rma(yi, vi, method = "FE"|"DL"|"REML")``, ``trimfill(<model>)``,
``trimfill(<FE model>, estimator = "R0")`` and
``regtest(<FE model>, model = "lm", predictor = "sei")`` and
``rma(..., test = "knha")`` for the Hartung-Knapp-Sidik-Jonkman intervals.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses import multi_study_analysis as msa  # noqa: E402

FIXTURE = json.loads((REPO_ROOT / "tests" / "fixtures" / "metafor_reference.json").read_text(encoding="utf-8"))
DATASETS = {ds["name"]: ds for ds in FIXTURE["datasets"]}
# Express (yi, vi) as the (mean, sd, n) rows the pipeline pools: vi = sd² / n.
_N = 100.0


def _summary(ds: dict) -> pd.DataFrame:
    yi = np.asarray(ds["yi"], dtype=float)
    vi = np.asarray(ds["vi"], dtype=float)
    return pd.DataFrame({
        "study": [f"s{i:02d}" for i in range(len(yi))],
        "dv": "effect",
        "n": _N,
        "mean": yi,
        "sd": np.sqrt(vi * _N),
    })


def _reml_by_optimisation(yi: np.ndarray, vi: np.ndarray) -> float:
    def negative_restricted_loglik(tau2: float) -> float:
        w = 1.0 / (vi + tau2)
        mu = np.sum(w * yi) / np.sum(w)
        return 0.5 * (np.sum(np.log(vi + tau2)) + np.log(np.sum(w)) + np.sum(w * (yi - mu) ** 2))

    return float(minimize_scalar(negative_restricted_loglik, bounds=(0.0, 10.0), method="bounded",
                                 options={"xatol": 1e-12}).x)


class MetaforReferenceTests(unittest.TestCase):
    def test_dersimonian_laird_matches_metafor(self):
        for name, ds in DATASETS.items():
            with self.subTest(dataset=name):
                row = msa.meta_analysis_summary(_summary(ds), estimator="DL").iloc[0]
                ref = ds["dl"]
                self.assertAlmostEqual(row["random_effects_mean"], ref["mu"], places=10)
                self.assertAlmostEqual(row["random_effects_se"], ref["se"], places=10)
                self.assertAlmostEqual(row["tau2"], ref["tau2"], places=10)
                self.assertAlmostEqual(row["heterogeneity_q"], ref["q"], places=8)
                self.assertAlmostEqual(row["heterogeneity_i2_pct"], ref["i2"], places=8)

    def test_hartung_knapp_interval_matches_metafor_knha(self):
        for name, ds in DATASETS.items():
            with self.subTest(dataset=name, k="all"):
                row = msa.meta_analysis_summary(_summary(ds), estimator="DL").iloc[0]
                ref = ds["dl_knha"]
                self.assertAlmostEqual(row["hksj_se"], ref["se"], places=10)
                self.assertAlmostEqual(row["hksj_ci95_low"], ref["ci_lb"], places=10)
                self.assertAlmostEqual(row["hksj_ci95_high"], ref["ci_ub"], places=10)
            with self.subTest(dataset=name, k=2):
                # k < 3 is pooled with fixed effects; HKSJ still applies (t with 1 df).
                row = msa.meta_analysis_summary(_summary(ds).iloc[:2], estimator="DL").iloc[0]
                ref = ds["fe_knha_first2"]
                self.assertAlmostEqual(row["random_effects_mean"], ref["mu"], places=10)
                self.assertAlmostEqual(row["hksj_ci95_low"], ref["ci_lb"], places=8)
                self.assertAlmostEqual(row["hksj_ci95_high"], ref["ci_ub"], places=8)

    def test_reml_reaches_the_restricted_likelihood_maximum(self):
        for name, ds in DATASETS.items():
            with self.subTest(dataset=name):
                yi = np.asarray(ds["yi"])
                vi = np.asarray(ds["vi"])
                tau2 = msa._estimate_tau2_reml(yi, vi)
                optimum = _reml_by_optimisation(yi, vi)
                self.assertLess(abs(tau2 - optimum) / optimum, 1e-6)
                # metafor stops Fisher scoring at a 1e-5 change in tau², so it
                # agrees to ~5 significant digits.
                self.assertLess(abs(tau2 - ds["reml"]["tau2"]) / ds["reml"]["tau2"], 1e-4)

    def test_trim_and_fill_matches_metafor(self):
        cases = [("FE", "L0", "trimfill_fe_l0"), ("DL", "L0", "trimfill_dl_l0"), ("FE", "R0", "trimfill_fe_r0")]
        for name, ds in DATASETS.items():
            for pooling, k0_estimator, key in cases:
                with self.subTest(dataset=name, pooling=pooling, k0_estimator=k0_estimator):
                    result = msa.trim_and_fill(_summary(ds), "effect", estimator=pooling, k0_estimator=k0_estimator)
                    ref = ds[key]
                    self.assertEqual(result["k_imputed"], ref["k0"])
                    self.assertEqual(result["side"], ref["side"])
                    self.assertAlmostEqual(result["adjusted_mean"], ref["mu"], places=10)
                    self.assertAlmostEqual(result["adjusted_se"], ref["se"], places=10)
                    self.assertEqual(result["k_total"], len(ds["yi"]) + ref["k0"])

    def test_trim_and_fill_unadjusted_estimate_is_the_pooled_model(self):
        ds = DATASETS["bcg_colditz1994"]
        fe = msa.trim_and_fill(_summary(ds), "effect", estimator="FE")
        dl = msa.trim_and_fill(_summary(ds), "effect", estimator="DL")
        self.assertAlmostEqual(fe["original_mean"], ds["fe"]["mu"], places=10)
        self.assertAlmostEqual(dl["original_mean"], ds["dl"]["mu"], places=10)

    def test_eggers_test_matches_metafor_regtest(self):
        for name, ds in DATASETS.items():
            with self.subTest(dataset=name):
                result = msa.eggers_test(_summary(ds), "effect")
                self.assertAlmostEqual(result["t_stat"], ds["egger"]["z"], places=8)
                self.assertAlmostEqual(result["p_value"], ds["egger"]["p"], places=8)


class TrimAndFillEdgeCaseTests(unittest.TestCase):
    def test_symmetric_funnel_imputes_nothing(self):
        summary = pd.DataFrame({
            "study": list("abcdefg"),
            "dv": "x",
            "n": [20, 40, 80, 160, 80, 40, 20],
            "mean": [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6],
            "sd": [1.0] * 7,
        })
        result = msa.trim_and_fill(summary, "x", estimator="FE")
        self.assertEqual(result["k_imputed"], 0)
        self.assertAlmostEqual(result["adjusted_mean"], result["original_mean"], places=12)

    def test_equal_standard_errors_default_to_left_side(self):
        # The small-study slope is undefined when every SE is identical;
        # metafor falls back to imputing on the left.
        summary = pd.DataFrame({
            "study": list("abcd"), "dv": "x", "n": [50] * 4,
            "mean": [3.0, 3.1, 3.2, 5.0], "sd": [0.5] * 4,
        })
        result = msa.trim_and_fill(summary, "x")
        self.assertEqual(result["side"], "left")

    def test_rejects_unknown_k0_estimator(self):
        with self.assertRaises(ValueError):
            msa.trim_and_fill(_summary(DATASETS["hackshaw1998"]), "effect", k0_estimator="Q9")


if __name__ == "__main__":
    unittest.main()
