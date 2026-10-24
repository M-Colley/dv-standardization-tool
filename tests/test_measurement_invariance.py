"""Measurement invariance: decision rules, and lavaan fits on simulated data with known answers.

The simulation tests need Rscript with lavaan and jsonlite and are skipped
otherwise (e.g. in CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyses import measurement_invariance as mi

needs_lavaan = pytest.mark.skipif(not mi.lavaan_available(), reason="Rscript with lavaan not available")


def _simulate(seed: int, loadings_b=None, intercepts_b=None, n: int = 400, per_cluster: int = 2) -> pd.DataFrame:
    """Two groups, one factor, five items; group B may differ in loadings or intercepts."""
    rng = np.random.default_rng(seed)
    lam = np.array([0.8, 0.75, 0.7, 0.8, 0.65])
    nu = np.full(5, 3.0)
    frames = []
    for group, lam_g, nu_g, mean in (("A", lam, nu, 0.0),
                                     ("B", lam if loadings_b is None else np.asarray(loadings_b),
                                      nu if intercepts_b is None else np.asarray(intercepts_b), 0.3)):
        clusters = np.repeat(np.arange(n // per_cluster), per_cluster)
        person = rng.normal(mean, 1.0, n // per_cluster)[clusters]
        eta = 0.7 * person + rng.normal(0, 0.7, n)  # repeated answers share the person effect
        items = nu_g + np.outer(eta, lam_g) + rng.normal(0, 0.5, (n, 5))
        frame = pd.DataFrame(items, columns=[f"item{i + 1}" for i in range(5)])
        frame.insert(0, "cluster", [f"{group}{c}" for c in clusters])
        frame.insert(0, "group", group)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _level(frame: pd.DataFrame) -> str:
    result = mi.fit_levels(frame, "sim", [f"item{i + 1}" for i in range(5)])
    return mi.decide(result)[1]


@needs_lavaan
def test_equal_groups_reach_scalar_invariance():
    assert _level(_simulate(1)) == "scalar"


@needs_lavaan
def test_different_loadings_fail_metric_invariance():
    assert _level(_simulate(2, loadings_b=[0.8, 0.1, 0.1, 0.8, 1.4])) == "configural"


@needs_lavaan
def test_different_intercepts_fail_scalar_invariance():
    assert _level(_simulate(3, intercepts_b=[3.0, 4.0, 2.0, 3.0, 3.0])) == "metric"


@needs_lavaan
def test_item_names_survive_the_round_trip():
    frame = _simulate(4).rename(columns={"item1": "SUS1", "item2": "mental_demand"})
    result = mi.fit_levels(frame, "sim", ["SUS1", "mental_demand", "item3", "item4", "item5"])
    items = {row["item"] for row in result["levels"]["configural"]["std_loadings"]}
    assert items == {"SUS1", "mental_demand", "item3", "item4", "item5"}


def test_decision_rules():
    def level(cfi, rmsea, srmr, **extra):
        return {"cfi.robust": cfi, "tli.robust": cfi, "rmsea.robust": rmsea, "srmr": srmr,
                "chisq.scaled": 10.0, "df.scaled": 10.0, **extra}

    result = {"levels": {
        "configural": level(0.99, 0.03, 0.02),
        "metric": level(0.985, 0.04, 0.03),  # small drop: holds
        "scalar": level(0.95, 0.07, 0.06),   # dCFI -0.035 with dRMSEA +0.03: fails
    }}
    rows, reached = mi.decide(result)
    assert reached == "metric"
    assert [r["holds"] for r in rows] == [True, True, False]
    assert rows[2]["delta_cfi"] == pytest.approx(-0.035)

    misfit = {"levels": {"configural": level(0.85, 0.12, 0.09)}}
    rows, reached = mi.decide(misfit)
    assert reached == "none" and len(rows) == 1

    # a CFI drop alone, without a RMSEA/SRMR increase, does not reject (Chen, 2007)
    alone = {"levels": {"configural": level(0.99, 0.03, 0.02), "metric": level(0.975, 0.035, 0.025),
                        "scalar": level(0.97, 0.036, 0.026)}}
    assert mi.decide(alone)[1] == "scalar"


def test_invariance_by_dv_takes_the_scale_level_for_its_items():
    table = pd.DataFrame([
        {"scale": "s1", "dv": "usability", "items": "SUS1; SUS2", "level_reached": "metric", "studies": "a; b"},
        {"scale": "s2", "dv": "other", "items": "SUS2", "level_reached": "none", "studies": "a; c"},
    ])
    assert mi.invariance_by_dv(table) == {
        "usability": "metric (a; b)", "SUS1": "metric (a; b)", "SUS2": "none (a; c)", "other": "none (a; c)",
    }
