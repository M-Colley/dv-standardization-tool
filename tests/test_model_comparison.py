"""Tests for running E4B through the benchmark and comparing it with E2B (docs/model_comparison_design.md)."""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd
import pytest

from opendv import llm_utils
from opendv.analysis import evaluate_llm_mapping as ev
from opendv.analysis import model_comparison as mc

E2B, E4B = "google/gemma-4-E2B-it", "google/gemma-4-E4B-it"


@pytest.fixture(autouse=True)
def clean_llm_state(monkeypatch):
    """No placement, cached pipeline or dropped-model warning leaks between tests."""
    monkeypatch.delenv("DV_LLM_MODELS", raising=False)
    llm_utils.clear_cpu_ple()
    llm_utils._WARNED_DROPPED_MODELS.clear()
    yield
    llm_utils.clear_cpu_ple()
    llm_utils._PIPELINE_CACHE.clear()
    llm_utils._WARNED_DROPPED_MODELS.clear()


@pytest.fixture
def gpu_16gb(monkeypatch):
    """A 16 GB GPU, as the RTX 5080 the committed runs used (14.4 GB after the headroom factor)."""
    monkeypatch.setattr(llm_utils, "_detect_cuda_memory_gb", lambda: 16.0)


# ── Model selection ──────────────────────────────────────────────────────────


def test_default_selection_on_16gb_is_unchanged(gpu_16gb):
    assert llm_utils.select_local_model_candidates() == [E2B]
    assert llm_utils._estimate_model_min_memory_gb(E4B) == 18.0
    assert llm_utils._estimate_model_min_memory_gb(E2B) == 10.0


def test_explicitly_requested_e4b_is_dropped_with_a_warning(gpu_16gb, monkeypatch, caplog):
    monkeypatch.setenv("DV_LLM_MODELS", E4B)
    with caplog.at_level(logging.WARNING, logger="opendv.llm_utils"):
        assert llm_utils.select_local_model_candidates() == []
        assert llm_utils.select_local_model_candidates() == []
    warnings = [r for r in caplog.records if E4B in r.getMessage()]
    assert len(warnings) == 1  # once per model, not once per column
    assert "configure_cpu_ple" in warnings[0].getMessage()


def test_default_list_drops_e4b_without_a_warning(gpu_16gb, caplog):
    with caplog.at_level(logging.WARNING, logger="opendv.llm_utils"):
        llm_utils.select_local_model_candidates()
    assert not caplog.records


def test_cpu_ple_placement_keeps_e4b_and_leaves_e2b_alone(gpu_16gb, monkeypatch, tmp_path):
    placement = llm_utils.configure_cpu_ple(E4B, tmp_path / "weights", tmp_path / "tokenizer")
    assert placement.min_memory_gb == llm_utils.CPU_PLE_MIN_MEMORY_GB[E4B] == 12.0
    assert llm_utils.cpu_ple_placement(E4B) == placement
    assert llm_utils.cpu_ple_placement(E2B) is None
    monkeypatch.setenv("DV_LLM_MODELS", E4B)
    assert llm_utils.select_local_model_candidates() == [E4B]
    monkeypatch.delenv("DV_LLM_MODELS")
    assert llm_utils.select_local_model_candidates([E4B]) == [E4B]
    # The default list still prefers E4B where it fits, which it now does.
    assert llm_utils.select_local_model_candidates() == [E4B, E2B]
    assert llm_utils.MODEL_MIN_MEMORY_GB == {E4B: 18.0, E2B: 10.0}  # the whole-model estimates stay
    llm_utils.clear_cpu_ple(E4B)
    assert llm_utils.select_local_model_candidates() == [E2B]


def test_cpu_ple_needs_a_memory_estimate(tmp_path):
    with pytest.raises(ValueError, match="min_memory_gb"):
        llm_utils.configure_cpu_ple("some/model", tmp_path, tmp_path)
    placement = llm_utils.configure_cpu_ple("some/model", tmp_path, tmp_path, min_memory_gb=3.0)
    assert llm_utils._estimate_model_min_memory_gb("some/model") == 3.0
    assert placement.device == "cuda:0"


def test_pipeline_loader_uses_the_placement(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(llm_utils, "_load_cpu_ple_pipeline", lambda placement: loaded.append(placement) or "pipe")
    llm_utils.configure_cpu_ple(E4B, tmp_path / "w", tmp_path / "t")
    assert llm_utils._get_text_generation_pipeline(E4B) == "pipe"
    assert llm_utils._get_text_generation_pipeline(E4B) == "pipe"  # cached
    assert [p.weights_dir.name for p in loaded] == ["w"]
    assert E4B in llm_utils.models_used()
    # A new placement drops the cached pipeline, so the next call loads with it.
    llm_utils.configure_cpu_ple(E4B, tmp_path / "w2", tmp_path / "t")
    assert E4B not in llm_utils._PIPELINE_CACHE
    llm_utils._get_text_generation_pipeline(E4B)
    assert [p.weights_dir.name for p in loaded] == ["w", "w2"]


def test_schema_stability_install_delegates(monkeypatch, tmp_path):
    from opendv.analysis import schema_stability as ss

    calls = []
    monkeypatch.setattr(llm_utils, "_load_cpu_ple_pipeline", lambda placement: calls.append(placement) or "pipe")
    ss.install_cpu_ple_pipeline(E4B, tmp_path / "w", tmp_path / "t")
    assert len(calls) == 1 and llm_utils._PIPELINE_CACHE[E4B] == "pipe"
    assert llm_utils._estimate_model_min_memory_gb(E4B) == 12.0


# ── Per-layer embeddings on the CPU (a tiny random Gemma 4) ──────────────────


def tiny_gemma4():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("transformers without Gemma 4")
    config = transformers.Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, global_head_dim=16, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, layer_types=["sliding_attention", "full_attention"], sliding_window=4,
        per_layer_config={},
    )
    torch.manual_seed(0)
    return transformers.Gemma4ForCausalLM(config).eval()


def test_cpu_lookup_gives_the_same_logits_on_cpu():
    import torch

    model = tiny_gemma4()
    ids = torch.tensor([[2, 5, 9, 3, 7, 11]])
    with torch.no_grad():
        before = model(ids).logits.clone()
    table = model.model.embed_tokens_per_layer
    n_params = sum(p.numel() for p in model.parameters())
    llm_utils.place_per_layer_embeddings_on_cpu(model, "cpu")
    with torch.no_grad():
        after = model(ids).logits
    assert torch.equal(before, after)
    lookup = model.model.embed_tokens_per_layer
    assert lookup.embedding is table
    # The table is no submodule any more: .to() and parameters() leave it out.
    assert sum(p.numel() for p in model.parameters()) == n_params - table.weight.numel()


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
def test_cpu_lookup_matches_the_model_wholly_on_the_gpu():
    import copy

    import torch

    model = tiny_gemma4()
    whole = copy.deepcopy(model).to("cuda:0")
    ids = torch.tensor([[2, 5, 9, 3, 7, 11]], device="cuda:0")
    llm_utils.place_per_layer_embeddings_on_cpu(model, "cuda:0")
    assert model.model.embed_tokens_per_layer.embedding.weight.device.type == "cpu"
    assert model.lm_head.weight.device.type == "cuda"
    with torch.no_grad():
        assert torch.equal(whole(ids).logits, model(ids).logits)


def test_placement_needs_exactly_one_table():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="found 0"):
        llm_utils.place_per_layer_embeddings_on_cpu(torch.nn.Linear(2, 2), "cpu")


# ── The benchmark's model check ──────────────────────────────────────────────


def test_benchmark_refuses_a_requested_model_it_would_not_use(gpu_16gb, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="--cpu-ple"):
        ev.check_requested_models([E4B])
    monkeypatch.setenv("DV_LLM_MODELS", E2B)
    with pytest.raises(ValueError, match="DV_LLM_MODELS"):
        ev.check_requested_models([E4B])
    monkeypatch.delenv("DV_LLM_MODELS")
    llm_utils.configure_cpu_ple(E4B, tmp_path / "fee6332", tmp_path / "ee0ef60")
    assert ev.check_requested_models([E4B]) == [E4B]
    assert ev.check_requested_models([E2B]) == [E2B]
    assert ev.model_placement([E4B, E2B]) == {
        E4B: {"per_layer_embeddings": "cpu", "weights_snapshot": "fee6332", "tokenizer_snapshot": "ee0ef60",
              "device": "cuda:0"},
        E2B: {"per_layer_embeddings": "default"},
    }


def test_benchmark_refuses_to_run_without_any_model(monkeypatch):
    monkeypatch.setattr(llm_utils, "_detect_cuda_memory_gb", lambda: 4.0)
    with pytest.raises(ValueError, match="no LLM fits"):
        ev.check_requested_models(None)


def test_benchmark_default_models_unchanged(gpu_16gb):
    assert ev.check_requested_models(None) == [E2B]


# ── The analysis on synthetic runs ───────────────────────────────────────────

DV_IDS = {"a", "b", "c"}
STUDIES = ["s1", "s2", "s3", "s4"]


def e1_items():
    """Per study: two DV columns (gold a, b) and two curated non-DV columns; one blocklisted column."""
    rows = []
    for half, studies in (("first_23", STUDIES[:2]), ("added_16", STUDIES[2:])):
        for study in studies:
            for i, gold in enumerate(["a", "b"]):
                rows.append({"half": half, "source_id": study, "dataset_id": "d.csv", "alias": f"dv{i}",
                             "label_kind": "dv", "gold": gold})
            for i in range(2):
                rows.append({"half": half, "source_id": study, "dataset_id": "d.csv", "alias": f"meta{i}",
                             "label_kind": "curated_non_dv", "gold": None})
            rows.append({"half": half, "source_id": study, "dataset_id": "d.csv", "alias": "id",
                         "label_kind": "blocklisted_non_dv", "gold": None})
    return pd.DataFrame(rows)


def answers_for(model: str, items: pd.DataFrame) -> list:
    """E2B: right on one DV column per study, maps one non-DV column to 'c'. E4B: right on both, maps none."""
    out = []
    for item in items.itertuples():
        if item.label_kind == "dv":
            out.append(item.gold if (model == "e4b" or item.alias == "dv0") else "c")
        elif item.label_kind == "curated_non_dv":
            out.append("c" if (model == "e2b" and item.alias == "meta0") else None)
        else:
            out.append("a")
    return out


def write_runs(root, answers=answers_for, errors=0, e2_answers=None):
    base = e1_items()
    e2 = pd.DataFrame({"alias": [f"x{i}" for i in range(6)], "gold": ["a", "a", "b", "b", "c", "c"]})
    for model, shortlist, half in itertools.product(mc.MODELS, mc.SHORTLISTS, mc.HALVES):
        items = base[base["half"] == half].copy()
        items["suggestion"] = answers(model, items)
        items = items.assign(experiment="E1", final_unchecked=items["suggestion"], final_checked=items["suggestion"],
                             gold_in_shortlist=True, prompt_sha256=[f"{model}-{half}-{i}" for i in range(len(items))],
                             seconds=0.5)
        frames = [items.drop(columns="half")]
        if half == "first_23":
            e2_items = e2.assign(experiment="E2", label_kind="dv", gold_in_shortlist=True)
            hit = e2_answers(model) if e2_answers else [model == "e4b" or i % 2 == 0 for i in range(6)]
            e2_items["suggestion"] = [g if h else None for g, h in zip(e2_items["gold"], hit, strict=True)]
            e2_items["final_unchecked"] = e2_items["final_checked"] = e2_items["suggestion"]
            frames.append(e2_items)
        folder = mc.cell_dir(root, model, shortlist, half)
        folder.mkdir(parents=True)
        pd.concat(frames, ignore_index=True).to_csv(folder / "llm_mapping_evaluation_items.csv", index=False)
        summary = {"models": [mc.MODELS[model]], "shortlist": shortlist, "llm_generation_errors": errors,
                   "runtime_seconds": 1.0, "cuda_peak_memory_gb": None, "model_placement": {}}
        (folder / "llm_mapping_evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_outcomes_and_comparisons(tmp_path):
    write_runs(tmp_path)
    items, _ = mc.load_runs(tmp_path)
    assert mc.check_same_items(items) == {"E1": 20, "E2": 6}
    table = mc.compare_models(items, DV_IDS, n_boot=500, n_perm=2000)
    assert len(table) == 16 and set(table["family"]) == {"E1", "E2"}
    row = table[(table.shortlist == "definition") & (table.answer == "checked")].set_index("outcome")
    assert row.loc["accuracy", ["rate_reference", "rate_candidate", "difference"]].tolist() == [0.5, 1.0, 0.5]
    assert row.loc["accuracy", ["n_candidate_only", "n_reference_only"]].tolist() == [4, 0]
    assert row.loc["non_dv_mapped", ["rate_reference", "rate_candidate"]].tolist() == [0.5, 0.0]
    # E2B accepts 4 right + 4 wrong DV answers on DV columns and 4 on non-DV columns: 4/12; E4B 8/8.
    assert row.loc["precision", ["k_reference", "n_reference", "k_candidate", "n_candidate"]].tolist() == [4, 12, 8, 8]
    assert row.loc["precision", "difference"] == pytest.approx(1 - 4 / 12)
    assert row.loc["e2_accuracy", ["rate_reference", "rate_candidate", "n_clusters"]].tolist() == [0.5, 1.0, 3]
    # Every study gains: the bootstrap cannot draw anything else, and the permutation p is 2 / 2^4 at best.
    assert row.loc["accuracy", ["ci_low", "ci_high"]].tolist() == [0.5, 0.5]
    assert row.loc["accuracy", "p"] == pytest.approx(2 / 16, abs=0.03)
    e1 = table[table.family == "E1"]
    p = e1["p"].to_numpy()
    order = np.argsort(p)
    expected = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    assert e1["p_holm"].to_numpy()[order] == pytest.approx(expected)


def test_rates_scopes(tmp_path):
    write_runs(tmp_path)
    items, _ = mc.load_runs(tmp_path)
    earlier = {("first_23", s, "d.csv", a) for s in STUDIES[:2] for a in ("dv0", "dv1", "meta0", "meta1", "id")}
    table = mc.rates(items, DV_IDS, earlier)
    e2b = table[(table.model == E2B) & (table.shortlist == "definition") & (table.answer == "checked")]
    acc = e2b[e2b.outcome == "accuracy"].set_index("scope")
    assert acc.loc["pooled", ["k", "n"]].tolist() == [4, 8]
    assert acc.loc["first_23", "n"] == 4 and acc.loc["added_16", "n"] == 4
    assert acc.loc["new_since_schema_stability", ["k", "n"]].tolist() == [2, 4]
    wrong = e2b[(e2b.outcome == "wrong_dv_on_dv") & (e2b.scope == "pooled")]
    assert wrong[["k", "n"]].values.tolist() == [[4, 8]]
    seconds = table[(table.outcome == "seconds_per_prompt") & (table.model == E4B)]
    assert seconds["rate"].tolist() == [0.5, 0.5] and seconds["n"].tolist() == [20, 20]


def test_load_runs_refuses_incomplete_or_failed_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        mc.load_runs(tmp_path)
    write_runs(tmp_path / "errors", errors=2)
    with pytest.raises(ValueError, match="generation errors"):
        mc.load_runs(tmp_path / "errors")


def test_check_same_items_stops_on_a_different_item(tmp_path):
    write_runs(tmp_path)
    items, _ = mc.load_runs(tmp_path)
    changed = items.copy()
    index = changed[(changed.model == "e4b") & (changed.experiment == "E1")].index[0]
    changed.loc[index, "alias"] = "renamed"
    with pytest.raises(ValueError, match="differ"):
        mc.check_same_items(changed)


def test_identical_models_have_no_difference(tmp_path):
    write_runs(tmp_path, answers=lambda model, items: answers_for("e2b", items),
               e2_answers=lambda model: [True, False] * 3)
    items, _ = mc.load_runs(tmp_path)
    table = mc.compare_models(items, DV_IDS, n_boot=200, n_perm=500)
    assert (table["difference"] == 0).all() and (table["p"] == 1.0).all()


def test_permutation_p_matches_exact_enumeration():
    rng = np.random.default_rng(1)
    sums = pd.DataFrame({
        "num_ref": rng.integers(0, 5, 5).astype(float), "den_ref": np.full(5, 6.0),
        "num_cand": rng.integers(1, 7, 5).astype(float), "den_cand": rng.integers(5, 9, 5).astype(float),
    })
    observed = abs(mc.paired_difference(sums))
    exceed = 0
    for swap in itertools.product([0, 1], repeat=5):
        s = sums.copy()
        flip = np.array(swap, dtype=bool)
        s.loc[flip, ["num_ref", "den_ref", "num_cand", "den_cand"]] = sums.loc[
            flip, ["num_cand", "den_cand", "num_ref", "den_ref"]].to_numpy()
        exceed += abs(mc.paired_difference(s)) >= observed - 1e-12
    assert mc.paired_permutation_p(sums, n_perm=50_000) == pytest.approx(exceed / 32, abs=0.01)


def test_bootstrap_interval_covers_the_estimate():
    sums = pd.DataFrame({"num_ref": [1.0, 2, 0, 3], "den_ref": [4.0, 4, 4, 4],
                         "num_cand": [3.0, 2, 2, 4], "den_cand": [4.0, 4, 4, 4]})
    low, high, undefined = mc.paired_bootstrap(sums, n_boot=2000)
    assert low <= mc.paired_difference(sums) <= high and undefined == 0


def test_reproduction_and_determinism(tmp_path):
    write_runs(tmp_path / "runs")
    items, _ = mc.load_runs(tmp_path / "runs")
    committed = tmp_path / "committed"
    committed.mkdir()
    old = pd.read_csv(mc.cell_dir(tmp_path / "runs", "e2b", "definition", "first_23")
                      / "llm_mapping_evaluation_items.csv")
    old.loc[0, "suggestion"] = "zzz"
    old.to_csv(committed / "llm_mapping_evaluation_items.csv", index=False)
    repro = mc.reproduction(items, {("definition", "first_23"): committed})
    e1 = repro[repro.experiment == "E1"].iloc[0]
    assert (e1.n_committed, e1.only_committed, e1.only_rerun, e1.suggestion_differ, e1.final_checked_differ) == (
        10, 0, 0, 1, 0)
    assert repro[repro.experiment == "E2"].iloc[0].suggestion_differ == 0

    stability = tmp_path / "stability"
    stability.mkdir()
    e4b = items[(items.model == "e4b") & (items.shortlist == "definition") & (items.experiment == "E1")]
    earlier = e4b[["prompt_sha256", "suggestion"]].assign(perturbation="base")
    earlier.loc[earlier.index[0], "suggestion"] = "other"
    earlier.to_csv(stability / "answers_e4b_definition.csv.gz", index=False)
    table = mc.determinism(items, stability).set_index("schema_stability_condition")
    assert table.loc["e4b_definition", ["n_same_prompt", "n_same_answer", "n_different_answer"]].tolist() == [
        20, 19, 1]


def test_analyze_writes_every_output(tmp_path):
    write_runs(tmp_path)
    summary = mc.analyze(tmp_path, n_boot=200, n_perm=500, committed={}, stability_dir=tmp_path / "none",
                         dv_ids=DV_IDS)
    for name in ("comparisons.csv", "rates.csv", "reproduction.csv", "determinism.csv", "summary.json"):
        assert (tmp_path / name).is_file()
    assert summary["n_items"] == {"E1": 20, "E2": 6}
    assert len(summary["runs"]) == 8
    json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
