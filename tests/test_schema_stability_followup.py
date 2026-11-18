"""Tests for the schema-stability follow-up: no zero-score padding and order votes.

Design: docs/schema_stability_followup_design.md.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from opendv import llm_utils
from opendv.analysis import evaluate_llm_mapping as ev
from opendv.analysis import schema_stability as ss
from opendv.dv_retrieval import definition_shortlist, get_retriever, without_zero_scores

MAPPING = {"mental_demand": "mental_demand", "mentaldemand": "mental_demand", "effort": "effort",
           "frustration": "frustration", "temporal_demand": "temporal_demand", "performance": "performance"}


@pytest.fixture
def fake_llm(monkeypatch):
    """The LLM answers with the candidate listed first; every call is recorded."""
    calls = []

    def answer(**kwargs):
        calls.append(list(kwargs["canonical_candidates"]))
        return kwargs["canonical_candidates"][0]

    monkeypatch.setattr(llm_utils, "deduce_standard_name_with_local_llm", answer)
    monkeypatch.setattr(llm_utils, "select_local_model_candidates", lambda models: list(models or ["m"]))
    return calls


# ── no zero-score padding ────────────────────────────────────────────────────


def test_without_zero_scores_keeps_order_and_positive_scores():
    ranked = [("a", 0.4), ("b", 0.1), ("c", 0.0), ("d", 0.0)]
    assert without_zero_scores(ranked) == [("a", 0.4), ("b", 0.1)]
    assert without_zero_scores([("c", 0.0)]) == []


def test_definition_shortlist_without_padding():
    column = "Qxz"  # shares no 3-gram with most definitions
    scores = dict(get_retriever().rank(column, k=100_000))
    padded = definition_shortlist(column, max_candidates=8)
    kept = definition_shortlist(column, max_candidates=8, drop_zero_scores=True)
    assert len(padded) == 8
    assert kept == [dv for dv in padded if scores[dv] > 0]
    assert len(kept) < 8 and all(scores[dv] > 0 for dv in kept)


def test_ask_llm_drops_zero_score_candidates(fake_llm):
    column = "MentalDemand_Qxz"
    padded, _, _ = ev._ask_llm(column, MAPPING, "", ["m"], "definition")
    kept, _, _ = ev._ask_llm(column, MAPPING, "", ["m"], "definition", options=ev.AskOptions(drop_zero_scores=True))
    scores = dict(get_retriever().rank(column, k=100_000, exclude_alias=column))
    assert kept == [dv for dv in padded if scores[dv] > 0]


def test_a_shortlist_with_no_positive_score_is_not_asked(fake_llm, monkeypatch):
    monkeypatch.setattr(ev, "get_retriever", lambda *_a: type("R", (), {"rank": lambda *_a, **_k: [("x", 0.0)]})())
    shortlist, top, answer = ev._ask_llm("MentalDemand", MAPPING, "", ["m"], "definition",
                                         options=ev.AskOptions(drop_zero_scores=True))
    assert shortlist == [] and answer is None and fake_llm == []


# ── order votes ──────────────────────────────────────────────────────────────


def _old_shuffle(candidates, alias, seed):
    """order_candidates' shuffle as committed with the reference (e2df354)."""
    digest = hashlib.sha256(f"{seed}\x1f{alias}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return [candidates[i] for i in rng.permutation(len(candidates))]


def test_shuffles_are_the_references():
    shortlist = [f"dv{i}" for i in range(8)]
    for seed in range(1, 7):
        for alias in ("TLX1", "Godspeed Anxious", "x"):
            expected = _old_shuffle(shortlist, alias, seed)
            assert llm_utils.shuffle_candidates(shortlist, alias, seed) == expected
            assert ev.order_candidates(shortlist, alias, "shuffle", seed) == expected


def test_candidate_orders():
    shortlist = ["a", "b", "c", "d"]
    orders = llm_utils.candidate_orders(shortlist, "col", (1, 2))
    assert orders[0] == shortlist and len(orders) == 3
    assert orders[1] == _old_shuffle(shortlist, "col", 1) and orders[2] == _old_shuffle(shortlist, "col", 2)
    assert llm_utils.candidate_orders(shortlist, "col", ()) == [shortlist]


def test_vote_over_orders():
    candidates = ["a", "b", "c"]
    assert llm_utils.vote_over_orders(["b", "a", "b"], candidates) == "b"
    assert llm_utils.vote_over_orders(["c", "b", None], candidates) == "b"  # all differ: best retrieval rank
    assert llm_utils.vote_over_orders([None, None, "a"], candidates) is None
    assert llm_utils.vote_over_orders(["b", "b", "b"], candidates, "unanimous") == "b"
    assert llm_utils.vote_over_orders(["b", "b", "a"], candidates, "unanimous") is None
    assert llm_utils.vote_over_orders([None, None, None], candidates, "unanimous") is None
    with pytest.raises(ValueError):
        llm_utils.vote_over_orders(["a"], candidates, "plurality")


def test_ask_options_validation_and_args():
    with pytest.raises(ValueError):
        ev.AskOptions(vote_seeds=(1, 2), num_samples=5)
    with pytest.raises(ValueError):
        ev.AskOptions(vote_rule="plurality")
    args = type("A", (), {"vote_seeds": [1, 2], "vote_rule": "unanimous", "drop_zero_scores": True})()
    options = ev.AskOptions.from_args(args)
    assert options.vote_seeds == (1, 2) and options.vote_rule == "unanimous" and options.drop_zero_scores
    assert ev.AskOptions.from_args(object()) == ev.AskOptions()


def test_votes_cache_each_order_under_the_single_order_key(fake_llm):
    options = ev.AskOptions(vote_seeds=(1, 2))
    cache: dict = {}
    details: dict = {}
    shortlist, _, answer = ev._ask_llm("MentalDemand", MAPPING, "ctx", ["m"], "definition", options=options,
                                       response_cache=cache, details=details)
    orders = llm_utils.candidate_orders(shortlist, "MentalDemand", (1, 2))
    assert fake_llm == orders and len(cache) == len({tuple(o) for o in orders})
    assert details["order_suggestions"] == [order[0] for order in orders]
    assert answer == llm_utils.majority_vote([order[0] for order in orders], shortlist)
    # A single-order run in rank order, or shuffled with seed 1, finds its prompt in the vote's cache.
    before = len(fake_llm)
    ev._ask_llm("MentalDemand", MAPPING, "ctx", ["m"], "definition", response_cache=cache)
    ev._ask_llm("MentalDemand", MAPPING, "ctx", ["m"], "definition",
                options=ev.AskOptions(candidate_order="shuffle", order_seed=1), response_cache=cache)
    assert len(fake_llm) == before
    # The unanimous rule reuses the same three answers.
    ev._ask_llm("MentalDemand", MAPPING, "ctx", ["m"], "definition",
                options=ev.AskOptions(vote_seeds=(1, 2), vote_rule="unanimous"), response_cache=cache)
    assert len(fake_llm) == before


def test_prompt_fingerprint_covers_every_order():
    shortlist = ["mental_demand", "effort", "frustration"]
    single = ev.prompt_fingerprint("MentalDemand", shortlist, "ctx", ev.AskOptions())
    prompt = llm_utils.build_mapping_prompt("MentalDemand", shortlist, "ctx")
    assert single == hashlib.sha256(prompt.encode()).hexdigest()
    vote = ev.prompt_fingerprint("MentalDemand", shortlist, "ctx", ev.AskOptions(vote_seeds=(1, 2)))
    other = ev.prompt_fingerprint("MentalDemand", shortlist, "ctx", ev.AskOptions(vote_seeds=(3, 4)))
    assert len({single, vote, other}) == 3


def test_pipeline_function_votes_over_orders(monkeypatch):
    seen = []

    def pipeline(prompt, **kwargs):
        listed = [line.split(" | ")[0].strip("- ") for line in prompt.splitlines() if " | label=" in line]
        seen.append(listed)
        return [{"generated_text": listed[-1]}]  # the last-listed candidate

    monkeypatch.setattr(llm_utils, "select_local_model_candidates", lambda preferred_models=None: ["fake"])
    monkeypatch.setattr(llm_utils, "_get_text_generation_pipeline", lambda name: pipeline)
    candidates = ["mental_demand", "effort", "frustration"]
    answer = llm_utils.deduce_standard_name_with_local_llm("x", candidates, repository_context="", vote_seeds=(1, 2))
    assert len(seen) == 3 and seen[0] == candidates
    assert answer == llm_utils.majority_vote([listed[-1] for listed in seen], candidates)
    unanimous = llm_utils.deduce_standard_name_with_local_llm(
        "x", candidates, repository_context="", vote_seeds=(1, 2), vote_rule="unanimous")
    assert unanimous == (seen[0][-1] if len({tuple(s[-1:]) for s in seen[-3:]}) == 1 else None)
    with pytest.raises(ValueError):
        llm_utils.deduce_standard_name_with_local_llm("x", candidates, repository_context="", vote_seeds=(1,),
                                                      num_samples=3)


# ── conditions ───────────────────────────────────────────────────────────────


def test_followup_conditions_and_their_perturbations():
    nopad, vote, vote_u = (ss.FOLLOWUP_CONDITIONS[n] for n in ss.FOLLOWUP_CONDITIONS)
    assert nopad.drop_zero_scores and not nopad.vote_seeds
    assert vote.vote_seeds == vote_u.vote_seeds == (1, 2) and vote_u.vote_rule == "unanimous"
    assert set(ss.ALL_CONDITIONS) == set(ss.CONDITIONS) | set(ss.FOLLOWUP_CONDITIONS)
    family = {f: ss.Perturbation(f, f) for f in ("identity", "candidate_order", "sample_seed", "add", "schema_order")}
    assert all(nopad.applies(family[f]) for f in ("identity", "candidate_order", "add", "schema_order"))
    assert not nopad.applies(family["sample_seed"])
    assert not vote.applies(family["identity"]) and not vote.applies(family["candidate_order"])
    assert vote.applies(family["sample_seed"]) and vote.applies(family["add"])
    assert ss.CONDITIONS["e2b_definition"].applies(family["identity"])  # the original conditions are unchanged
    assert vote.order_seeds(ss.Perturbation("base", "base")) == (1, 2)
    assert vote.order_seeds(ss.Perturbation("s1", "sample_seed", sample_seed_offset=1)) == (3, 4)
    assert vote.order_seeds(ss.Perturbation("s2", "sample_seed", sample_seed_offset=2)) == (5, 6)
    assert nopad.order_seeds(ss.Perturbation("s1", "sample_seed", sample_seed_offset=1)) == ()


def test_freeze_mappings(tmp_path):
    manifest = tmp_path / "manifest23.yaml"
    manifest.write_text(
        "sources:\n"
        "- source_id: a  # first\n"
        "  mapping_path: schemas/a_mapping.yaml\n"
        "- source_id: b\n"
        "  mapping_path: 'schemas/b_mapping.yaml'\n"
        "- source_id: c\n",
        encoding="utf-8",
    )
    shown = []

    def git_show(commit, path):
        shown.append((commit, path))
        return f"aliases: {{x: {path}}}\n".encode()

    record = ss.freeze_mappings([manifest], "abc", tmp_path / "out", git_show=git_show)
    assert shown == [("abc", "schemas/a_mapping.yaml"), ("abc", "schemas/b_mapping.yaml")]
    frozen = (tmp_path / "out" / "manifest23.yaml").read_text(encoding="utf-8")
    assert "- source_id: a  # first" in frozen and "schemas/a_mapping.yaml" not in frozen.replace("mappings/a_", "")
    sources = yaml.safe_load(frozen)["sources"]
    assert sources[0]["mapping_path"] == (tmp_path / "out" / "mappings" / "a_mapping.yaml").as_posix()
    assert (tmp_path / "out" / "mappings" / "b_mapping.yaml").read_bytes() == b"aliases: {x: schemas/b_mapping.yaml}\n"
    assert "mapping_path" not in sources[2]
    assert record["commit"] == "abc" and set(record["mapping_files_sha256"]) == {
        "schemas/a_mapping.yaml", "schemas/b_mapping.yaml"}
    assert json.loads((tmp_path / "out" / "frozen_mappings.json").read_text(encoding="utf-8")) == record


def test_extra_caches_are_read_but_not_written(tmp_path, monkeypatch):
    extra = tmp_path / "reference_cache.jsonl"
    extra.write_text(json.dumps({"key": "old", "answer": "a"}) + "\n", encoding="utf-8")
    seen = {}

    def run_perturbation(condition, perturbation, halves, items, work_dir, cache):
        seen[perturbation.name] = dict(cache)
        cache["new"] = "b"
        return pd.DataFrame({"llm_error": [None], "prompt_sha256": ["p"]})

    monkeypatch.setattr(ss, "base_dv_ids", lambda: set())
    monkeypatch.setattr(ss, "benchmark_items", lambda half, dv_ids: pd.DataFrame())
    monkeypatch.setattr(ss, "run_perturbation", run_perturbation)
    design = {"perturbations": [{"name": "base", "family": "base"}]}
    condition = ss.FOLLOWUP_CONDITIONS["e2b_definition_nopad"]
    ss.run_condition(condition, design, [], tmp_path / "work", extra_caches=[extra])
    assert seen["base"] == {"old": "a"}
    own = ss.load_response_cache(tmp_path / "work" / condition.name / "response_cache.jsonl")
    assert own == {"new": "b"}
    record = json.loads((tmp_path / "work" / condition.name / "extra_caches.json").read_text(encoding="utf-8"))
    assert record[0]["sha256"] == hashlib.sha256(extra.read_bytes()).hexdigest()


# ── statistics ───────────────────────────────────────────────────────────────


def test_ratio_comparison():
    clusters = pd.Series(np.repeat(np.arange(12), 5))
    ones = pd.Series(np.ones(60))
    hits = pd.Series(np.tile([1.0, 1.0, 0.0, 0.0, 0.0], 12))
    same = ss.ratio_comparison(hits, ones, hits, ones, clusters, n_boot=300, n_perm=2000)
    assert same["difference"] == 0 and same["p"] == 1.0 and same["value"] == pytest.approx(0.4)
    better = pd.Series(np.tile([1.0, 1.0, 1.0, 1.0, 0.0], 12))
    result = ss.ratio_comparison(better, ones, hits, ones, clusters, n_boot=300, n_perm=5000)
    assert result["difference"] == pytest.approx(0.4)
    assert result["difference_ci_low"] == pytest.approx(0.4) and result["p"] < 0.005  # 2 / 2**12 + Monte Carlo
    # Denominators of 1: the swap test is the sign-flip test on study totals.
    mixed = pd.Series(np.r_[np.tile([1.0, 1.0, 1.0, 0.0, 0.0], 6), np.tile([1.0, 1.0, 0.0, 0.0, 0.0], 6)])
    ratio_p = ss.ratio_comparison(mixed, ones, hits, ones, clusters, n_perm=20_000)["p"]
    flip_p = ss.sign_flip_test(mixed - hits, clusters, n_perm=20_000)
    assert ratio_p == pytest.approx(flip_p, abs=0.02)
    # Precision: different numbers accepted.
    accepted_a = pd.Series(np.tile([1.0, 1.0, 0.0, 0.0, 0.0], 12))
    accepted_b = pd.Series(np.tile([1.0, 1.0, 1.0, 1.0, 0.0], 12))
    correct = pd.Series(np.tile([1.0, 0.0, 0.0, 0.0, 0.0], 12))
    precision = ss.ratio_comparison(correct, accepted_a, correct, accepted_b, clusters, n_boot=300, n_perm=2000)
    assert precision["value"] == pytest.approx(0.5) and precision["value_reference"] == pytest.approx(0.25)


# ── the follow-up analysis ───────────────────────────────────────────────────


def _rows(condition, perturbation, study, column, **overrides):
    gold = "a" if column < 3 else None
    row = {"condition": condition, "perturbation": perturbation, "half": "h", "source_id": f"s{study}",
           "dataset_id": "t.csv", "alias": f"c{column}", "label_kind": "dv" if gold else "curated_non_dv",
           "gold": gold, "shortlist": "a|b|c", "prompt_sha256": f"p{column}", "llm_error": None,
           "suggestion": "a", "final_unchecked": "a", "final_checked": "a" if column % 2 == 0 else None}
    return {**row, **overrides}


def _synthetic(tmp_path):
    design = {"perturbations": [
        {"name": "base", "family": "base"},
        {"name": "add_k1_d1", "family": "add", "k": 1, "draw": 1, "added": ["n"]},
        {"name": "remove_k1_d1", "family": "remove", "k": 1, "draw": 1, "removed": ["z"]},
        {"name": "sample_seed_d1", "family": "sample_seed", "draw": 1, "sample_seed_offset": 1},
    ]}
    reference, nopad, vote, vote_u = [], [], [], []
    for study in range(6):
        for column in range(5):
            for perturbation in ("base", "add_k1_d1", "remove_k1_d1"):
                flipped = perturbation == "add_k1_d1" and column == 1
                answer = {"final_checked": "b", "suggestion": "b", "final_unchecked": "b"} if flipped else {}
                ref = _rows("e2b_definition", perturbation, study, column, **answer)
                reference.append(ref)
                if column == 4:  # padded: nopad's shortlist is shorter and its answer is no DV
                    nopad.append({**ref, "condition": "e2b_definition_nopad", "shortlist": "a",
                                  "prompt_sha256": "q", "suggestion": None, "final_unchecked": None,
                                  "final_checked": None})
                else:
                    nopad.append({**ref, "condition": "e2b_definition_nopad"})
                orders = f"{ref['suggestion'] or ''}|{ref['suggestion'] or ''}|c" if column != 2 else "b|a|c"
                votes = {"order_suggestions": orders, "prompt_sha256": f"v{column}"}
                vote.append({**ref, **votes, "condition": "e2b_definition_vote3"})
                unanimous = {"final_checked": None, "final_unchecked": None, "suggestion": None}
                vote_u.append({**ref, **votes, **unanimous, "condition": "e2b_definition_vote3u"})
            for frame, name in ((vote, "e2b_definition_vote3"), (vote_u, "e2b_definition_vote3u")):
                frame.append({**frame[-1], "perturbation": "sample_seed_d1", "condition": name})
    (tmp_path / "ref").mkdir()
    reference_path = tmp_path / "ref" / "answers_e2b_definition.csv.gz"
    pd.DataFrame(reference).to_csv(reference_path, index=False)
    for name, frame in (("e2b_definition_nopad", nopad), ("e2b_definition_vote3", vote),
                        ("e2b_definition_vote3u", vote_u)):
        (tmp_path / "work" / name).mkdir(parents=True)
        pd.DataFrame(frame).to_csv(tmp_path / "work" / name / "all.csv", index=False)
    return design, reference_path


def test_analyze_followup_end_to_end(tmp_path):
    design, reference_path = _synthetic(tmp_path)
    out = tmp_path / "out"
    result = ss.analyze_followup(design, tmp_path / "work", out, reference_path, n_boot=200, n_perm=2000)
    assert result["conditions"] == ["e2b_definition", "e2b_definition_nopad", "e2b_definition_vote3",
                                    "e2b_definition_vote3u"]
    integrity = pd.read_csv(out / "integrity_checks.csv").set_index(["condition", "check"])
    nopad = integrity.loc["e2b_definition_nopad"]
    assert nopad.loc["shortlist as the reference's", "n_mismatched"] == 18  # 6 studies x 3 runs of column c4
    assert (nopad.drop("shortlist as the reference's")["n_mismatched"] == 0).all()
    assert integrity.loc[("e2b_definition_vote3", "rank-order answer as the reference's"), "n_mismatched"] == 18
    assert integrity.loc[("e2b_definition_vote3 / e2b_definition_vote3u", "same answers under the three orders"),
                         "n_mismatched"] == 0
    comparisons = pd.read_csv(out / "outcome_comparisons.csv").set_index(["answer", "condition", "outcome"])
    assert len(comparisons.loc["checked"]) == 12 and comparisons.loc["checked", "p_holm"].notna().all()
    assert comparisons.loc["unchecked", "p_holm"].isna().all()
    row = comparisons.loc[("checked", "e2b_definition_nopad", "non_dv_mapped")]
    assert row["k"] == 0 and row["k_reference"] == 6 and row["n_lost"] == 6 and row["difference"] < 0
    flips = comparisons.loc[("checked", "e2b_definition_vote3u", "schema_edit_flip_rate")]
    assert flips["value"] == 0 and flips["value_reference"] > 0
    precision = comparisons.loc[("checked", "e2b_definition_nopad", "precision")]
    assert precision["n_accepted"] == 12 and precision["n_accepted_reference"] == 18
    padded = pd.read_csv(out / "padded_columns.csv")
    assert len(padded) == 6 and set(padded["status_reference"]) == {"non-DV column, mapped to a DV"}
    assert set(padded["status_nopad"]) == {"non-DV column, no DV"}
    agreement = pd.read_csv(out / "vote_agreement.csv").set_index("agreement")
    assert agreement.loc["two agree", "n_asked"] == 24 and agreement.loc["all three differ", "n_asked"] == 6
    rates = pd.read_csv(out / "flip_rates.csv").set_index(["condition", "answer", "set"])
    assert ("e2b_definition_vote3", "checked", "sample_seed") in rates.index
    outcomes = pd.read_csv(out / "condition_outcomes.csv").set_index(["condition", "answer"])
    assert outcomes.loc[("e2b_definition", "checked"), "dv_accuracy_k"] == 12  # c0, c2 of 6 studies
    assert outcomes.loc[("e2b_definition_vote3u", "checked"), "accepted_dv_mappings"] == 0
    assert (out / "answers_e2b_definition_vote3.csv.gz").exists()
    assert not (out / "answers_e2b_definition.csv.gz").exists()  # the reference is read, not rewritten
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["padded_columns"]["n"] == 6 and summary["padded_columns"]["checked_answer_changed"] == 6
