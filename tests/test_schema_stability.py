"""Tests for the schema-stability benchmark (perturbations, response cache, statistics)."""

from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import yaml

from opendv import llm_utils
from opendv.analysis import evaluate_llm_mapping as ev
from opendv.analysis import schema_stability as ss
from opendv.dv_retrieval import get_retriever
from opendv.llm_plausibility import normalize_name

SCHEMA = {
    "version": "9.9",
    "dvs": [
        {"id": "a", "label": "A", "cluster": "c1", "aliases": ["a1"]},
        {"id": "b", "label": "B", "cluster": "c1", "aliases": ["b1"], "notes": "about b"},
        {"id": "c", "label": "C", "cluster": "c2", "aliases": ["c1x"]},
        {"id": "d", "label": "D", "cluster": "c2", "aliases": ["d1"]},
        {"id": "e", "label": "E", "cluster": "c3", "aliases": ["e1"]},
    ],
}
POOL = {
    "n1": {"id": "n1", "label": "N1", "cluster": "c1", "aliases": ["n1a"]},
    "n2": {"id": "n2", "label": "N2", "cluster": "c9", "aliases": ["n2a"]},
}
REWORD = {"b": {"label": "The B", "notes": "on b"}, "d": {"label": "The D"}}


def ids(schema):
    return [e["id"] for e in schema["dvs"]]


# ── plan ─────────────────────────────────────────────────────────────────────


def test_plan_is_deterministic_and_draws_from_the_right_pools():
    unrelated, pool, rewordable = ["b", "c", "d", "e"], ["n1", "n2", "n3"], ["b", "d"]
    plan = ss.plan_perturbations(unrelated, pool, rewordable, {"edit": 4}, k_levels=(1, 2))
    assert plan == ss.plan_perturbations(unrelated, pool, rewordable, {"edit": 4}, k_levels=(1, 2))
    names = [p.name for p in plan]
    assert len(names) == len(set(names))
    assert names[:2] == ["base", "identity"] and not plan[1].use_cache
    for p in plan:
        if p.family == "add":
            assert set(p.added) <= set(pool) and len(set(p.added)) == p.k
        if p.family == "remove":
            assert set(p.removed) <= set(unrelated) and len(set(p.removed)) == p.k
        if p.family == "reword":
            assert set(p.reworded) <= set(rewordable) and len(set(p.reworded)) == p.k
    counts = Counter((p.family, p.k) for p in plan if p.family in ss.EDIT_FAMILIES)
    assert set(counts.values()) == {4}
    assert plan[-1].family == "real_3_2" and plan[-1].removed == ("touch_offset",)


def test_more_draws_keep_the_earlier_ones():
    small = ss.plan_perturbations(["b", "c", "d", "e"], ["n1", "n2"], ["b", "d"], {"edit": 2}, k_levels=(1,))
    large = ss.plan_perturbations(["b", "c", "d", "e"], ["n1", "n2"], ["b", "d"], {"edit": 5}, k_levels=(1,))
    by_name = {p.name: p for p in large}
    for p in small:
        assert by_name[p.name] == p


def test_draws_vary_across_draws():
    unrelated = [f"x{i}" for i in range(40)]
    plan = ss.plan_perturbations(unrelated, ["n1"], unrelated, {"edit": 5}, k_levels=(1,))
    removed = {p.removed for p in plan if p.family == "remove"}
    assert len(removed) > 1


# ── apply ────────────────────────────────────────────────────────────────────


def test_remove_reword_add_and_permute():
    original = copy.deepcopy(SCHEMA)
    p = ss.Perturbation("x", "remove", removed=("c",))
    assert ids(ss.apply_perturbation(SCHEMA, p, POOL, REWORD)) == ["a", "b", "d", "e"]
    assert original == SCHEMA  # input untouched

    p = ss.Perturbation("x", "reword", reworded=("b", "d"))
    out = {e["id"]: e for e in ss.apply_perturbation(SCHEMA, p, POOL, REWORD)["dvs"]}
    assert (out["b"]["label"], out["b"]["notes"]) == ("The B", "on b")
    assert out["d"]["label"] == "The D" and "notes" not in out["d"]
    assert out["b"]["aliases"] == ["b1"] and out["a"] == SCHEMA["dvs"][0]

    p = ss.Perturbation("x", "add", added=("n1", "n2"))
    assert ids(ss.apply_perturbation(SCHEMA, p, POOL, REWORD)) == ["a", "b", "n1", "c", "d", "e", "n2"]

    p = ss.Perturbation("x", "schema_order", permutation_seed=3)
    permuted = ss.apply_perturbation(SCHEMA, p, POOL, REWORD)
    assert sorted(ids(permuted)) == sorted(ids(SCHEMA))
    assert permuted == ss.apply_perturbation(SCHEMA, p, POOL, REWORD)
    assert {e["id"]: e for e in permuted["dvs"]} == {e["id"]: e for e in SCHEMA["dvs"]}


def test_check_rejects_protected_unknown_and_present_ids():
    protected = {"a"}
    ss.check_perturbation(SCHEMA, ss.Perturbation("ok", "remove", removed=("b",)), protected)
    with pytest.raises(ValueError, match="protected"):
        ss.check_perturbation(SCHEMA, ss.Perturbation("x", "remove", removed=("a",)), protected)
    with pytest.raises(ValueError, match="protected"):
        ss.check_perturbation(SCHEMA, ss.Perturbation("x", "reword", reworded=("a",)), protected)
    with pytest.raises(ValueError, match="missing"):
        ss.check_perturbation(SCHEMA, ss.Perturbation("x", "remove", removed=("zz",)), protected)
    with pytest.raises(ValueError, match="already"):
        ss.check_perturbation(SCHEMA, ss.Perturbation("x", "add", added=("b",)), protected)


def test_protected_dvs_are_gold_labels_and_curated_targets(tmp_path):
    mapping = tmp_path / "m.yaml"
    mapping.write_text(yaml.safe_dump({"aliases": {"colX": "c"}, "composites": [{"id": "d", "items": ["q1"]}],
                                       "meta": {"colY": "meta_age"}}), encoding="utf-8")
    protected = ss.protected_dvs(["a", None], [mapping], {"a", "b", "c", "d", "e"})
    assert protected == {"a", "c", "d"}
    assert ss.unrelated_dvs(SCHEMA, protected) == ["b", "e"]


# ── the committed design inputs ──────────────────────────────────────────────


def _schema_aliases() -> set[str]:
    names: set[str] = set()
    for path in ss.SCHEMAS_DIR.glob("standard_*_mapping.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in data.get("dvs") or []:
            if isinstance(entry, dict):
                names.add(normalize_name(str(entry.get("id"))))
                names.update(normalize_name(str(a)) for a in entry.get("aliases") or [])
        for key, value in (data.get("mappings") or data.get("aliases") or {}).items():
            names.add(normalize_name(str(key)))
            if isinstance(value, str):
                names.add(normalize_name(value))
    return names


def test_added_pool_is_new_and_mirrors_the_schema_clusters():
    pool = ss.load_yaml(ss.DESIGN_DIR / "added_dv_pool.yaml")["dvs"]
    schema = ss.load_yaml(ss.BASE_SCHEMA)
    assert len(pool) == 20 and len({e["id"] for e in pool}) == 20
    taken = _schema_aliases()
    for entry in pool:
        new = {normalize_name(entry["id"]), *(normalize_name(a) for a in entry["aliases"])}
        assert not new & taken, entry["id"]
    clusters = {c["id"] for c in ss.load_yaml(ss.SCHEMAS_DIR / "thematic_clusters.yaml")["clusters"]}
    assert {e["cluster"] for e in pool} <= clusters
    # Largest-remainder allocation of 20 over the schema's cluster sizes.
    sizes = Counter(e["cluster"] for e in schema["dvs"])
    quota = {c: 20 * n / sum(sizes.values()) for c, n in sizes.items()}
    expected = {c: int(q) for c, q in quota.items()}
    for c in sorted(quota, key=lambda c: quota[c] - int(quota[c]), reverse=True)[: 20 - sum(expected.values())]:
        expected[c] += 1
    assert Counter(e["cluster"] for e in pool) == {c: n for c, n in expected.items() if n}


def test_rewordings_change_wording_only():
    rewordings = ss.load_yaml(ss.DESIGN_DIR / "reworded_definitions.yaml")["dvs"]
    schema = {e["id"]: e for e in ss.load_yaml(ss.BASE_SCHEMA)["dvs"]}
    for dv_id, rewording in rewordings.items():
        assert set(rewording) <= {"label", "notes"}
        assert rewording["label"] != schema[dv_id]["label"]
        assert ("notes" in rewording) == bool(schema[dv_id].get("notes")), dv_id


def test_a_rewritten_base_schema_gives_the_same_prompts_and_rankings(tmp_path):
    path = tmp_path / "base.yaml"
    ss.write_schema(ss.load_yaml(ss.BASE_SCHEMA), path)
    candidates = ["mental_demand", "sart_demand", "tia_trust_in_automation", "sts_ad_trust"]
    for column in ("MentalDemand", "Trust_3"):
        assert llm_utils.build_mapping_prompt(column, candidates, "ctx", path) == llm_utils.build_mapping_prompt(
            column, candidates, "ctx", None
        )
        assert get_retriever(str(path)).rank(column, k=24, exclude_alias=column) == get_retriever().rank(
            column, k=24, exclude_alias=column
        )


def test_a_perturbed_schema_reaches_retrieval_and_prompt(tmp_path):
    pool = {e["id"]: e for e in ss.load_yaml(ss.DESIGN_DIR / "added_dv_pool.yaml")["dvs"]}
    p = ss.Perturbation("add", "add", added=("text_entry_rate",))
    path = tmp_path / "add.yaml"
    ss.write_schema(ss.apply_perturbation(ss.load_yaml(ss.BASE_SCHEMA), p, pool, {}), path)
    assert get_retriever(str(path)).rank("typing_speed_wpm", k=1)[0][0] == "text_entry_rate"
    prompt = llm_utils.build_mapping_prompt("wpm", ["text_entry_rate"], "", path)
    assert "text_entry_rate | label=Text Entry Rate" in prompt


# ── benchmark options and the response cache ─────────────────────────────────


def test_candidate_orders():
    shortlist = ["mental_demand", "Effort", "arousal", "TLX_SCORE"]
    assert ev.order_candidates(shortlist, "x", "rank") == shortlist
    assert ev.order_candidates(shortlist, "x", "alphabetical") == ["arousal", "Effort", "mental_demand", "TLX_SCORE"]
    shuffled = ev.order_candidates(shortlist, "x", "shuffle", seed=1)
    assert sorted(shuffled) == sorted(shortlist)
    assert shuffled == ev.order_candidates(shortlist, "x", "shuffle", seed=1)
    orders = {tuple(ev.order_candidates(shortlist, "x", "shuffle", seed=s)) for s in range(10)}
    assert len(orders) > 1
    with pytest.raises(ValueError):
        ev.order_candidates(shortlist, "x", "random")


def test_response_cache_reuses_identical_prompts_only(monkeypatch):
    calls = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        return kwargs["canonical_candidates"][0]

    monkeypatch.setattr(llm_utils, "deduce_standard_name_with_local_llm", fake_llm)
    monkeypatch.setattr(llm_utils, "select_local_model_candidates", lambda models: list(models or ["m"]))
    mapping = {"mental_demand": "mental_demand", "mentaldemand": "mental_demand", "effort": "effort",
               "frustration": "frustration"}
    cache: dict = {}
    first = ev._ask_llm("MentalDemand", mapping, "ctx", ["m"], "definition", response_cache=cache)
    again = ev._ask_llm("MentalDemand", mapping, "ctx", ["m"], "definition", response_cache=cache)
    assert first == again and len(calls) == 1 and len(cache) == 1
    ev._ask_llm("MentalDemand", mapping, "other context", ["m"], "definition", response_cache=cache)
    ev._ask_llm("MentalDemand", mapping, "ctx", ["m"], "definition",
                options=ev.AskOptions(num_samples=5), response_cache=cache)
    assert len(calls) == 3 and len(cache) == 3
    ev._ask_llm("MentalDemand", mapping, "ctx", ["m"], "definition", response_cache=None)
    assert len(calls) == 4


def test_cache_keys_depend_on_decoding():
    greedy = ev.response_cache_key(["m"], "p", ev.AskOptions())
    assert greedy == ev.response_cache_key(["m"], "p", ev.AskOptions(temperature=0.3, sample_seed=9))
    assert greedy != ev.response_cache_key(["m"], "p", ev.AskOptions(num_samples=5))
    assert ev.response_cache_key(["m"], "p", ev.AskOptions(num_samples=5, sample_seed=1)) != ev.response_cache_key(
        ["m"], "p", ev.AskOptions(num_samples=5, sample_seed=2)
    )
    assert greedy != ev.response_cache_key(["n"], "p", ev.AskOptions())


def test_majority_vote_breaks_ties_by_shortlist_order():
    candidates = ["a", "b", "c"]
    assert llm_utils.majority_vote(["b", "b", "a"], candidates) == "b"
    assert llm_utils.majority_vote(["c", "a", None], candidates) == "a"  # three-way tie: first listed
    assert llm_utils.majority_vote(["b", "a", None, None, "c"], candidates) is None  # "no answer" is a vote
    assert llm_utils.majority_vote([None, None, "c"], candidates) is None
    assert llm_utils.majority_vote(["c", None], candidates) == "c"
    assert llm_utils.majority_vote([], candidates) is None


def test_sampled_answers_are_voted(monkeypatch):
    seen = {}

    def pipeline(prompt, **kwargs):
        seen.update(kwargs)
        return [{"generated_text": t} for t in ("trust_rating", "effort", "trust_rating")]

    monkeypatch.setattr(llm_utils, "select_local_model_candidates", lambda preferred_models=None: ["fake"])
    monkeypatch.setattr(llm_utils, "_get_text_generation_pipeline", lambda name: pipeline)
    answer = llm_utils.deduce_standard_name_with_local_llm(
        "x", ["effort", "trust_rating"], repository_context="", num_samples=3, temperature=0.7, sample_seed=4
    )
    assert answer == "trust_rating"
    assert seen["do_sample"] is True and seen["num_return_sequences"] == 3 and seen["temperature"] == 0.7


# ── flips and statistics ─────────────────────────────────────────────────────


def _answers(rows):
    base = {"half": "h", "dataset_id": "t.csv", "label_kind": "dv", "llm_error": None, "shortlist": "a|b",
            "prompt_sha256": "p0"}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_flip_table_flags_forced_and_gold_removing_runs():
    perturbations = {
        "remove": ss.Perturbation("remove", "remove", k=1, removed=("b",)),
        "add": ss.Perturbation("add", "add", k=1, added=("n",)),
        "real_3_2": ss.Perturbation("real_3_2", "real_3_2", k=1, removed=("touch_offset",)),
    }
    answers = _answers([
        {"perturbation": "base", "source_id": "s1", "alias": "x", "gold": "a", "final_checked": "a"},
        {"perturbation": "base", "source_id": "s1", "alias": "y", "gold": "a", "final_checked": "b"},
        {"perturbation": "base", "source_id": "s2", "alias": "z", "gold": "touch_offset",
         "final_checked": "touch_offset"},
        {"perturbation": "base", "source_id": "s2", "alias": "w", "gold": None, "label_kind": "curated_non_dv",
         "final_checked": None, "prompt_sha256": None},
        {"perturbation": "remove", "source_id": "s1", "alias": "x", "gold": "a", "final_checked": "c",
         "prompt_sha256": "p1", "shortlist": "a|c"},
        {"perturbation": "remove", "source_id": "s1", "alias": "y", "gold": "a", "final_checked": "a",
         "prompt_sha256": "p1", "shortlist": "a|c"},
        {"perturbation": "add", "source_id": "s1", "alias": "x", "gold": "a", "final_checked": "n",
         "prompt_sha256": "p2", "shortlist": "b|a"},
        {"perturbation": "add", "source_id": "s2", "alias": "w", "gold": None, "label_kind": "curated_non_dv",
         "final_checked": "a", "prompt_sha256": "p3"},
        {"perturbation": "real_3_2", "source_id": "s2", "alias": "z", "gold": "touch_offset", "final_checked": "a"},
    ])
    flips = ss.flip_table(answers, perturbations).set_index(["perturbation", "alias"])
    assert flips.loc[("remove", "x"), "flip"] and flips.loc[("remove", "x"), "valid"]
    assert flips.loc[("remove", "x"), "set_changed"]
    assert flips.loc[("remove", "y"), "forced"] and not flips.loc[("remove", "y"), "valid"]
    assert flips.loc[("add", "x"), "to_touched_dv"] and not flips.loc[("add", "x"), "set_changed"]
    assert flips.loc[("add", "x"), "prompt_changed"]
    assert flips.loc[("add", "w"), "flip"] and flips.loc[("add", "w"), "base_answer"] is None
    assert not flips.loc[("real_3_2", "z"), "valid"]
    summary = ss.summarize_set(ss.select_set(flips.reset_index(), "add_k1"), n_boot=200)
    assert summary["flip_rate"] == 1.0 and summary["none_to_dv_rate"] == 1.0
    assert summary["flips_to_touched_dv"] == 1


def test_cluster_bootstrap_and_sign_flip_test():
    values = pd.Series([0.2] * 30)
    clusters = pd.Series(np.repeat(np.arange(10), 3))
    assert ss.cluster_bootstrap(values, clusters, n_boot=500) == pytest.approx((0.2, 0.2, 0.2))
    assert ss.sign_flip_test(pd.Series([0.0] * 30), clusters, n_perm=500) == 1.0
    p = ss.sign_flip_test(pd.Series([0.1] * 30), clusters, n_perm=5000)
    assert p < 0.01  # all 10 clusters agree: 2 / 2**10 exactly, up to Monte Carlo error
    mixed = pd.Series(np.tile([0.1, -0.1, 0.0], 10))
    assert ss.sign_flip_test(mixed, clusters, n_perm=500) == 1.0


def test_holm():
    assert ss.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ss.holm([0.5]) == [0.5]


def test_leave_alias_out_skips_the_refit_only_when_no_definition_lists_the_alias(monkeypatch):
    from opendv.dv_retrieval import DefinitionRetriever, load_dv_definitions

    retriever = DefinitionRetriever(load_dv_definitions())
    fits = []
    original = retriever._fit
    monkeypatch.setattr(retriever, "_fit", lambda alias: fits.append(alias) or original(alias))
    shortcut = retriever.rank("Pz_alpha Max", k=24, exclude_alias="Pz_alpha Max")
    assert fits == []
    forced = DefinitionRetriever(load_dv_definitions())
    forced._aliases = {normalize_name("Pz_alpha Max")}  # pretend a definition lists it: refit
    assert forced.rank("Pz_alpha Max", k=24, exclude_alias="Pz_alpha Max") == shortcut
    retriever.rank("time_on_task", k=24, exclude_alias="time_on_task")  # an alias of task_completion_time
    assert fits == ["time_on_task"]


def test_analyze_end_to_end(tmp_path):
    design = {"perturbations": [
        {"name": "base", "family": "base"},
        {"name": "add_k1_d1", "family": "add", "k": 1, "draw": 1, "added": ["n"]},
        {"name": "remove_k1_d1", "family": "remove", "k": 1, "draw": 1, "removed": ["b"]},
    ]}
    rng = np.random.default_rng(0)
    for condition, flip_p in (("e2b_definition", 0.5), ("e2b_dense_h", 0.0)):
        rows = []
        for study in range(6):
            for column in range(5):
                gold = "a" if column < 3 else None
                kind = "dv" if gold else "curated_non_dv"
                base_answer = "a" if column % 2 == 0 else None
                item = {"condition": condition, "half": "h", "source_id": f"s{study}", "dataset_id": "t.csv",
                        "alias": f"c{column}", "label_kind": kind, "gold": gold, "shortlist": "a|b",
                        "prompt_sha256": "p", "llm_error": None, "final_unchecked": base_answer}
                rows.append({**item, "perturbation": "base", "final_checked": base_answer})
                for name in ("add_k1_d1", "remove_k1_d1"):
                    flipped = rng.random() < flip_p
                    answer = ("c" if base_answer is None else None) if flipped else base_answer
                    rows.append({**item, "perturbation": name, "final_checked": answer,
                                 "prompt_sha256": "q" if flipped else "p"})
        (tmp_path / "work" / condition).mkdir(parents=True)
        pd.DataFrame(rows).to_csv(tmp_path / "work" / condition / "all.csv", index=False)
    result = ss.analyze(design, tmp_path / "work", tmp_path / "out", n_boot=300)
    assert result["conditions"] == ["e2b_definition", "e2b_dense_h"]
    rates = pd.read_csv(tmp_path / "out" / "flip_rates.csv").set_index(["condition", "answer", "set"])
    stable = rates.loc[("e2b_dense_h", "checked", "schema_edits")]
    assert stable["flip_rate"] == 0 and stable["flip_rate_ci_high"] == 0 and stable["n_items"] == 30
    assert stable["accuracy_change"] == 0
    unstable = rates.loc[("e2b_definition", "checked", "schema_edits")]
    assert 0.2 < unstable["flip_rate"] < 0.8
    assert unstable["flips_with_unchanged_prompt"] == 0
    comparison = pd.read_csv(tmp_path / "out" / "comparisons.csv").set_index(["answer", "condition"])
    row = comparison.loc[("checked", "e2b_dense_h")]
    assert row["difference"] < 0 and row["n_studies"] == 6 and 0 < row["p_sign_flip"] <= 1
    assert (tmp_path / "out" / "answers_e2b_definition.csv.gz").exists()
    reference = pd.read_csv(tmp_path / "out" / "reference_accuracy.csv")
    assert set(reference["dv_n"]) == {18}


# ── exploratory ──────────────────────────────────────────────────────────────


def test_tie_flags_groups():
    scores = {"a": 0.5, "b": 0.2, "c": 0.2, "d": 0.0, "e": 0.0}
    tied = ss.tie_flags(scores, ["a", "b"])  # c (left out) ties with b
    assert tied["tie_at_cut"] and tied["n_zero"] == 0 and tied["tie_group"] == "tie at the cut"
    assert ss.tie_flags(scores, ["a", "b", "c"])["tie_group"] == "no tie"
    padded = ss.tie_flags(scores, ["a", "b", "c", "d"])
    assert padded["n_zero"] == 1 and padded["tie_group"] == "zero-score candidates"
    assert ss.tie_flags(scores, list(scores))["tie_at_cut"] is False  # nothing left out
    assert ss.tie_flags(scores, ["zz"])["tie_group"] is None


def test_answer_position():
    assert ss.answer_position("b", "a|b|c") == 2
    assert ss.answer_position(None, "a|b") == 0
    assert ss.answer_position(np.nan, "a|b") == 0
    assert ss.answer_position("x", "a|b") == -1


def test_explore_end_to_end(tmp_path, monkeypatch):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(yaml.safe_dump(SCHEMA), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    design = {"perturbations": [
        {"name": "base", "family": "base"},
        {"name": "remove_k1_d1", "family": "remove", "k": 1, "draw": 1, "removed": ["e"]},
        {"name": "candidate_order_d1", "family": "candidate_order", "draw": 1},
    ]}
    (out / "design.json").write_text(json.dumps(design), encoding="utf-8")
    for condition, right in (("e2b_definition", False), ("e2b_definition_alpha", True)):
        work = tmp_path / "work" / condition
        work.mkdir(parents=True)
        rows = []
        for study in range(4):
            for column, alias in enumerate(("a1", "b1", "d1")):
                gold = "a" if column < 2 else None
                answer = "a" if right or column == 1 else None
                rows.append({"condition": condition, "half": "h", "source_id": f"s{study}", "dataset_id": "t.csv",
                             "alias": alias, "label_kind": "dv" if gold else "curated_non_dv", "gold": gold,
                             "shortlist": "a|b", "prompt_sha256": "p", "llm_error": None, "suggestion": answer,
                             "final_unchecked": answer, "final_checked": answer})
        frame = pd.DataFrame(rows)
        frame.assign(perturbation="base").to_csv(work / "base.csv", index=False)
        frame.assign(perturbation="remove_k1_d1").to_csv(work / "remove_k1_d1.csv", index=False)
        frame.assign(perturbation="candidate_order_d1", shortlist="b|a").to_csv(
            work / "candidate_order_d1.csv", index=False)
    result = ss.explore(tmp_path / "work", out, n_boot=200, schema_path=schema_path)
    assert result["conditions"] == ["e2b_definition", "e2b_definition_alpha"]
    ties = pd.read_csv(out / "shortlist_ties.csv")
    assert len(ties) == 12 and ties["tie_group"].notna().all()
    by_tie = pd.read_csv(out / "flip_rate_by_tie.csv")
    assert set(by_tie["condition"]) == {"e2b_definition", "e2b_definition_alpha"}
    assert by_tie["flip_rate"].fillna(0).eq(0).all()  # nothing flipped
    positions = pd.read_csv(out / "answer_positions.csv").set_index(["condition", "runs", "position"])
    assert positions.loc[("e2b_definition", "reference", 1), "n"] == 4  # b1 answered "a", listed first
    assert positions.loc[("e2b_definition", "candidate_order", 2), "n"] == 4  # "a" listed second
    accuracy = pd.read_csv(out / "accuracy_comparisons.csv").set_index(["answer", "metric"])
    row = accuracy.loc[("checked", "accuracy")]
    assert row["n_items"] == 8 and row["n_gained"] == 4 and row["n_lost"] == 0
    assert row["difference"] == pytest.approx(0.5) and row["largest_study_net"] == 1
    assert accuracy.loc[("checked", "non_dv_mapped"), "n_gained"] == 4
    by_answer = pd.read_csv(out / "stability_by_answer.csv").set_index(["condition", "answer", "group"])
    reference = by_answer.loc[("e2b_definition", "checked")]
    assert reference.loc["DV column, correct DV", "n_items"] == 4  # b1: answer "a" = gold
    assert reference.loc["DV column, no DV", "n_items"] == 4  # a1: no answer
    assert reference.loc["non-DV column, no DV", "n_items"] == 4
    assert reference.loc["DV answer, never flipped", "precision"] == 1.0


def test_answer_status():
    assert ss.answer_status("dv", "a", "a") == "DV column, correct DV"
    assert ss.answer_status("dv", "a", "b") == "DV column, other DV"
    assert ss.answer_status("dv", "a", None) == "DV column, no DV"
    assert ss.answer_status("curated_non_dv", None, "b") == "non-DV column, mapped to a DV"
    assert ss.answer_status("curated_non_dv", None, None) == "non-DV column, no DV"
