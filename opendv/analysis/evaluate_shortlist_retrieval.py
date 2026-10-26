#!/usr/bin/env python3
"""Compare candidate-retrieval strategies for LLM DV mapping (recall@k).

The LLM can only pick a DV that the shortlist offers, and the evaluation in
``evaluate_llm_mapping.py`` shows that the fuzzy *alias-string* shortlist misses
the correct DV for most real dataset columns. This script replays the same
held-out items (``llm_mapping_evaluation_items.csv``) through three retrievers
and reports recall@k with Wilson 95% CIs:

* ``fuzzy``      — alias-string similarity (rapidfuzz), the original shortlist;
* ``definition`` — ``scripts/dv_retrieval.py``: TF-IDF over DV definitions
  (mean of name and full-definition similarity);
* ``hybrid``     — reciprocal-rank fusion of the two (Cormack et al., 2009);
* ``dense:<model>`` and ``hybrid:<model>`` (with ``--dense-model``) — cosine
  similarity of sentence embeddings of the same definitions
  (``scripts.dv_retrieval.DenseRetriever``), alone and fused with the
  definition ranking.

Every item is leave-alias-out: the held-out alias is removed from the fuzzy
mapping and from every DV definition before ranking. No LLM is called; a
dense model is loaded from the Hugging Face cache (downloaded if missing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


from opendv.dv_retrieval import (  # noqa: E402
    N_DENSE_EXTRA,
    DenseRetriever,
    get_retriever,
    load_dv_definitions,
    mix_rankings,
)
from opendv.llm_coordinator import _select_llm_candidate_shortlist  # noqa: E402
from opendv.llm_plausibility import normalize_name  # noqa: E402
from opendv.run_batch_standardization import SchemaFamilies  # noqa: E402
from opendv._paths import SCHEMAS_DIR

KS = (1, 3, 5, 8, 10)
RRF_K = 60


def _fuzzy_ranking(alias: str, mapping: dict[str, str]) -> list[str]:
    target = normalize_name(alias)
    reduced = {k: v for k, v in mapping.items() if normalize_name(k) != target}
    ranked, _ = _select_llm_candidate_shortlist(reduced, alias, max_candidates=max(KS))
    return ranked


def _rrf(*rankings: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _wilson(k: int, n: int) -> dict:
    ci = binomtest(k, n).proportion_ci(method="wilson") if n else None
    return {"k": k, "n": n, "rate": k / n if n else None,
            "ci95": [float(ci.low), float(ci.high)] if ci else [None, None]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", default="analyses/output_llm_evaluation/llm_mapping_evaluation_items.csv")
    parser.add_argument("--output", default="analyses/output_llm_evaluation/shortlist_retrieval_summary.json")
    parser.add_argument("--dense-model", action="append", default=[],
                        help="sentence-transformers model id to add as a dense retriever (repeatable)")
    parser.add_argument("--device", default="cpu", help="Device for dense models")
    args = parser.parse_args()

    retriever = get_retriever()
    definitions = load_dv_definitions()
    dense = {
        model: DenseRetriever.from_model(definitions, model, device=args.device)
        for model in args.dense_model
    }
    dv_ids = {dv for dv, _ in retriever.rank("x", k=10_000)}
    mapping = SchemaFamilies.load(SCHEMAS_DIR / "standard_dv_mapping.yaml").standard_mapping_for(
        "questionnaire"
    )
    items = pd.read_csv(args.items)
    items = items[(items["label_kind"] == "dv") & items["gold"].isin(dv_ids)]
    results: dict[str, dict] = {}
    for experiment, group in items.groupby("experiment", sort=True):
        names = ["fuzzy", "definition", "hybrid"]
        names += [f"{kind}:{model}" for model in dense for kind in ("dense", "hybrid", "mixed")]
        hits = {name: {k: 0 for k in KS} for name in names}
        for row in group.itertuples(index=False):
            fuzzy = _fuzzy_ranking(row.alias, mapping)
            definition = [dv for dv, _ in retriever.rank(row.alias, k=max(KS) * 2, exclude_alias=row.alias)]
            rankings = {"fuzzy": fuzzy, "definition": definition, "hybrid": _rrf(fuzzy, definition)}
            for model, dense_retriever in dense.items():
                ranking = [dv for dv, _ in dense_retriever.rank(row.alias, k=max(KS) * 2, exclude_alias=row.alias)]
                rankings[f"dense:{model}"] = ranking
                rankings[f"hybrid:{model}"] = _rrf(definition, ranking)
                # recall@k of the mixed list: k - N_DENSE_EXTRA TF-IDF + dense-only extras
                mixed_at = {k: mix_rankings(definition, ranking, k, min(N_DENSE_EXTRA, k - 1)) for k in KS}
                rankings[f"mixed:{model}"] = mixed_at
            for name, ranking in rankings.items():
                for k in KS:
                    top = ranking[k] if isinstance(ranking, dict) else ranking[:k]
                    hits[name][k] += int(row.gold in top)
        results[experiment] = {
            name: {f"recall@{k}": _wilson(count, len(group)) for k, count in by_k.items()}
            for name, by_k in hits.items()
        }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for experiment, by_name in results.items():
        print(experiment)
        for name, by_k in by_name.items():
            print(f"  {name:40s}", "  ".join(f"{key}={val['rate']:.2f}" for key, val in by_k.items()))


if __name__ == "__main__":
    main()
