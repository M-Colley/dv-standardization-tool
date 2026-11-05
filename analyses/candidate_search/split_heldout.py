"""Split the verified datasets of the 2026-11-05 search into a held-out and a development half.

The split uses only metadata (titles, notes, search cues); no data file has been opened.
Strata, from that metadata:

    automotive, safety/risk/trust measure, no NASA-TLX
    automotive, NASA-TLX
    automotive, neither
    other domains, NASA-TLX
    other domains, no NASA-TLX

Within each stratum the datasets are sorted by DOI and permuted with a fixed seed, chosen
before the split was run (the date of the split). The first half goes to the held-out set;
in strata of odd size the extra dataset goes alternately to the held-out and the development
set, starting with the held-out set, in the stratum order above.

    python -m analyses.candidate_search.split_heldout
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20261105
STRATA = ("automotive_safety", "automotive_tlx", "automotive_other", "other_tlx", "other_no_tlx")
TLX = r"nasa[-_ ]?tlx|\btlx\b|workload"
SAFETY = r"perceived[-_ ]safety|perceived risk|safety|trust"


def stratum(row: pd.Series) -> str:
    text = f"{row['note']} {row['title']} {row['cues']}"
    tlx = bool(pd.Series([text]).str.contains(TLX, case=False).iloc[0])
    safety = bool(pd.Series([text]).str.contains(SAFETY, case=False).iloc[0])
    if row["domain"] == "automotive":
        if tlx:
            return "automotive_tlx"
        return "automotive_safety" if safety else "automotive_other"
    return "other_tlx" if tlx else "other_no_tlx"


def split(decisions: pd.DataFrame, hits: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    included = decisions[decisions["decision"] == "include"].copy()
    included["cues"] = included["doi"].map(hits.set_index("doi")["cues"]).fillna("")
    included["stratum"] = included.apply(stratum, axis=1)
    rng = np.random.default_rng(seed)
    parts, extra_to_heldout = [], True
    for name in STRATA:
        group = included[included["stratum"] == name].sort_values("doi").reset_index(drop=True)
        group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        n_heldout = len(group) // 2
        if len(group) % 2:
            n_heldout += int(extra_to_heldout)
            extra_to_heldout = not extra_to_heldout
        group["set"] = ["heldout"] * n_heldout + ["development"] * (len(group) - n_heldout)
        parts.append(group)
    columns = ["doi", "set", "stratum", "domain", "title", "paper_doi", "venue", "paper_year", "publisher",
               "licence", "url", "note"]
    return pd.concat(parts, ignore_index=True)[columns]


def main() -> None:
    decisions = pd.read_csv(HERE / "screening_decisions.csv")
    hits = pd.read_csv(HERE / "hits.csv")
    result = split(decisions, hits)
    result.to_csv(HERE / "heldout_split.csv", index=False)
    print(pd.crosstab(result["stratum"], result["set"], margins=True).to_string())


if __name__ == "__main__":
    main()
