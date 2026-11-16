"""Corrected copies of held-out mappings, for the corpus analysis only.

The held-out mappings are frozen with their pre-registered tests and are not revised
(``docs/preregistration_heldout_relations*.md``). A scale audit of the corpus (2026-11-15) found
NASA-TLX columns that the frozen mappings send to schema DVs on another scale than the DV's: the
mappings' own notes give the scale (1-10, 1-9, 1-7, 0-1) but no ``scale_conversions``, and the
analysis cannot see a narrower scale that lies inside the DV's range. Pooled as they are, they pull
the corpus's NASA-TLX means down.

This script writes ``mappings/<source_id>_mapping.yaml``: the frozen file, unchanged, plus the
corrections below, each with its reason. ``build_manifest.py`` points the corpus manifest at these
copies. The held-out results are not affected.

    python -m analyses.opendv_corpus.corrections
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "mappings"
TLX_SUBSCALES = ("mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration")

CORRECTIONS: dict[str, dict] = {
    "ehmi_infrastructure_2024": {
        "frozen": "analyses/heldout/mappings/ehmi_infrastructure_2024_mapping.yaml",
        "scale_conversions": [
            {"columns": ["Mental demand (1 to 10)", "Physical demand (1 to 10)", "Temporal demand (1 to 10)",
                         "Performance (1 to 10)", "Effort (1 to 10)", "Frustration (1 to 10)"],
             "from": [1, 10], "to": [0, 20],
             "reason": "NASA-TLX subscales on 1-10 (mapping notes, column headers)"},
            {"columns": ["Total workload (1 to 10)"], "from": [1, 10], "to": [0, 100],
             "reason": "NASA-TLX total on 1-10 (column header); TLX_SCORE is 0-100"},
        ],
    },
    "human_swarm_pfm_2025": {
        "frozen": "analyses/heldout_2/mappings/human_swarm_pfm_2025_mapping.yaml",
        "scale_conversions": [
            {"columns": ["md", "pd", "td", "pf", "ef", "fr"], "from": [1, 10], "to": [0, 20],
             "reason": "NASA-TLX items on 1-10 (Readme.txt, mapping notes); pf is reversed first on 1-10"},
        ],
    },
    "mr_airplane_layouts_2021": {
        "frozen": "analyses/heldout_2/mappings/mr_airplane_layouts_2021_mapping.yaml",
        "scale_conversions": [
            {"columns": ["mentaldemand", "physicaldemand", "effort"], "from": [1, 9], "to": [0, 20],
             "reason": "NASA-TLX subscales on 1-9 (paper, mapping notes)"},
        ],
        # left unlabelled in the frozen mapping, but the shared alias tlxscore -> TLX_SCORE matched it
        "relock": {"tlxscore": "meta_derived_partial_tlx"},
        "relock_reason": "a partial TLX of three subscales on 1-9 (mapping notes), not the NASA-TLX total",
        "override_standard_aliases": True,
    },
    "music_intensity_driving_2021": {
        "frozen": "analyses/heldout/mappings/music_intensity_driving_2021_mapping.yaml",
        "relock": {f"{drive}NASAScore": "meta_derived_tlx_total_unstated"
                   for drive in ("ConTalk", "ConUrb", "LyrLou", "LyrSof", "NonLyrLou", "NonLyrSof")},
        "relock_reason": ("total of 40-1720 whose formula the data do not state (mapping notes); the analysis "
                          "computes TLX_SCORE from the six 0-20 subscales instead"),
    },
    "shared_autonomy_2026": {
        "frozen": "analyses/heldout/mappings/shared_autonomy_2026_mapping.yaml",
        "scale_conversions": [
            {"columns": [f"NASA_TLX_RAW_C_{i}" for i in range(1, 7)], "from": [0, 1], "to": [0, 20],
             "reason": "raw NASA-TLX subscales rescaled to 0-1 (mapping notes)"},
        ],
        "relock": {"NASA_TLX_SCORE": "meta_derived_tlx_weighted_total"},
        "relock_reason": ("the weighted total on 0-1 (mapping notes); the analysis computes the raw TLX_SCORE "
                          "from the converted subscales, as for every other study"),
    },
    "vr_classroom_noise_2025": {
        "frozen": "analyses/heldout_2/mappings/vr_classroom_noise_2025_mapping.yaml",
        "scale_conversions": [
            {"columns": ["q1", "q2", "q4", "q5"], "from": [1, 7], "to": [0, 20],
             "reason": "TLX-type items on 1-7 (paper, mapping notes); q4 is reversed first on 1-7"},
        ],
    },
}


def corrected_text(source_id: str) -> str:
    spec = CORRECTIONS[source_id]
    frozen_path = REPO / spec["frozen"]
    text = frozen_path.read_text(encoding="utf-8")
    original = yaml.safe_load(text) or {}
    if "scale_conversions" in original and spec.get("scale_conversions"):
        raise SystemExit(f"{source_id}: the frozen mapping has scale_conversions; merge them by hand")
    lines = text.splitlines()
    notes = []
    for alias, target in (spec.get("relock") or {}).items():
        reason = spec["relock_reason"]
        for i, line in enumerate(lines):
            if line.split(":", 1)[0].strip() == alias and not line.lstrip().startswith("#"):
                lines[i] = f"{alias}: {target}   # corpus correction (was {original.get(alias)}): {reason}"
                break
        else:
            lines.append(f"{alias}: {target}   # corpus correction (was unmapped): {reason}")
        notes.append(f"{alias} -> {target}")
    body = "\n".join(lines).rstrip() + "\n"
    if spec.get("scale_conversions"):
        block = yaml.safe_dump({"scale_conversions": spec["scale_conversions"]}, sort_keys=False,
                               allow_unicode=True, width=110)
        body += "\n# --- corpus correction: declared recording scales ---------------------------------\n" + block
        notes += [f"{c['columns'][0]}{' ...' if len(c['columns']) > 1 else ''}: {c['from']} -> {c['to']}"
                  for c in spec["scale_conversions"]]
    header = (
        f"# CORPUS COPY of {spec['frozen']} (frozen with its pre-registered test; not revised).\n"
        "# Written by python -m analyses.opendv_corpus.corrections; the corrections found by the corpus scale\n"
        "# audit of 2026-11-15 are marked 'corpus correction'. Used only by analyses/opendv_corpus/.\n"
        f"# Corrections: {'; '.join(notes)}\n"
        + ("# The corpus manifest sets override_standard_aliases: true for this study.\n"
           if spec.get("override_standard_aliases") else "")
        + "#\n"
    )
    return header + body


def corrected_path(source_id: str) -> Path:
    return OUT_DIR / f"{source_id}_mapping.yaml"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_id in CORRECTIONS:
        path = corrected_path(source_id)
        path.write_text(corrected_text(source_id), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
