"""Write the batch manifest of the whole OpenDV corpus: every study the repository maps.

The corpus is the union of three study sets, each with curated mappings:

- the 39 development studies (``sources_manifest_no4tu.yaml``, mappings in ``schemas/``);
- the 19 studies of the first held-out test (``analyses/heldout/``);
- the 30 studies of the second held-out test (``analyses/heldout_2/``).

The held-out folders are pre-registered records and are only read here. Each entry is copied
unchanged, with one added key, ``study_set``, so the analysis can tell the sets apart, except for
the studies whose held-out mapping has corrections for the corpus (``corrections.py``): their
``mapping_path`` points to the corrected copy in ``mappings/`` and ``mapping_corrected_from`` names
the frozen file.

    python -m analyses.opendv_corpus.build_manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from analyses.opendv_corpus import corrections

REPO = Path(__file__).resolve().parents[2]
SETS = (
    ("development", REPO / "sources_manifest_no4tu.yaml"),
    ("heldout_1", REPO / "analyses" / "heldout" / "sources_manifest.yaml"),
    ("heldout_2", REPO / "analyses" / "heldout_2" / "sources_manifest.yaml"),
)
OUT = REPO / "analyses" / "opendv_corpus" / "sources_manifest.yaml"

HEADER = """\
# Batch manifest of the whole OpenDV corpus (88 studies), written by
#   python -m analyses.opendv_corpus.build_manifest
# Do not edit by hand; edit the set's own manifest and rerun the script.
# Sets: development (sources_manifest_no4tu.yaml), heldout_1 (analyses/heldout/sources_manifest.yaml),
# heldout_2 (analyses/heldout_2/sources_manifest.yaml). Data: the 11 remote development sources are
# fetched by the batch (use --cache-dir); the others are read from data/raw/development,
# data/raw/heldout and data/raw/heldout_2 (see each set's fetch command in docs/reproducibility.md).
"""


def build() -> list[dict]:
    sources: list[dict] = []
    seen: dict[str, str] = {}
    for study_set, path in SETS:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in manifest["sources"]:
            sid = entry["source_id"]
            if sid in seen:
                raise SystemExit(f"{sid} is in both {seen[sid]} and {study_set}")
            seen[sid] = study_set
            entry = {"study_set": study_set, **entry}
            if sid in corrections.CORRECTIONS:
                spec = corrections.CORRECTIONS[sid]
                if entry.get("mapping_path") != spec["frozen"]:
                    raise SystemExit(f"{sid}: manifest maps {entry.get('mapping_path')}, corrections expect {spec['frozen']}")
                entry["mapping_path"] = corrections.corrected_path(sid).relative_to(REPO).as_posix()
                entry["mapping_corrected_from"] = spec["frozen"]
                if spec.get("override_standard_aliases"):
                    entry["override_standard_aliases"] = True
            sources.append(entry)
    return sources


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sources = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"sources": sources}, sort_keys=False, allow_unicode=True, width=120)
    args.output.write_text(HEADER + body, encoding="utf-8")
    counts = {name: sum(s["study_set"] == name for s in sources) for name, _ in SETS}
    print(f"wrote {args.output} ({len(sources)} sources: {counts})")


if __name__ == "__main__":
    main()
