"""Check a study mapping file against the schema before running the batch.

    opendv check-mapping schemas/my_study_mapping.yaml [--override-standard-aliases] [--json]

A study mapping maps the raw column names of one study to schema DVs (``schemas/
standard_dv_mapping.yaml``), to the canonical names of the metadata, sensor and detection
schemas, or to ``meta_*`` names for everything that is not a DV (design factors, demographics,
identifiers, measures without a schema DV). The check reports:

- **errors**: targets that are not strings;
- **warnings**:
    - targets that are none of the above: the batch keeps such a column as study-specific and never
      pools it, so it is either a typo of a DV id (the closest ids are listed) or should be ``meta_*``
      or a new schema DV;
    - mappings the batch will not apply as written:
        - the alias is also an alias of the shared DV schema with another target. The shared
          schema wins unless the manifest entry sets ``override_standard_aliases: true``;
        - the alias is also an alias of the metadata, sensor or detection schema. Those can match
          first, so a DV target can be lost (osf_cwd6h's ``ArduinoData1`` in schema 3.4);
        - the alias is on the batch's never-map list (identifiers and condition columns), so a
          DV target is ignored;
        - the same alias appears twice, differing only in case, with different targets;
- **coverage**: the DVs mapped, by cluster, and which instrument scores of
  ``schemas/instruments.yaml`` the analysis can compute because all of their items are mapped.

Exit code 1 when there are errors.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from opendv._paths import SCHEMAS_DIR

OTHER_SCHEMAS = ("standard_metadata_mapping.yaml", "standard_sensor_mapping.yaml",
                 "standard_detection_mapping.yaml")


def _walk_ids(node: Any, out: dict[str, set[str]]) -> None:
    """Collect id -> aliases from every ``{id, aliases}`` entry of a schema file."""
    if isinstance(node, dict):
        if "id" in node and isinstance(node.get("aliases", []), list):
            out.setdefault(str(node["id"]), set()).update(str(a) for a in node.get("aliases") or [])
        for value in node.values():
            _walk_ids(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_ids(value, out)


def load_vocabulary(schema_dir: Path = SCHEMAS_DIR) -> dict[str, Any]:
    dv_doc = yaml.safe_load((schema_dir / "standard_dv_mapping.yaml").read_text(encoding="utf-8"))
    dvs = {str(d["id"]): d for d in dv_doc["dvs"]}
    dv_aliases = {}
    for dv_id, d in dvs.items():
        for alias in [dv_id, *(d.get("aliases") or [])]:
            dv_aliases.setdefault(str(alias).lower(), dv_id)
    other: dict[str, str] = {}
    other_ids: set[str] = set()
    for name in OTHER_SCHEMAS:
        path = schema_dir / name
        if not path.exists():
            continue
        ids: dict[str, set[str]] = {}
        _walk_ids(yaml.safe_load(path.read_text(encoding="utf-8")), ids)
        for cid, aliases in ids.items():
            other_ids.add(cid)
            for alias in {cid, *aliases}:
                other.setdefault(alias.lower(), f"{cid} ({name})")
    instruments = yaml.safe_load((schema_dir / "instruments.yaml").read_text(encoding="utf-8"))
    instruments = instruments.get("instruments", {})
    item_ids = {str(item["id"]) for inst in instruments.values() for item in inst.get("items") or []
                if isinstance(item, dict) and item.get("id")}
    return {"version": dv_doc.get("version"), "dvs": dvs, "dv_aliases": dv_aliases,
            "other_aliases": other, "other_ids": other_ids, "item_ids": item_ids - set(dvs),
            "instruments": instruments}


def read_mapping(path: Path) -> list[tuple[str, Any]]:
    """The (alias, target) pairs of a study mapping, in file order; option keys
    (``scales``, ``reverse_items``, ``exclude_rows``, ...) are not column names."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: a study mapping must be a YAML mapping of column name -> target")
    if isinstance(doc.get("dvs"), list):  # a custom schema in the DV-schema format
        pairs = []
        for entry in doc["dvs"]:
            if isinstance(entry, dict) and entry.get("id"):
                pairs.extend((str(a), entry["id"]) for a in [entry["id"], *(entry.get("aliases") or [])])
        return pairs
    from opendv.convert_dv import RESERVED_MAPPING_KEYS

    return [(str(k), v) for k, v in doc.items() if k not in RESERVED_MAPPING_KEYS]


def _never_map(alias: str) -> bool:
    from opendv.run_batch_standardization import _is_never_map_column

    return _is_never_map_column(alias)


def check_mapping(path: Path, *, override_standard_aliases: bool = False,
                  vocabulary: dict[str, Any] | None = None) -> dict[str, Any]:
    vocab = vocabulary or load_vocabulary()
    dvs, dv_aliases, other = vocab["dvs"], vocab["dv_aliases"], vocab["other_aliases"]
    findings: list[dict[str, str]] = []

    def add(level: str, alias: str, target: Any, message: str) -> None:
        findings.append({"level": level, "alias": alias, "target": str(target), "message": message})

    pairs = read_mapping(Path(path))
    seen: dict[str, tuple[str, Any]] = {}
    targets: dict[str, list[str]] = defaultdict(list)
    for alias, target in pairs:
        if not isinstance(target, str) or not target.strip():
            add("error", alias, target, "the target must be a non-empty string")
            continue
        low = alias.lower()
        if low in seen and seen[low][1] != target:
            add("warning", alias, target,
                f"also mapped as '{seen[low][0]}' -> {seen[low][1]}; aliases match case-insensitively")
        seen.setdefault(low, (alias, target))
        is_dv = target in dvs
        is_meta = target.startswith("meta_")
        is_other = target in vocab["other_ids"]
        is_item = target in vocab["item_ids"]
        if not (is_dv or is_meta or is_other or is_item):
            close = difflib.get_close_matches(target, list(dvs), n=3, cutoff=0.6)
            hint = f"; closest DV ids: {', '.join(close)}" if close else ""
            add("warning", alias, target, "not in the schema: the batch keeps it as a study-specific column, "
                "which is never pooled with other studies (a typo of a DV id, or a measure for meta_* or a "
                "new DV?)" + hint)
            targets[target].append(alias)
            continue
        targets[target].append(alias)
        shared = dv_aliases.get(low)
        if shared and shared != target and not override_standard_aliases:
            add("warning", alias, target,
                f"the shared DV schema maps this name to {shared} and wins; set "
                "override_standard_aliases: true in the manifest entry to apply this mapping")
        if low in other and not is_other and not (shared and shared == target):
            add("warning", alias, target,
                f"also an alias of {other[low]}, which can match first; check the column in the batch's meta_view.csv")
        if _never_map(alias) and (is_dv or is_other):
            add("warning", alias, target, "on the batch's never-map list (identifiers, conditions); this mapping is ignored")

    mapped_dvs = {t for t in targets if t in dvs}
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for dv in sorted(mapped_dvs):
        by_cluster[str(dvs[dv].get("cluster"))].append(dv)
    scores = []
    for inst_id, inst in vocab["instruments"].items():
        for score_name, score in (inst.get("scores") or {}).items():
            items = [str(i) for i in score.get("items") or []]
            have = [i for i in items if i in mapped_dvs or i in targets]
            if have:
                scores.append({"instrument": inst_id, "score": score_name, "dv": score.get("dv"),
                               "items_mapped": len(have), "items": len(items),
                               "computed": len(have) == len(items),
                               "missing": [i for i in items if i not in mapped_dvs and i not in targets]})
    return {
        "mapping": str(path), "schema_version": vocab["version"],
        "aliases": len(pairs),
        "to_dvs": sum(len(v) for t, v in targets.items() if t in dvs),
        "to_meta": sum(len(v) for t, v in targets.items() if t.startswith("meta_")),
        "to_instrument_items": sum(len(v) for t, v in targets.items() if t in vocab["item_ids"]),
        "to_study_specific": sum(len(v) for t, v in targets.items()
                                 if t not in dvs and t not in vocab["item_ids"] and t not in vocab["other_ids"]
                                 and not t.startswith("meta_")),
        "to_other_schemas": sum(len(v) for t, v in targets.items() if t in vocab["other_ids"]),
        "dvs": {k: v for k, v in sorted(by_cluster.items())},
        "instrument_scores": scores,
        "errors": [f for f in findings if f["level"] == "error"],
        "warnings": [f for f in findings if f["level"] == "warning"],
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [f"Mapping: {report['mapping']} (schema {report['schema_version']})",
             f"  {report['aliases']} column names: {report['to_dvs']} to DVs, {report['to_instrument_items']} to "
             f"instrument items, {report['to_meta']} to meta_*, {report['to_other_schemas']} to metadata/sensor/"
             f"detection names, {report['to_study_specific']} study-specific"]
    for cluster, dvs in report["dvs"].items():
        lines.append(f"  {cluster}: {', '.join(dvs)}")
    for s in report["instrument_scores"]:
        state = "computed" if s["computed"] else f"not computed, missing {', '.join(s['missing'])}"
        lines.append(f"  score {s['dv']} ({s['instrument']}): {s['items_mapped']}/{s['items']} items, {state}")
    for level in ("errors", "warnings"):
        for f in report[level]:
            lines.append(f"  [{level[:-1].upper()}] {f['alias']} -> {f['target']}: {f['message']}")
    if not report["errors"] and not report["warnings"]:
        lines.append("  [OK] no errors or warnings")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opendv check-mapping", description=__doc__.splitlines()[0])
    parser.add_argument("mappings", nargs="+", type=Path, help="study mapping YAML file(s)")
    parser.add_argument("--override-standard-aliases", action="store_true",
                        help="the manifest entry sets override_standard_aliases: true")
    parser.add_argument("--json", action="store_true", help="print the reports as JSON")
    args = parser.parse_args(argv)
    vocab = load_vocabulary()
    reports = [check_mapping(p, override_standard_aliases=args.override_standard_aliases, vocabulary=vocab)
               for p in args.mappings]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print("\n\n".join(format_report(r) for r in reports))
    return 1 if any(r["errors"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
