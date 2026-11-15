"""Export the DV schema as a table and a reference page.

    python -m opendv.schema_export            # writes the three files below
    python -m opendv.schema_export --check    # exit 1 if any of them is out of date

- ``schemas/export/opendv_dvs.csv`` and ``.json``: one record per DV (id, label, cluster,
  measurement, range, direction, instruments, aliases, notes), for spreadsheets, R or other tools.
- ``docs/schema_reference.md``: the same, grouped by cluster, for people choosing a DV.

The YAML schema (``schemas/standard_dv_mapping.yaml``) stays the source; these files are generated.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import yaml

from opendv._paths import REPO_ROOT, SCHEMAS_DIR

FIELDS = ("id", "label", "cluster", "category", "primary_unit", "scale_type", "direction",
          "range_min", "range_max", "instruments", "subitem_of", "aliases", "notes")


def records(schema_dir: Path = SCHEMAS_DIR) -> tuple[str, list[dict]]:
    doc = yaml.safe_load((schema_dir / "standard_dv_mapping.yaml").read_text(encoding="utf-8"))
    rows = []
    for dv in doc["dvs"]:
        m = dv.get("measurement") or {}
        rng = m.get("canonical_range") or [None, None]
        rows.append({
            "id": dv["id"], "label": dv.get("label", ""), "cluster": dv.get("cluster", ""),
            "category": m.get("category", ""), "primary_unit": m.get("primary_unit", ""),
            "scale_type": m.get("scale_type", ""), "direction": m.get("direction", ""),
            "range_min": rng[0], "range_max": rng[1],
            "instruments": "; ".join(str(i) for i in dv.get("instruments") or []),
            "subitem_of": dv.get("subitem_of", ""),
            "aliases": "; ".join(str(a) for a in dv.get("aliases") or []),
            "notes": " ".join(str(dv.get("notes", "")).split()),
        })
    return str(doc.get("version")), rows


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if row[k] is None else row[k] for k in FIELDS})
    return buf.getvalue()


def to_json(version: str, rows: list[dict]) -> str:
    return json.dumps({"schema": "OpenDV-HCI standard DV schema", "version": version, "dvs": rows},
                      indent=1, ensure_ascii=False) + "\n"


def _clusters(schema_dir: Path) -> dict[str, str]:
    path = schema_dir / "thematic_clusters.yaml"
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    items = doc.get("clusters", doc)
    if isinstance(items, dict):
        items = [{"id": k, **(v if isinstance(v, dict) else {})} for k, v in items.items()]
    for c in items or []:
        if isinstance(c, dict) and c.get("id"):
            out[str(c["id"])] = str(c.get("label") or c.get("name") or c["id"])
    return out


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def to_markdown(version: str, rows: list[dict], schema_dir: Path = SCHEMAS_DIR) -> str:
    labels = _clusters(schema_dir)
    order: list[str] = []
    for row in rows:
        if row["cluster"] not in order:
            order.append(row["cluster"])
    lines = [
        "# DV schema reference",
        "",
        f"Generated from `schemas/standard_dv_mapping.yaml` (version {version}) by "
        "`python -m opendv.schema_export`; do not edit by hand. "
        f"{len(rows)} DVs in {len(order)} clusters. The same records are in "
        "`schemas/export/opendv_dvs.csv` and `.json`.",
        "",
        "Map a column to the **id**. *Range* is the canonical range the batch checks values "
        "against (blank: unbounded); *direction* says which end is better. Items of a scale "
        "name their scale under *item of*. How to write a study mapping: "
        "[Mapping a study for OpenDV](sigchi_mapping_guide.md).",
        "",
    ]
    for cluster in order:
        members = [r for r in rows if r["cluster"] == cluster]
        lines += [f"## {labels.get(cluster, cluster)} (`{cluster}`, {len(members)})", "",
                  "| id | label | range | unit | direction | instruments | item of |",
                  "|---|---|---|---|---|---|---|"]
        for r in members:
            rng = "" if r["range_min"] is None else f"{r['range_min']} to {r['range_max']}"
            lines.append("| " + " | ".join(_cell(v) for v in (
                f"`{r['id']}`", r["label"], rng, r["primary_unit"], r["direction"],
                r["instruments"], r["subitem_of"])) + " |")
        lines.append("")
    return "\n".join(lines)


def outputs(schema_dir: Path = SCHEMAS_DIR, repo_root: Path = REPO_ROOT) -> dict[Path, str]:
    version, rows = records(schema_dir)
    return {
        schema_dir / "export" / "opendv_dvs.csv": to_csv(rows),
        schema_dir / "export" / "opendv_dvs.json": to_json(version, rows),
        repo_root / "docs" / "schema_reference.md": to_markdown(version, rows, schema_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a generated file is out of date")
    args = parser.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    for path in stale:
        print(f"out of date: {path} (run python -m opendv.schema_export)", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
