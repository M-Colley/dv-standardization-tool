"""The generated schema exports stay in sync with the schema, and schema 3.4's additions are safe."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import yaml

from opendv import schema_export
from opendv._paths import SCHEMAS_DIR

V34_NEW = {
    "umux_score", "umux_lite_score", "seq_rating", "csi_score", "misc_rating", "slater_usoh_steed_presence",
    "embodiment_score", "words_per_minute", "kspc", "msd_error_rate", "total_error_rate", "fitts_throughput",
    "ues_focused_attention", "ues_perceived_usability", "ues_aesthetic_appeal", "ues_reward",
    "flow_fks_total", "flow_fks_fluency", "flow_fks_absorption",
}


def _schema():
    return yaml.safe_load((SCHEMAS_DIR / "standard_dv_mapping.yaml").read_text(encoding="utf-8"))


def test_generated_exports_are_up_to_date():
    assert schema_export.main(["--check"]) == 0, "run python -m opendv.schema_export"


def test_export_has_one_record_per_dv():
    schema = _schema()
    version, rows = schema_export.records()
    assert version == str(schema["version"])
    assert [r["id"] for r in rows] == [d["id"] for d in schema["dvs"]]
    parsed = list(csv.DictReader(io.StringIO(schema_export.to_csv(rows))))
    assert len(parsed) == len(schema["dvs"])
    assert json.loads(schema_export.to_json(version, rows))["version"] == version


def test_v34_dvs_are_present_with_ranges_and_instrument_aliases():
    dvs = {d["id"]: d for d in _schema()["dvs"]}
    assert V34_NEW <= set(dvs)
    assert sum(1 for d in dvs if d.startswith("pxi_")) == 11
    assert sum(1 for d in dvs if d.startswith("geq_")) == 7
    assert sum(1 for d in dvs if d.startswith("imi_")) == 6
    for dv_id in V34_NEW:
        for alias in dvs[dv_id].get("aliases") or []:
            assert len(str(alias)) >= 3, (dv_id, alias)


def test_aliases_are_unique_and_subitems_point_to_dvs():
    dvs = _schema()["dvs"]
    ids = {d["id"] for d in dvs}
    owner: dict[str, str] = {}
    for d in dvs:
        for alias in d.get("aliases") or []:
            key = str(alias).lower()
            assert owner.setdefault(key, d["id"]) == d["id"], f"{alias}: {owner[key]} and {d['id']}"
        if d.get("subitem_of"):
            assert d["subitem_of"] in ids, d["id"]


def test_reference_page_lists_every_dv():
    text = (Path(schema_export.REPO_ROOT) / "docs" / "schema_reference.md").read_text(encoding="utf-8")
    for d in _schema()["dvs"]:
        assert f"`{d['id']}`" in text
