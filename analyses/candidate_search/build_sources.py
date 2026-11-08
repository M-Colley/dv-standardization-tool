"""Pin the files of the split datasets before anything is downloaded.

For every dataset in ``heldout_split.csv`` the repository's file listing gives each
file's download URL, size and published checksum (MD5 or SHA-256). Files are chosen by
fixed rules (``classify``):

- tables (CSV, TSV, Excel, ODS, SPSS, Stata, Parquet, Feather) up to 50 MB;
- documentation (README, codebook, data dictionary, questionnaire) up to 20 MB;
- zip archives up to 1 GB are downloaded whole and checked against their checksum;
  from larger ones only the chosen members are fetched by HTTP range requests;
- everything else is left out: videos, images, code, raw signal files over 50 MB,
  and pickles and other formats whose loading can run code.

The members of each zip archive are listed from its central directory, read by HTTP
range requests (``remotezip``), and chosen by the same rules; each member is pinned by
its size and CRC-32. The listing is cached in ``data/cache/candidate_search``.

Writes ``analyses/heldout/sources.yaml`` and ``analyses/development_sources/sources_search.yaml``,
which ``python -m opendv.analysis.confirmatory fetch --sources ...`` downloads and verifies.

    python -m analyses.candidate_search.build_sources [--offline]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import yaml

from analyses.candidate_search.search_open_data import CACHE, UNSAFE, file_records

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLIT = HERE / "heldout_split.csv"
OUTPUTS = {
    "heldout": ROOT / "analyses" / "heldout" / "sources.yaml",
    "development": ROOT / "analyses" / "development_sources" / "sources_search.yaml",
}
DATA_DIRS = {"heldout": "data/raw/heldout", "development": "data/raw/development"}

TABLES = {".csv", ".tsv", ".xlsx", ".xls", ".ods", ".sav", ".dta", ".parquet", ".feather"}
DOCUMENTS = {".pdf", ".docx", ".doc", ".md", ".txt", ".rtf", ".html", ".odt"}
DOC_NAME = re.compile(r"read[ _-]?me|code[ _-]?book|dictionary|data[ _-]?description|questionnaire|"
                      r"variables|metadata", re.IGNORECASE)
README_NAME = re.compile(r"read[ _-]?me|code[ _-]?book|dictionary", re.IGNORECASE)
LEFT_OUT_NAME = re.compile(r"video|unity|\.mov$|\.mp4$|raw[ _-]?emg|eeg|ecg|\bgsr\b|\bppg\b", re.IGNORECASE)
MAX_TABLE = 50_000_000
MAX_DOCUMENT = 20_000_000
MAX_WHOLE_ARCHIVE = 1_000_000_000
LOG_FOLDER_TABLES = 15  # a folder with this many tables holds per-trial or per-session logs

# One readable identifier per dataset (data DOI -> source_id).
SOURCE_IDS = {
    "10.5281/zenodo.20627050": "driver_trust_index_2026",
    "10.17605/osf.io/dvsbh": "av_acceptance_diverse_2025",
    "10.17605/osf.io/6un4h": "ai_label_trust_2025",
    "10.4121/18bb5d48-c1e5-4d50-ad71-d9010e4494d9.v2": "partial_automation_ui_2024",
    "10.4121/14465715.v2": "ehmi_behaviour_crowd_2021",
    "10.5061/dryad.5qfttdzpg": "cooperativity_scale_2026",
    "10.4121/b61ddc8f-0c07-44b5-b570-6208edc2f4cb.v2": "passenger_certainty_2026",
    "10.4121/16866709.v3": "driver_eye_contact_2022",
    "10.4121/95a4bb4e-3ca4-4fcc-ba34-4be76a9ab578.v1": "perceived_risk_merging_2022",
    "10.6084/m9.figshare.13385834": "music_intensity_driving_2021",
    "10.18419/darus-2824": "tactile_sa_display_2022",
    "10.6084/m9.figshare.13348046": "music_tempo_taskload_2022",
    "10.5281/zenodo.15281988": "ndrt_takeover_l3_2025",
    "10.4121/6dfe7bd5-16f7-4e8d-9ab7-f8203b902310.v1": "ehmi_flash_text_2023",
    "10.4121/12682826": "ehmi_location_2019",
    "10.4121/20224281.v2": "coupled_sim_2022",
    "10.4121/57e36cf2-4f74-4d2f-86e2-7fae7250438a.v1": "driving_style_takeover_2024",
    "10.4121/858e4b36-895b-4c84-928a-d1cdcf0920d6.v1": "ehmi_gimmick_2022",
    "10.5061/dryad.vx0k6dk33": "tactile_dvi_braking_2026",
    "10.4121/14406944.v1": "ped_gesture_vr_2021",
    "10.4121/20005565": "coupled_hmd_2022",
    "10.4121/20103188.v2": "cyclist_av_decisions_2022",
    "10.4121/8e1c6604-53c7-4905-8ef9-4a6b2acc4e7a.v1": "ehmi_infrastructure_2024",
    "10.4121/a1f9f15c-1213-4657-8e4d-a154a725d747.v1": "ar_pedestrian_outdoor_2024",
    "10.4121/12948650.v3": "ehmi_colours_crowd_2020",
    "10.4121/14077352.v2": "driving_style_automated_2021",
    "10.4121/uuid:9ea78136-3ffc-4194-84f7-3116b6a55758": "ehmi_survey_crowd_2019",
    "10.4121/uuid:4343cfa8-c246-4bac-bcfa-e3725aceacf7": "ped_woz_field_2018",
    "10.6084/m9.figshare.1311647.v1": "visual_fidelity_curves_2015",
    "10.4121/14096067.v2": "bioinspired_intent_2021",
    "10.4121/de7105b7-6200-4e9b-bc17-db3613bef6a7.v3": "passenger_uncertainty_2026",
    "10.4121/14046107.v2": "lateral_position_2021",
    "10.4121/14068553.v1": "ehmi_perspective_2021",
    "10.4121/21603678.v1": "ar_pedestrian_online_2023",
    "10.5683/sp3/cinmpg": "car_sickness_physio_2023",
    "10.7910/dvn/8srajs": "l2_transfer_control_2022",
    "10.17605/osf.io/psmrx": "ehmi_car_body_2020",
    "10.5281/zenodo.19040256": "physio_hrc_2025",
    "10.5281/zenodo.14198913": "shared_autonomy_2026",
    "10.6084/m9.figshare.31198108": "dev_creativity_eyetracking_2026",
    "10.5281/zenodo.23417050": "robot_teaching_2023",
    "10.17605/osf.io/uxk58": "tool_embodiment_2025",
    "10.4121/d55393f4-f6b6-4076-86c1-7bf351f16787.v1": "function_allocation_hrc_2026",
    "10.5281/zenodo.10522829": "multiscale_teleport_2024",
    "10.17605/osf.io/prsxt": "robot_feeding_2025",
    "10.5061/dryad.bcc2fqzw2": "llm_chatbot_care_2026",
    "10.5281/zenodo.14718504": "reflexion_mapping_2025",
    "10.18130/v3/97zjnl": "tlx_formats_uas_2023",
    "10.5281/zenodo.21411181": "nl2tap_filter_2026",
    "10.6084/m9.figshare.14743422": "concurrent_speech_2022",
    "10.5061/dryad.qv9s4mwjp": "acoustic_touch_blv_2023",
    "10.5281/zenodo.18834139": "myocontrol_intent_2025",
    "10.5281/zenodo.14753191": "sensecobot_fusion_2024",
    "10.6084/m9.figshare.30833309": "midair_map_making_2026",
    "10.18419/darus-2525": "proxemics_ar_agent_2022",
    "10.18419/darus-3327": "short_videos_memory_2023",
    "10.5281/zenodo.20574410": "artv_ornaments_2025",
}


def classify(path: str, size: int | None) -> tuple[str, str]:
    """(role, reason): role is ``table``, ``documentation``, ``archive`` or ``skip``."""
    name = PurePosixPath(path)
    suffix = name.suffix.lower()
    size = int(size or 0)
    parts = [p for p in name.parts if p not in ("", ".")]
    if "__MACOSX" in parts or name.name.startswith("._") or name.name == ".DS_Store":
        return "skip", "macOS metadata"
    if suffix in UNSAFE:
        return "skip", "loading can run code"
    if LEFT_OUT_NAME.search(path):
        return "skip", "video, project or raw signal"
    if suffix == ".zip":
        return "archive", ""
    if suffix in TABLES and README_NAME.search(name.name):
        return "documentation", ""
    if suffix in TABLES:
        return ("table", "") if size <= MAX_TABLE else ("skip", "table over 50 MB")
    if suffix in DOCUMENTS and DOC_NAME.search(name.name):
        return ("documentation", "") if size <= MAX_DOCUMENT else ("skip", "document over 20 MB")
    return "skip", f"{suffix or 'no suffix'} not needed"


def list_zip(url: str, offline: bool = False) -> dict[str, Any]:
    """Members of a remote zip archive (name, size, CRC-32) from its central directory, cached."""
    key = "zip_" + hashlib.sha256(url.encode()).hexdigest()[:32]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["body"]
    if offline:
        return {"members": [], "error": "not cached"}
    import zipfile

    from remotezip import RemoteZip, RemoteZipError

    try:
        # Some hosts answer a suffix range larger than the file with a broken Content-Range,
        # so the file size is asked for first.
        with RemoteZip(url, support_suffix_range=False, timeout=120) as archive:
            body: dict[str, Any] = {"members": [
                {"path": info.filename, "size": info.file_size, "crc32": f"{info.CRC:08x}"}
                for info in archive.infolist() if not info.is_dir()
            ], "error": ""}
    except (RemoteZipError, zipfile.BadZipFile, OSError, ValueError) as error:  # no range support, not a zip
        body = {"members": [], "error": f"{type(error).__name__}: {error}"[:300]}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "fetched": date.today().isoformat(), "body": body}),
                    encoding="utf-8")
    return body


def pin_dataset(row: pd.Series, offline: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(sources.yaml entry, one log row per file and archive member with its role)."""
    records, note = file_records(row["doi"], {"url": row["url"]}, offline)
    log: list[dict[str, Any]] = []
    for record in records:
        role, reason = classify(record["path"], record["size"])
        if role != "skip" and not (record.get("md5") or record.get("sha256")):
            role, reason = "skip", "no published checksum"
        if role != "skip" and not record.get("url"):
            role, reason = "skip", "no download URL"
        if role != "skip" and record.get("restricted"):
            role, reason = "skip", "restricted access"
        log.append({"doi": row["doi"], "file": record["path"], "member": "", "bytes": record["size"],
                    "role": role, "reason": reason, "record": record})
        if role != "archive":
            continue
        listing = list_zip(record["url"], offline)
        if listing["error"]:
            log[-1]["role"], log[-1]["reason"] = "skip", f"members not listed ({listing['error'][:80]})"
        for member in listing["members"]:
            m_role, m_reason = classify(member["path"], member["size"])
            if m_role == "archive":
                m_role, m_reason = "skip", "nested archive"
            log.append({"doi": row["doi"], "file": record["path"], "member": member["path"],
                        "bytes": member["size"], "role": m_role, "reason": m_reason, "record": member})
    # Per-trial and per-session logs: a folder holding many tables.
    folder_counts: dict[tuple[str, str], int] = {}
    for item in log:
        if item["role"] == "table":
            key = (item["file"] if item["member"] else "", str(PurePosixPath(item["member"] or item["file"]).parent))
            folder_counts[key] = folder_counts.get(key, 0) + 1
    for item in log:
        key = (item["file"] if item["member"] else "", str(PurePosixPath(item["member"] or item["file"]).parent))
        if item["role"] == "table" and folder_counts[key] >= LOG_FOLDER_TABLES:
            item["role"], item["reason"] = "skip", f"log folder ({folder_counts[key]} tables)"

    files: list[dict[str, Any]] = []
    for item in log:
        if item["member"] or item["role"] == "skip":
            continue
        record = item["record"]
        spec: dict[str, Any] = {"path": record["path"], "size": record["size"],
                                **{k: record[k] for k in ("md5", "sha256") if record.get(k)}, "url": record["url"]}
        if item["role"] == "documentation":
            spec["role"] = "codebook"
        if item["role"] == "archive":
            members = [m for m in log if m["member"] and m["file"] == record["path"] and m["role"] != "skip"]
            if not members:
                item["role"], item["reason"] = "skip", "no needed member"
                continue
            spec["fetch"] = "whole" if int(record["size"] or 0) <= MAX_WHOLE_ARCHIVE else "range"
            spec["members"] = [{"path": m["member"], "size": m["bytes"], "crc32": m["record"]["crc32"]}
                               for m in members]
        files.append(spec)
    entry: dict[str, Any] = {
        "source_id": SOURCE_IDS[row["doi"]],
        "title": row["title"],
        "venue": f"{row['venue']} {int(row['paper_year'])}" if pd.notna(row["paper_year"]) else row["venue"],
        "doi": row["paper_doi"],
        "data_doi": row["doi"],
        "license": row["licence"],
        "repository": row["url"],
        "domain": row["domain"],
        "stratum": row["stratum"],
    }
    if not any(item["role"] == "table" for item in log):
        counts: dict[str, int] = {}
        for i in log:
            suffix = PurePosixPath(i["member"] or i["file"]).suffix.lower() or "no suffix"
            if i["reason"] != "macOS metadata" and i["role"] != "documentation" and suffix != ".zip":
                label = f"{suffix} in log folders" if i["reason"].startswith("log folder") else suffix
                counts[label] = counts.get(label, 0) + 1
        found = ", ".join(f"{n} {label}" for label, n in sorted(counts.items(), key=lambda kv: -kv[1])[:6])
        entry["dropped"] = f"no participant-level table the pipeline reads; other files: {found}"
        files = []
    entry["files"] = files
    if note:
        entry["listing_note"] = note
    for item in log:
        item.pop("record")
    return entry, log


def build(offline: bool = False) -> pd.DataFrame:
    split = pd.read_csv(SPLIT, dtype={"paper_year": "Float64"})
    missing = sorted(set(split["doi"]) - set(SOURCE_IDS))
    if missing:
        raise SystemExit(f"no source_id for {missing}")
    logs = []
    for set_name, path in OUTPUTS.items():
        entries = []
        for _, row in split[split["set"] == set_name].sort_values("doi").iterrows():
            entry, log = pin_dataset(row, offline)
            entries.append(entry)
            logs.extend({"set": set_name, "source_id": entry["source_id"], **r} for r in log)
        header = (
            f"# {set_name.capitalize()} datasets of the 2026-11-05 search ({SPLIT.relative_to(ROOT).as_posix()}).\n"
            "# Generated by `python -m analyses.candidate_search.build_sources`; each file is pinned by its size\n"
            "# and the checksum its repository publishes, each archive member by its size and CRC-32.\n"
            "# Download and verify with\n"
            f"#   python -m opendv.analysis.confirmatory fetch --sources {path.relative_to(ROOT).as_posix()} \\\n"
            f"#     --data-dir {DATA_DIRS[set_name]}\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump({"sources": entries}, sort_keys=False, allow_unicode=True, width=120)
        path.write_text(header + body, encoding="utf-8", newline="\n")
    frame = pd.DataFrame(logs)
    frame.to_csv(HERE / "file_selection.csv", index=False, lineterminator="\n")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use cached listings only")
    args = parser.parse_args()
    frame = build(offline=args.offline)
    summary = frame.groupby(["set", "role"])["bytes"].agg(["size", "sum"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
