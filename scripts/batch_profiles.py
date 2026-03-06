"""Dataset-type and mapping-domain helpers for batch standardization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

DATASET_TYPE_RESULTS = "results_table"
DATASET_TYPE_QUESTIONNAIRE = "questionnaire"
DATASET_TYPE_SENSOR = "sensor_stream"
DATASET_TYPE_DETECTION = "object_detection"
DATASET_TYPE_PROCESS = "process_log"

MAPPING_DOMAIN_DV = "dv"
MAPPING_DOMAIN_SENSOR = "sensor"
MAPPING_DOMAIN_DETECTION = "detection"
MAPPING_DOMAIN_METADATA = "metadata"
MAPPING_DOMAIN_CUSTOM = "custom"
MAPPING_DOMAIN_UNMAPPED = "unmapped"
MAPPING_DOMAIN_BLOCKED = "blocked"

DETECTION_COLUMN_MARKERS = {
    "video",
    "frame",
    "objectid",
    "object_id",
    "class",
    "x",
    "y",
    "width",
    "height",
    "confidence",
}
SENSOR_COLUMN_MARKERS = {
    "gazeforward",
    "gazeorigin",
    "leftpupildiameterinmm",
    "rightpupildiameterinmm",
    "leftirisdiameterinmm",
    "rightirisdiameterinmm",
    "projectedx",
    "projectedy",
    "quadlocalx",
    "quadlocaly",
    "pixelx",
    "pixely",
}
PROCESS_COLUMN_MARKERS = {
    "timestamp",
    "phase",
    "run",
    "ispareto",
    "framenumber",
    "capturetime",
    "logtime",
    "videoframe",
    "hmdposition",
    "hmdrotation",
    "unnamed0",
    "unnamed1",
}


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(value).strip().lower())


def classify_dataset_type(relative_path: Path, column_names: Iterable[str]) -> str:
    path_text = relative_path.as_posix().lower()
    raw_columns = [str(column) for column in column_names]
    normalized_columns = {normalize_column_name(column) for column in raw_columns}

    detection_hits = len(normalized_columns & DETECTION_COLUMN_MARKERS)
    sensor_hits = len(normalized_columns & SENSOR_COLUMN_MARKERS)
    process_hits = len(normalized_columns & PROCESS_COLUMN_MARKERS)
    question_hits = sum(
        1
        for column in raw_columns
        if "?" in column
        or len(column.strip()) >= 50
        or column.strip().lower().startswith(("what ", "which ", "how ", "i "))
    )

    if "yolo" in path_text or detection_hits >= 5:
        return DATASET_TYPE_DETECTION
    if sensor_hits >= 4:
        return DATASET_TYPE_SENSOR
    if question_hits >= max(3, len(raw_columns) // 2):
        return DATASET_TYPE_QUESTIONNAIRE
    if process_hits >= max(3, len(raw_columns) // 3):
        return DATASET_TYPE_PROCESS
    return DATASET_TYPE_RESULTS


def infer_mapping_domain(
    canonical_name: str,
    mapping_status: str,
    canonical_domain_lookup: dict[str, str],
    mapping_source: str | None = None,
) -> str:
    if mapping_status == "blocked":
        return MAPPING_DOMAIN_BLOCKED
    if mapping_status == "unmapped":
        return MAPPING_DOMAIN_UNMAPPED

    domain = canonical_domain_lookup.get(str(canonical_name))
    if domain:
        return domain

    if mapping_source and mapping_source not in {"llm_deduction", "in_memory_mapping"}:
        return MAPPING_DOMAIN_CUSTOM

    return MAPPING_DOMAIN_CUSTOM
