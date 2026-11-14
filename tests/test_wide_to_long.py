"""The ``wide_to_long`` table option and stacked sheets (opendv.data_loaders)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opendv.data_loaders import SHEET_COLUMN, TABLE_OPTION_KEYS, load_input_file


def _csv(tmp_path: Path, frame: pd.DataFrame, name: str = "wide.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


WIDE = pd.DataFrame({
    "participant": ["p1", "p2", None],
    "age": [30, 40, None],
    "tlx_A": [10, 20, None],
    "tlx_B": [30, 40, None],
    "sus_A": [70, 80, None],
    "sus_B": [60, 50, None],
})


def test_option_is_registered():
    assert "wide_to_long" in TABLE_OPTION_KEYS


def test_columns_of_each_level_become_rows(tmp_path):
    path = _csv(tmp_path, WIDE)
    frame = load_input_file(str(path), {"wide_to_long": {
        "id": ["participant", "age"], "level": "condition", "levels": ["A", "B"],
        "columns": {"tlx": ["tlx_A", "tlx_B"], "sus": ["sus_A", "sus_B"]},
    }})
    assert list(frame.columns) == ["participant", "age", "condition", "tlx", "sus"]
    # The blank third row has no id and is dropped.
    assert len(frame) == 4
    row = frame[(frame.participant == "p2") & (frame.condition == "B")].iloc[0]
    assert row.tlx == 40 and row.sus == 50 and row.age == 40


def test_a_level_without_a_measure_is_missing(tmp_path):
    path = _csv(tmp_path, WIDE)
    frame = load_input_file(str(path), {"wide_to_long": {
        "id": ["participant"], "level": "condition", "levels": ["A", "B"],
        "columns": {"tlx": ["tlx_A", "tlx_B"], "sus": ["sus_A", None]},
    }})
    assert frame.loc[frame.condition == "B", "sus"].isna().all()
    assert frame.loc[frame.condition == "A", "sus"].tolist() == [70, 80]


def test_names_are_matched_stripped_and_must_exist(tmp_path):
    path = _csv(tmp_path, WIDE.rename(columns={"tlx_A": " tlx_A "}))
    frame = load_input_file(str(path), {"wide_to_long": {
        "id": ["participant"], "levels": ["A", "B"], "columns": {"tlx": ["tlx_A", "tlx_B"]}}})
    assert frame["level"].tolist() == ["A", "A", "B", "B"]
    with pytest.raises(ValueError, match="not found"):
        load_input_file(str(path), {"wide_to_long": {
            "id": ["participant"], "levels": ["A", "B"], "columns": {"tlx": ["tlx_A", "tlx_C"]}}})


def test_each_new_column_needs_one_input_per_level(tmp_path):
    path = _csv(tmp_path, WIDE)
    with pytest.raises(ValueError, match="one input column per level"):
        load_input_file(str(path), {"wide_to_long": {
            "id": ["participant"], "levels": ["A", "B"], "columns": {"tlx": ["tlx_A"]}}})
    with pytest.raises(ValueError, match="takes id, level"):
        load_input_file(str(path), {"wide_to_long": {"id": ["participant"], "melt": True}})
    with pytest.raises(ValueError, match="overwrite id"):
        load_input_file(str(path), {"wide_to_long": {
            "id": ["participant"], "level": "participant", "levels": ["A", "B"],
            "columns": {"tlx": ["tlx_A", "tlx_B"]}}})


def test_sheets_stack_then_reshape_to_one_row_per_participant_and_condition(tmp_path):
    # One sheet per subscale, one column per condition (as in a NASA-TLX workbook).
    path = tmp_path / "tlx.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"participant": ["p1", "p2", None], "c1": [5, 10, np.nan], "c2": [15, 20, np.nan]}
                     ).to_excel(writer, sheet_name="mental", index=False)
        pd.DataFrame({"participant": ["p1", "p2", None], "c1": [50, 60, np.nan], "c2": [70, 80, np.nan]}
                     ).to_excel(writer, sheet_name="effort", index=False)
        pd.DataFrame({"note": ["ignored"]}).to_excel(writer, sheet_name="readme", index=False)
    stacked = load_input_file(str(path), {"sheet": ["mental", "effort"]})
    assert SHEET_COLUMN in stacked.columns and len(stacked) == 4  # Excel drops the empty rows
    frame = load_input_file(str(path), {
        "sheet": ["mental", "effort"],
        "wide_to_long": {"id": ["participant", "sheet"], "level": "condition", "levels": ["c1", "c2"],
                         "columns": {"score": ["c1", "c2"]}},
        "pivot": {"index": ["participant", "condition"], "columns": "sheet", "values": "score"},
    })
    assert sorted(frame.columns) == ["condition", "effort", "mental", "participant"]
    assert len(frame) == 4
    row = frame[(frame.participant == "p2") & (frame.condition == "c2")].iloc[0]
    assert row.mental == 20 and row.effort == 80


def test_wide_to_long_then_rename(tmp_path):
    path = _csv(tmp_path, WIDE)
    frame = load_input_file(str(path), {
        "wide_to_long": {"id": ["participant"], "level": "condition", "levels": ["A", "B"],
                         "columns": {"tlx": ["tlx_A", "tlx_B"]}},
        "rename": {"tlx": "TLX_total"},
    })
    assert list(frame.columns) == ["participant", "condition", "TLX_total"]


def test_row_id_numbers_rows_after_sheets_are_stacked(tmp_path):
    path = tmp_path / "ueq.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"item_1": [1, 2]}).to_excel(writer, sheet_name="Data1", index=False)
        pd.DataFrame({"item_1": [3, 4, 5]}).to_excel(writer, sheet_name="Data2", index=False)
    frame = load_input_file(str(path), {"sheet": ["Data1", "Data2"], "row_id": "respondent"})
    assert frame["respondent"].tolist() == [1, 2, 3, 4, 5]
    assert frame.columns[0] == "respondent"
    with pytest.raises(ValueError, match="already a column"):
        load_input_file(str(path), {"sheet": "Data1", "row_id": "item_1"})
