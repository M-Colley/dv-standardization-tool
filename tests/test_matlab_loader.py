"""MATLAB .mat files in the reader registry (opendv.data_loaders._load_matlab).

The files are written with ``scipy.io.savemat`` in the shapes open datasets use:
a struct of equal-length vectors, a struct array with one element per
participant, a matrix with a separate cell array of column names, and a
participants x conditions array of one measure.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.io import savemat

from opendv.archive_utils import DATA_FILE_SUFFIXES, OPT_IN_SUFFIXES
from opendv.data_loaders import (
    _FORMAT_REGISTRY,
    MATLAB_OPTION_KEYS,
    TABLE_OPTION_KEYS,
    load_input_file,
    table_options_for,
)
from opendv.run_batch_standardization import _match_files


def _cells(*values: str) -> np.ndarray:
    """A MATLAB cell array of strings (1 x n)."""
    cells = np.empty((1, len(values)), dtype=object)
    for i, value in enumerate(values):
        cells[0, i] = value
    return cells


def _struct_array(records: list[dict]) -> np.ndarray:
    """A MATLAB 1 x n struct array from equal-keyed dicts."""
    fields = list(records[0])
    array = np.zeros((1, len(records)), dtype=[(f, "O") for f in fields])
    for i, record in enumerate(records):
        array[0, i] = tuple(record[f] for f in fields)
    return array


class MatlabRegistryTests(unittest.TestCase):
    def test_mat_is_registered_but_opt_in(self):
        self.assertIn(".mat", _FORMAT_REGISTRY)
        self.assertIn(".mat", OPT_IN_SUFFIXES)
        self.assertNotIn(".mat", DATA_FILE_SUFFIXES)

    def test_matlab_options_are_table_options(self):
        for key in MATLAB_OPTION_KEYS:
            self.assertIn(key, TABLE_OPTION_KEYS)


class MatlabLoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, name: str, variables: dict, **kwargs) -> Path:
        path = self.dir / name
        savemat(path, variables, **kwargs)
        return path

    def test_struct_of_vectors_becomes_one_column_per_field(self):
        path = self._save("s.mat", {"data": {
            "participant": np.array([1, 2, 3]),
            "mental_demand": np.array([55.0, 60.0, 20.0]),
            "condition": np.array(["a", "b", "c"], dtype=object),
        }})
        frame = load_input_file(str(path))
        self.assertEqual(list(frame.columns), ["participant", "mental_demand", "condition"])
        self.assertEqual(frame["mental_demand"].tolist(), [55.0, 60.0, 20.0])
        self.assertEqual(frame["condition"].tolist(), ["a", "b", "c"])

    def test_top_level_vectors_of_one_length_read_as_one_table(self):
        path = self._save("flat.mat", {"id": np.arange(1, 5), "rt": np.array([0.5, 0.6, 0.7, 0.8])})
        frame = load_input_file(str(path))
        self.assertEqual(sorted(frame.columns), ["id", "rt"])
        self.assertEqual(len(frame), 4)

    def test_scalar_fields_repeat_and_matrix_fields_split(self):
        path = self._save("p.mat", {"P": {
            "id": 7,
            "trust": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        }})
        frame = load_input_file(str(path), {"variable": "P"})
        self.assertEqual(list(frame.columns), ["id", "trust_1", "trust_2"])
        self.assertEqual(frame["id"].tolist(), [7, 7, 7])
        self.assertEqual(frame["trust_2"].tolist(), [2.0, 4.0, 6.0])

    def test_nested_struct_fields_are_dotted(self):
        path = self._save("n.mat", {"R": {"tlx": {"mental": np.array([1.0, 2.0]), "effort": np.array([3.0, 4.0])},
                                          "id": np.array([10, 11])}})
        frame = load_input_file(str(path), {"variable": "R"})
        self.assertEqual(sorted(frame.columns), ["id", "tlx.effort", "tlx.mental"])
        frame = load_input_file(str(path), {"variable": "R.tlx"})
        self.assertEqual(sorted(frame.columns), ["effort", "mental"])

    def test_struct_array_stacks_elements_with_an_index(self):
        path = self._save("sa.mat", {"subj": _struct_array([
            {"id": 1, "rt": np.array([0.4, 0.5])},
            {"id": 2, "rt": np.array([0.6, 0.7])},
        ])})
        frame = load_input_file(str(path), {"variable": "subj"})
        self.assertEqual(list(frame.columns), ["subj_index", "id", "rt"])
        self.assertEqual(frame["subj_index"].tolist(), [1, 1, 2, 2])
        self.assertEqual(frame["id"].tolist(), [1, 1, 2, 2])
        self.assertEqual(frame["rt"].tolist(), [0.4, 0.5, 0.6, 0.7])

    def test_matrix_columns_from_a_list_or_a_variable(self):
        matrix = np.array([[1, 50.0, 3.2], [2, 70.0, 2.8]])
        path = self._save("m.mat", {"M": matrix, "header": _cells("participant", "tlx", "rt")})
        by_list = load_input_file(str(path), {"variable": "M", "columns": ["participant", "tlx", "rt"]})
        by_variable = load_input_file(str(path), {"variable": "M", "columns": "header"})
        pd.testing.assert_frame_equal(by_list, by_variable)
        self.assertEqual(by_list["tlx"].tolist(), [50.0, 70.0])
        default = load_input_file(str(path), {"variable": "M"})
        self.assertEqual(list(default.columns), ["M_1", "M_2", "M_3"])

    def test_dims_turn_an_array_into_a_long_table(self):
        # 2 participants x 3 conditions of one measure.
        scores = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        path = self._save("d.mat", {"tlx": scores})
        frame = load_input_file(str(path), {"variable": "tlx", "dims": ["participant", "condition"]})
        self.assertEqual(list(frame.columns), ["participant", "condition", "tlx"])
        row = frame[(frame["participant"] == 2) & (frame["condition"] == 3)]
        self.assertEqual(row["tlx"].item(), 60.0)
        self.assertEqual(len(frame), 6)

    def test_three_dimensional_array_needs_dims(self):
        path = self._save("a3.mat", {"A": np.arange(24.0).reshape(2, 3, 4)})
        with self.assertRaisesRegex(ValueError, "dims"):
            load_input_file(str(path), {"variable": "A"})
        frame = load_input_file(str(path), {"variable": "A", "dims": ["p", "c", "t"]})
        self.assertEqual(len(frame), 24)
        self.assertEqual(frame.loc[(frame.p == 2) & (frame.c == 1) & (frame.t == 4), "A"].item(), 15.0)

    def test_wrong_dims_length_is_refused(self):
        path = self._save("w.mat", {"A": np.ones((2, 3))})
        with self.assertRaisesRegex(ValueError, "must name each"):
            load_input_file(str(path), {"variable": "A", "dims": ["only_one"]})

    def test_unequal_fields_are_refused_with_their_lengths(self):
        path = self._save("u.mat", {"S": {"a": np.arange(3), "b": np.arange(4)}})
        with self.assertRaisesRegex(ValueError, r"a \(3\).*b \(4\)"):
            load_input_file(str(path), {"variable": "S"})

    def test_without_variable_the_most_data_rich_one_is_taken(self):
        path = self._save("several.mat", {"small": np.arange(3.0), "big": np.ones((5, 4))})
        frame = load_input_file(str(path))
        self.assertEqual(frame.shape, (5, 4))

    def test_unknown_variable_names_the_available_ones(self):
        path = self._save("x.mat", {"results": {"tlx": np.arange(2.0)}})
        with self.assertRaisesRegex(ValueError, r"results.*\['tlx'\]"):
            load_input_file(str(path), {"variable": "results.trust"})

    def test_matlab_options_are_refused_for_other_formats(self):
        csv = self.dir / "t.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "MATLAB"):
            load_input_file(str(csv), {"variable": "a"})

    def test_hdf5_files_get_a_clear_error(self):
        # A v7.3 MAT file is an HDF5 file with a 512-byte MATLAB header; scipy refuses it.
        path = self.dir / "v73.mat"
        header = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Mon Jan  1 00:00:00 2024 HDF5 schema 1.00 ."
        header = header.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
        path.write_bytes(header.ljust(512, b"\x00") + b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
        with self.assertRaisesRegex(ValueError, "v7.3"):
            load_input_file(str(path))

    def test_pivot_and_rename_apply_after_reading(self):
        path = self._save("long.mat", {"L": {
            "id": np.array([1, 1, 2, 2]),
            "scale": np.array(["mental", "effort", "mental", "effort"], dtype=object),
            "score": np.array([10.0, 20.0, 30.0, 40.0]),
        }})
        frame = load_input_file(str(path), {
            "variable": "L",
            "pivot": {"index": "id", "columns": "scale", "values": "score"},
            "rename": {"mental": "mental_demand"},
        })
        self.assertEqual(sorted(frame.columns), ["effort", "id", "mental_demand"])
        self.assertEqual(frame.set_index("id").loc[2, "mental_demand"], 30.0)

    def test_table_options_glob_reaches_mat_files(self):
        options = table_options_for({"**/sub*.mat": {"variable": "S"}}, "raw/sub01.mat")
        self.assertEqual(options, {"variable": "S"})


class OptInDiscoveryTests(unittest.TestCase):
    def test_mat_and_txt_files_are_read_only_when_include_globs_name_them(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            savemat(root / "data" / "results.mat", {"x": np.arange(3.0)})
            (root / "data" / "README.txt").write_text("notes", encoding="utf-8")
            (root / "data" / "trials.txt").write_text("a\tb\n1\t2\n", encoding="utf-8")
            (root / "data" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            default = {p.name for p in _match_files(root, None, None)}
            self.assertEqual(default, {"table.csv"})

            explicit = {p.name for p in _match_files(root, ["data/*.mat", "data/trials.txt"], None)}
            self.assertEqual(explicit, {"results.mat", "trials.txt"})

            # A broad explicit glob opts in too; a source that names its files decides.
            broad = {p.name for p in _match_files(root, ["data/*.csv"], None)}
            self.assertEqual(broad, {"table.csv"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
