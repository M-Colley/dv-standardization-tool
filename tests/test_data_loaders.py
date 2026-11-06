"""Direct unit tests for scripts.data_loaders.

Validates the extracted format-registry, sniffers, and loader entry
points as a standalone import (not via the
``scripts.convert_dv`` re-export). Behavioural coverage of
``load_input_file`` and ``save_output_file`` end-to-end already lives
in ``tests/test_convert_dv.py`` and continues to pass through the
re-export path.
"""

from __future__ import annotations

import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from scripts.data_loaders import (
    PICKLE_SUFFIXES,
    _FORMAT_REGISTRY,
    _MAGIC_BYTES,
    PickleRefusedError,
    _coerce_pickled_payload_to_dataframe,
    _detect_decimal,
    _detect_delimiter,
    _detect_encoding,
    _register_format,
    _sniff_format,
    load_input_file,
    save_output_file,
)


class FormatRegistrySurfaceTests(unittest.TestCase):
    def test_text_extensions_registered(self):
        for ext in (".csv", ".tsv", ".txt", ".dat"):
            self.assertIn(ext, _FORMAT_REGISTRY)

    def test_excel_extensions_registered(self):
        for ext in (".xlsx", ".xls", ".xlsm", ".ods"):
            self.assertIn(ext, _FORMAT_REGISTRY)

    def test_pickle_parquet_etc_registered(self):
        for ext in (".pkl", ".pickle", ".parquet", ".sav", ".zsav", ".dta",
                    ".feather", ".arrow", ".json", ".jsonl", ".ndjson"):
            self.assertIn(ext, _FORMAT_REGISTRY)


class RegisterFormatDecoratorTests(unittest.TestCase):
    def test_decorator_registers_extension(self):
        # Use an unlikely-to-conflict extension and clean up afterwards.
        before = dict(_FORMAT_REGISTRY)
        try:
            @_register_format(".__test_ext__")
            def _stub(_path: Path) -> pd.DataFrame:
                return pd.DataFrame()

            self.assertIn(".__test_ext__", _FORMAT_REGISTRY)
            self.assertIs(_FORMAT_REGISTRY[".__test_ext__"], _stub)
        finally:
            _FORMAT_REGISTRY.clear()
            _FORMAT_REGISTRY.update(before)

    def test_decorator_normalizes_to_lowercase(self):
        before = dict(_FORMAT_REGISTRY)
        try:
            @_register_format(".UPPER_EXT")
            def _stub(_path: Path) -> pd.DataFrame:
                return pd.DataFrame()

            self.assertIn(".upper_ext", _FORMAT_REGISTRY)
        finally:
            _FORMAT_REGISTRY.clear()
            _FORMAT_REGISTRY.update(before)


class SniffFormatTests(unittest.TestCase):
    def test_parquet_magic(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.bin"
            p.write_bytes(b"PAR1somecontent")
            self.assertEqual(_sniff_format(p), ".parquet")

    def test_json_object_magic(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.bin"
            p.write_bytes(b'{"a":1}')
            self.assertEqual(_sniff_format(p), ".json")

    def test_returns_none_for_unknown_bytes(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.bin"
            p.write_bytes(b"\xde\xad\xbe\xef")
            self.assertIsNone(_sniff_format(p))

    def test_magic_bytes_table_includes_known_signatures(self):
        magic_set = {magic for magic, _ in _MAGIC_BYTES}
        self.assertIn(b"PAR1", magic_set)
        self.assertIn(b"$FL2", magic_set)


class DetectEncodingTests(unittest.TestCase):
    def test_utf8_bom_detected(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n")
            self.assertEqual(_detect_encoding(p), "utf-8-sig")

    def test_utf16_le_bom_detected(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_bytes(b"\xff\xfehello")
            self.assertEqual(_detect_encoding(p), "utf-16-le")

    def test_falls_back_to_utf8_for_ascii(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_text("a,b\n1,2\n", encoding="ascii")
            self.assertEqual(_detect_encoding(p), "utf-8")


class DetectDelimiterTests(unittest.TestCase):
    def test_csv_comma(self):
        sample = "a,b,c\n1,2,3\n4,5,6\n"
        self.assertEqual(_detect_delimiter(sample), ",")

    def test_tsv_tab(self):
        sample = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
        self.assertEqual(_detect_delimiter(sample, prefer="\t"), "\t")

    def test_semicolon_when_consistent(self):
        sample = "a;b;c\n1;2;3\n4;5;6\n"
        self.assertEqual(_detect_delimiter(sample), ";")

    def test_falls_back_to_prefer_when_no_rows(self):
        self.assertEqual(_detect_delimiter("", prefer=";"), ";")
        self.assertEqual(_detect_delimiter("", prefer=None), ",")


class DetectDecimalTests(unittest.TestCase):
    def test_decimal_comma_in_semicolon_file(self):
        sample = "Success;Time;Type\nTrue;24,825;Positioning,Rotating\nFalse;30;Rotating\n"
        self.assertEqual(_detect_decimal(sample, ";"), ",")

    def test_comma_delimited_file_keeps_decimal_point(self):
        self.assertEqual(_detect_decimal('a,b\n"1,5",2\n', ","), ".")

    def test_any_decimal_point_keeps_the_default(self):
        # With decimal="," pandas would read 1.5 as text, so mixed files keep ".".
        self.assertEqual(_detect_decimal("a;b\n0,5;1.5\n", ";"), ".")
        self.assertEqual(_detect_decimal("a;b\n1;2\n", ";"), ".")

    def test_header_and_vector_strings_are_ignored(self):
        sample = "x,1;pos\n0,25;(0.00, -0.44, 0.00)\n"
        self.assertEqual(_detect_decimal(sample, ";"), ",")


class CoercePickledPayloadTests(unittest.TestCase):
    def test_dataframe_passthrough(self):
        df = pd.DataFrame({"a": [1, 2]})
        out = _coerce_pickled_payload_to_dataframe(df)
        self.assertIs(out, df)

    def test_series_to_frame(self):
        s = pd.Series([1, 2, 3], name="x")
        out = _coerce_pickled_payload_to_dataframe(s)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(list(out.columns), ["x"])

    def test_dict_of_lists(self):
        out = _coerce_pickled_payload_to_dataframe({"a": [1, 2], "b": [3, 4]})
        self.assertEqual(list(out.columns), ["a", "b"])
        self.assertEqual(len(out), 2)

    def test_empty_dict_returns_empty_dataframe(self):
        out = _coerce_pickled_payload_to_dataframe({})
        self.assertIsInstance(out, pd.DataFrame)
        self.assertTrue(out.empty)

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _coerce_pickled_payload_to_dataframe(42)
        self.assertIn("Unsupported pickle payload", str(ctx.exception))


class LoadInputFileTests(unittest.TestCase):
    def test_loads_csv_round_trip(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
            out = load_input_file(str(p))
            self.assertEqual(list(out.columns), ["a", "b"])
            self.assertEqual(out.shape, (2, 2))

    def test_loads_decimal_comma_csv_as_numbers(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_text("UserID;Time;Type\n2;0,581;Selection\n2;24,825;Positioning,Rotating\n", encoding="utf-8")
            out = load_input_file(str(p))
            self.assertEqual(out["Time"].tolist(), [0.581, 24.825])
            self.assertEqual(out["Type"].tolist(), ["Selection", "Positioning,Rotating"])

    def test_loads_json_records(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
            out = load_input_file(str(p))
            self.assertEqual(out["a"].tolist(), [1, 2])

    def test_unsupported_extension_raises(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.unknownext"
            p.write_text("garbage", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_input_file(str(p))
            self.assertIn("Unsupported input format", str(ctx.exception))

    def test_falls_back_to_magic_byte_sniff(self):
        # Extension is unknown but magic bytes say JSON.
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.unknownext"
            p.write_text(json.dumps([{"a": 1}]), encoding="utf-8")
            out = load_input_file(str(p))
            self.assertEqual(out["a"].tolist(), [1])


def _record_unpickle(value: str) -> str:
    _RunsCodeWhenUnpickled.calls.append(value)
    return value


class _RunsCodeWhenUnpickled:
    """Unpickling this object calls ``_record_unpickle`` (stands in for any callable)."""

    calls: list[str] = []

    def __reduce__(self):
        return (_record_unpickle, ("ran",))


class PickleRefusalTests(unittest.TestCase):
    """Pickles are refused unless the caller opts in: unpickling runs code."""

    def setUp(self):
        _RunsCodeWhenUnpickled.calls.clear()

    def _payload_file(self, tmp: str, name: str = "evil.pkl") -> Path:
        path = Path(tmp) / name
        path.write_bytes(pickle.dumps(_RunsCodeWhenUnpickled()))
        return path

    def test_payload_runs_code_when_unpickled(self):
        # Guards the tests below: the payload really does call a function.
        pickle.loads(pickle.dumps(_RunsCodeWhenUnpickled()))
        self.assertEqual(_RunsCodeWhenUnpickled.calls, ["ran"])

    def test_load_input_file_refuses_pickles_without_running_them(self):
        with TemporaryDirectory() as tmp:
            for name in ("evil.pkl", "evil.pickle", "EVIL.PKL"):
                path = self._payload_file(tmp, name)
                with self.subTest(name=name), mock.patch("pandas.read_pickle") as read_pickle:
                    with self.assertRaises(PickleRefusedError) as ctx:
                        load_input_file(str(path))
                    read_pickle.assert_not_called()
                    self.assertIn("runs code", str(ctx.exception))
        self.assertEqual(_RunsCodeWhenUnpickled.calls, [])

    def test_registry_loader_refuses_pickles(self):
        with TemporaryDirectory() as tmp:
            path = self._payload_file(tmp)
            for suffix in PICKLE_SUFFIXES:
                with self.subTest(suffix=suffix), self.assertRaises(PickleRefusedError):
                    _FORMAT_REGISTRY[suffix](path)
        self.assertEqual(_RunsCodeWhenUnpickled.calls, [])

    def test_refusal_is_a_value_error(self):
        # Callers that treat unreadable files as ValueError keep working.
        self.assertTrue(issubclass(PickleRefusedError, ValueError))

    def test_allow_pickle_reads_a_trusted_dataframe(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mine.pkl"
            expected = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            expected.to_pickle(path)
            pd.testing.assert_frame_equal(load_input_file(str(path), allow_pickle=True), expected)


class SaveOutputFileTests(unittest.TestCase):
    def test_csv_round_trip(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.csv"
            df = pd.DataFrame({"a": [1, 2]})
            save_output_file(df, str(p))
            self.assertTrue(p.exists())
            roundtrip = pd.read_csv(p)
            self.assertEqual(roundtrip["a"].tolist(), [1, 2])

    def test_pickle_round_trip(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.pkl"
            df = pd.DataFrame({"a": [1, 2]})
            save_output_file(df, str(p))
            roundtrip = pd.read_pickle(p)
            self.assertEqual(roundtrip["a"].tolist(), [1, 2])

    def test_unsupported_extension_raises(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.weird"
            with self.assertRaises(ValueError) as ctx:
                save_output_file(pd.DataFrame(), str(p))
            self.assertIn("Unsupported output format", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
