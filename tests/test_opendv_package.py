"""The opendv package: public API, compatibility aliases and entry-point plugins."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import opendv
from opendv import plugins

REPO = Path(__file__).resolve().parents[1]


# ── Public API ───────────────────────────────────────────────────────────────

def test_standardize_maps_schema_aliases_and_keeps_unknown_columns():
    out = opendv.standardize(pd.DataFrame({"TLX1": [5], "participant": ["P1"], "whatever": [1]}))
    assert list(out.columns) == ["mental_demand", "participant", "whatever"]


def test_standardize_with_an_explicit_mapping(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"Q1": [1, 2]}).to_csv(path, index=False)
    assert list(opendv.standardize(path, mapping={"Q1": "usability"}).columns) == ["usability"]


def test_score_uses_the_instrument_registry():
    df = pd.DataFrame({f"SUS{i}": [5 if i % 2 else 1] for i in range(1, 11)})
    assert opendv.score(df)["sus_score"].iloc[0] == pytest.approx(100.0)


def test_version_matches_pyproject_and_cli():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{opendv.__version__}"' in pyproject
    from opendv import cli

    assert cli.__version__ == opendv.__version__


# ── Compatibility aliases ────────────────────────────────────────────────────

@pytest.mark.parametrize("old, new", [
    ("scripts.run_batch_standardization", "opendv.run_batch_standardization"),
    ("scripts.sources.repositories", "opendv.sources.repositories"),
    ("analyses.multi_study_analysis", "opendv.analysis.multi_study_analysis"),
])
def test_old_module_paths_are_the_same_module_objects(old, new):
    import importlib

    assert importlib.import_module(old) is importlib.import_module(new)


def test_old_script_paths_still_run():
    env_python = sys.executable
    for args in (["scripts/validate_schema.py", "schemas/instruments.yaml"],
                 ["-m", "scripts.validate_schema", "schemas/study_designs.yaml"],
                 ["analyses/multi_study_analysis.py", "--help"],
                 ["__main__.py", "--version"]):
        result = subprocess.run([env_python, *args], cwd=REPO, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (args, result.stderr[-800:])


def test_parallel_batch_through_the_old_module_path(tmp_path):
    """python -m scripts.run_batch_standardization with worker processes (spawn) still works."""
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        pd.DataFrame({"TLX1": [i, i + 1], "SUS1": [3, 4]}).to_csv(data / f"s{i}.csv", index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": [
        {"source_id": "par", "source_type": "local_path", "location": str(data)}]}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_batch_standardization", "--manifest", str(manifest),
         "--output-dir", str(tmp_path / "out"), "--disable-llm-deduction", "--workers", "2", "--no-fair-export"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(list((tmp_path / "out" / "standardized" / "par").glob("*-standardized.csv"))) == 3


# ── Plugins ──────────────────────────────────────────────────────────────────

PLUGIN_MODULE = '''
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path(__file__).with_name("plugin_instruments.yaml")


def materialize(source, target, refresh_remote_cache):
    target.mkdir(parents=True, exist_ok=True)
    (target / "answers.pipe").write_text("TLX1|demo_q1|demo_q2\\n7|2|4\\n9|3|5\\n", encoding="utf-8")
    return target, "plugin:v1"


def load_pipe(path):
    return pd.read_csv(path, sep="|")
'''

PLUGIN_REGISTRY = {"instruments": {"demo_scale": {
    "label": "Plugin demo scale", "citation": "Test fixture", "response_codings": [[1, 5]],
    "items": [{"id": "demo_q1", "key": 1}, {"id": "demo_q2", "key": 1}],
    "scores": {"usability_demo": {"dv": "usability", "column": "plugin_demo_score",
                                  "items": ["demo_q1", "demo_q2"], "centering": "none", "aggregate": "mean"}},
}}}


@pytest.fixture()
def fake_plugin(tmp_path, monkeypatch):
    """An 'installed' plugin distribution: a module plus a .dist-info with entry points."""
    (tmp_path / "fake_opendv_plugin.py").write_text(PLUGIN_MODULE, encoding="utf-8")
    (tmp_path / "plugin_instruments.yaml").write_text(yaml.safe_dump(PLUGIN_REGISTRY), encoding="utf-8")
    dist = tmp_path / "fake_opendv_plugin-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-opendv-plugin\nVersion: 0.1\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[opendv.sources]\nfake_archive = fake_opendv_plugin:materialize\n\n"
        "[opendv.loaders]\npipe = fake_opendv_plugin:load_pipe\n\n"
        "[opendv.instruments]\ndemo = fake_opendv_plugin:REGISTRY_PATH\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plugins.reset()
    yield tmp_path
    plugins.reset()
    sys.modules.pop("fake_opendv_plugin", None)


def test_plugins_are_discovered_and_used(fake_plugin, tmp_path):
    loaded = opendv.load_plugins(force=True)
    assert loaded["opendv.sources"] == ["fake_archive"]
    assert loaded["opendv.loaders"] == [".pipe"]
    assert loaded["opendv.instruments"] == ["demo"]

    # the plugin instrument is scored next to the built-in ones
    scored = opendv.score(pd.DataFrame({"demo_q1": [2], "demo_q2": [4]}))
    assert scored["plugin_demo_score"].iloc[0] == pytest.approx(3.0)

    # the plugin source type and file format work in a batch run
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": [
        {"source_id": "plug", "source_type": "fake_archive", "location": "anything"}]}), encoding="utf-8")
    summary = opendv.run_batch(manifest, tmp_path / "out", llm_deduction_enabled=False,
                               cache_root=tmp_path / "cache", fair_export=False)
    assert summary["successful_sources"] == 1, summary["results"]
    meta = pd.read_csv(tmp_path / "out" / "meta_view.csv")
    assert set(meta["commit"]) == {"plugin:v1"}
    assert "mental_demand" in set(meta["canonical_dv"])


def test_broken_plugin_is_skipped(tmp_path, monkeypatch, caplog):
    dist = tmp_path / "broken_plugin-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: broken-plugin\nVersion: 0.1\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text("[opendv.sources]\nbroken = no_such_module:materialize\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    plugins.reset()
    try:
        with caplog.at_level("WARNING"):
            loaded = opendv.load_plugins(force=True)
        assert "broken" not in loaded["opendv.sources"]
        assert "Skipping OpenDV plugin broken" in caplog.text
    finally:
        plugins.reset()
