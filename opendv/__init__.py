"""OpenDV-HCI: schema-driven standardization and meta-analysis of dependent variables in HCI research.

Public API (stable across 3.x; everything else is internal):

* :func:`standardize` - map one table's columns to canonical DV names;
* :func:`run_batch` - fetch and standardize every source of a manifest;
* :func:`analyze` - cross-study overlap, meta-analysis, condition contrasts,
  measurement invariance and causal discovery on standardized studies;
* :func:`review_queue` - LLM-deduced and unmapped columns for human review;
* :func:`score` - questionnaire scores from the instrument registry;
* :func:`load_plugins` - register sources, file formats and instruments
  contributed by installed packages (see :mod:`opendv.plugins`).

Command line: ``opendv --help``. The modules formerly under ``scripts/`` and
``analyses/`` live in this package (``opendv.<module>``, ``opendv.analysis.<module>``);
the old import paths still work in a source checkout.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__version__ = "3.1.0"


def standardize(
    data: pd.DataFrame | str | Path,
    mapping: Mapping[str, str] | None = None,
    schema_path: str | Path | None = None,
) -> pd.DataFrame:
    """Rename a table's columns to canonical DV ids.

    ``data`` is a DataFrame or a path to any supported file format. Columns
    are matched case-insensitively against ``mapping`` (raw name -> DV id) when
    given, else against the aliases of the standard DV schema (or the schema at
    ``schema_path``). Unmatched columns keep their names.
    """
    from opendv._paths import SCHEMAS_DIR
    from opendv.convert_dv import load_schema, standardize_columns
    from opendv.data_loaders import load_input_file

    frame = load_input_file(str(data)) if isinstance(data, (str, Path)) else data
    if mapping is None:
        schema = schema_path or SCHEMAS_DIR / "standard_dv_mapping.yaml"
        mapping = load_schema(str(schema))["mapping"]
    return standardize_columns(frame, dict(mapping))


def run_batch(manifest: str | Path, output_dir: str | Path, **options: Any) -> dict[str, Any]:
    """Standardize every source of a manifest; returns the run summary.

    Options are those of ``opendv.run_batch_standardization.run_batch``
    (e.g. ``workers``, ``llm_deduction_enabled``, ``cache_root``,
    ``output_format``). Use ``schema_path`` for a custom DV schema.
    """
    from opendv._paths import SCHEMAS_DIR
    from opendv.run_batch_standardization import run_batch as _run_batch

    schema_path = Path(options.pop("schema_path", SCHEMAS_DIR / "standard_dv_mapping.yaml"))
    if options.get("cache_root") is not None:
        options["cache_root"] = Path(options["cache_root"])
    return _run_batch(Path(manifest), Path(output_dir), schema_path, **options)


def analyze(input_dir: str | Path, output_dir: str | Path, *, exclude_llm_deduced: bool = True,
            estimator: str = "DL", extra_args: list[str] | None = None) -> Path:
    """Run the multi-study analysis on a folder of standardized studies.

    Writes every output (``meta_analysis_summary.csv``, ``contrast_meta_analysis.csv``,
    ``measurement_invariance.csv``, figures, ...) to ``output_dir`` and returns it.
    ``extra_args`` are passed to the command line (``opendv analyze --help``).
    """
    from opendv.analysis.multi_study_analysis import main

    argv = ["--input-dir", str(input_dir), "--output-dir", str(output_dir), "--estimator", estimator]
    if exclude_llm_deduced:
        argv.append("--exclude-llm-deduced")
    main([*argv, *(extra_args or [])])
    return Path(output_dir)


def review_queue(batch_dir: str | Path) -> pd.DataFrame:
    """Columns of a batch run that need a human decision, most urgent first."""
    from opendv.mapping_review import build_review_queue

    return build_review_queue(Path(batch_dir))


def score(data: pd.DataFrame) -> pd.DataFrame:
    """Add every registered questionnaire score whose items are all present."""
    from opendv.instruments import score_frame

    return score_frame(data)[0]


def load_plugins(force: bool = False) -> dict[str, list[str]]:
    """Register plugin sources, file loaders and instruments; returns what was loaded."""
    from opendv.plugins import load_plugins as _load_plugins

    return _load_plugins(force=force)


__all__ = ["__version__", "analyze", "load_plugins", "review_queue", "run_batch", "score", "standardize"]
