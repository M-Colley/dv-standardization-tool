"""Inventory of the LLM-mapped columns that feed a pooled result.

Reads one batch run (its ``meta_view.csv`` and ``standardized/`` tables) and
lists every standardized column whose mapping category is ``llm_deduced``
and that the all-mappings analysis keeps, i.e. that
``load_studies(..., llm_columns="prefer_curated")`` does not set aside. The
decision is made with the analysis's own functions
(:func:`opendv.analysis.multi_study_analysis.load_column_provenance`,
``_file_column_categories`` and ``_llm_columns_to_drop``), so the list is the
one the analysis used, not a re-implementation of its rule.

Each column is traced back to its raw column through the table's meta_view
rows, which the batch writes in column order. Columns behind an ``llm``-level
finding of the cross-study relations (``relations_by_study.csv`` rows whose
construct reached the ``llm`` level) are flagged.

Usage::

    python -I analyses/llm_mapping_audit/inventory.py --run-dir data/processed/lock39 \
        --relations-dir analyses/output_relations --output analyses/llm_mapping_audit/inventory.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opendv.analysis import multi_study_analysis as msa
from opendv.analysis.construct_relations import LEVEL_RANK

# relations_by_study.csv gives the highest level a construct needed as a rank.
LLM_RANK = LEVEL_RANK["llm"]


def _standardized_tables(standardized: Path) -> list[tuple[str, str, Path]]:
    """``(study, dataset_id, path)`` of every standardized table of the run."""
    out = []
    for study_dir in sorted(p for p in standardized.iterdir() if p.is_dir()):
        ids = msa.dataset_ids_by_file(study_dir)
        for path in sorted(study_dir.iterdir()):
            if path.suffix.lower() not in {".csv", ".tsv", ".xlsx", ".parquet"}:
                continue
            out.append((study_dir.name, ids.get(path.name, path.name), path))
    return out


def build_inventory(run_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(run_dir / "meta_view.csv", dtype=str, keep_default_na=False)
    rows_by_table = {key: frame for key, frame in meta.groupby(["source_id", "dataset_id"], sort=False)}
    provenance = msa.load_column_provenance(run_dir / "meta_view.csv")
    derived_items = msa._derived_scale_item_names()

    tables_by_study: dict[str, list[tuple[str, list[str], dict[str, str]]]] = {}
    for study, dataset_id, path in _standardized_tables(run_dir / "standardized"):
        columns = [c for c in msa._read_header(path) if c != msa.SOURCE_FILE_COLUMN]
        categories = msa._file_column_categories(columns, provenance.get((study, dataset_id), []))
        tables_by_study.setdefault(study, []).append((dataset_id, columns, categories))

    records = []
    for study, tables in tables_by_study.items():
        cats = [categories for _, _, categories in tables]
        llm_all = msa._llm_columns_to_drop(cats, "exclude", derived_items)
        set_aside = msa._llm_columns_to_drop(cats, "prefer_curated", derived_items)
        for (dataset_id, columns, _categories), llm, aside in zip(tables, llm_all, set_aside, strict=True):
            meta_rows = rows_by_table.get((study, dataset_id))
            aligned = meta_rows is not None and len(meta_rows) == len(columns)
            for position, column in enumerate(columns):
                if column not in llm or column in aside:
                    continue
                row = meta_rows.iloc[position] if aligned else None
                records.append({
                    "study": study,
                    "file": dataset_id,
                    "standardized_column": column,
                    "column": row["original_alias"] if row is not None else "",
                    "llm_dv": msa._analysis_dv_key(column, derived_items),
                    "meta_canonical_dv": row["canonical_dv"] if row is not None else "",
                    "item_of": row["item_of"] if row is not None else "",
                    "mapping_confidence": row["mapping_confidence"] if row is not None else "",
                    "n": row["n"] if row is not None else "",
                    "mean": row["mean"] if row is not None else "",
                    "min": row["min"] if row is not None else "",
                    "max": row["max"] if row is not None else "",
                })
    columns = ["study", "file", "standardized_column", "column", "llm_dv", "meta_canonical_dv",
               "item_of", "mapping_confidence", "n", "mean", "min", "max"]
    return pd.DataFrame.from_records(records, columns=columns)


def relation_flags(relations_dir: Path) -> pd.DataFrame:
    """``(study, dv)`` pairs whose construct reached the ``llm`` level in a
    pooled ``llm``-level finding of the relations."""
    pooled = pd.read_csv(relations_dir / "relations_pooled.csv")
    by_study = pd.read_csv(relations_dir / "relations_by_study.csv")
    findings = pooled[(pooled["level"] == "llm") & pooled["finding"].astype(bool) & pooled["includes_llm_columns"].astype(bool)]
    keys = findings[["association", "dv_x", "dv_y"]]
    rows = by_study[by_study["level"] == "llm"].merge(keys, on=["association", "dv_x", "dv_y"])
    out = []
    for _, r in rows.iterrows():
        label = f"{r['association']}: {r['dv_x']} x {r['dv_y']}"
        if r["max_level_x"] == LLM_RANK:
            out.append({"study": r["study"], "llm_dv": r["dv_x"], "relations_finding": label})
        if r["max_level_y"] == LLM_RANK:
            out.append({"study": r["study"], "llm_dv": r["dv_y"], "relations_finding": label})
    if not out:
        return pd.DataFrame(columns=["study", "llm_dv", "relations_finding"])
    flags = pd.DataFrame(out)
    return flags.groupby(["study", "llm_dv"], as_index=False)["relations_finding"].agg(
        lambda s: "; ".join(sorted(set(s)))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--relations-dir", type=Path, default=Path("analyses/output_relations"))
    parser.add_argument("--output", type=Path, default=Path("analyses/llm_mapping_audit/inventory.csv"))
    args = parser.parse_args()
    inventory = build_inventory(args.run_dir)
    flags = relation_flags(args.relations_dir)
    inventory = inventory.merge(flags, on=["study", "llm_dv"], how="left")
    inventory["relations_finding"] = inventory["relations_finding"].fillna("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output, index=False)
    print(f"{len(inventory)} pooled LLM-mapped columns in {inventory['study'].nunique()} studies -> {args.output}")
    print(inventory.groupby("study").size().to_string())
    print(f"behind an llm-level relations finding: {(inventory['relations_finding'] != '').sum()}")


if __name__ == "__main__":
    main()
