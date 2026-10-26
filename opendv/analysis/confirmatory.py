"""Confirmatory test of the hierarchical shortlist (docs/preregistration_confirmatory_shortlist.md).

``fetch`` downloads exactly the files listed in ``analyses/confirmatory/sources.yaml``
into ``data/raw/confirmatory/<source_id>/`` and verifies each against its
published checksum (SHA-256 for OSF, MD5 for Zenodo, the git blob hash for
files pinned to a GitHub commit). ``manifest`` writes the batch manifest
(``local_path`` sources with the frozen gold-label mappings).

    python -m opendv.analysis.confirmatory fetch
    python -m opendv.analysis.confirmatory manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import yaml

from opendv._paths import REPO_ROOT

logger = logging.getLogger(__name__)

CONFIRMATORY_DIR = REPO_ROOT / "analyses" / "confirmatory"
SOURCES_PATH = CONFIRMATORY_DIR / "sources.yaml"
DATA_DIR = REPO_ROOT / "data" / "raw" / "confirmatory"
MANIFEST_PATH = CONFIRMATORY_DIR / "sources_manifest.yaml"


def load_sources(path: Path = SOURCES_PATH) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]


def checksum_ok(content: bytes, spec: dict[str, Any]) -> bool:
    """True when ``content`` matches every checksum and the size given in ``spec``."""
    if "size" in spec and len(content) != int(spec["size"]):
        return False
    if "sha256" in spec and hashlib.sha256(content).hexdigest() != spec["sha256"]:
        return False
    if "md5" in spec and hashlib.md5(content).hexdigest() != spec["md5"]:  # noqa: S324 - Zenodo publishes MD5
        return False
    if "git_blob" in spec:
        blob = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()  # noqa: S324 - git object id
        if blob != spec["git_blob"]:
            return False
    return True


def fetch(data_dir: Path = DATA_DIR, sources_path: Path = SOURCES_PATH) -> dict[str, list[str]]:
    """Download and verify every listed file; return ``{source_id: [problems]}``."""
    problems: dict[str, list[str]] = {}
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for source in load_sources(sources_path):
            target_dir = data_dir / source["source_id"]
            issues = problems.setdefault(source["source_id"], [])
            for spec in source["files"]:
                target = target_dir / spec["path"]
                if target.is_file() and checksum_ok(target.read_bytes(), spec):
                    continue
                response = client.get(spec["url"])
                if response.status_code != 200:
                    issues.append(f"{spec['path']}: HTTP {response.status_code}")
                    continue
                if not checksum_ok(response.content, spec):
                    issues.append(f"{spec['path']}: checksum or size mismatch")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                logger.info("%s: %s (%d bytes)", source["source_id"], spec["path"], len(response.content))
    return problems


def write_manifest(path: Path = MANIFEST_PATH, data_dir: Path = DATA_DIR) -> Path:
    """Batch manifest over the downloaded files, with each source's gold-label mapping."""
    sources = []
    for source in load_sources():
        entry = {
            "source_id": source["source_id"],
            "source_type": "local_path",
            "location": str((data_dir / source["source_id"]).relative_to(REPO_ROOT).as_posix()),
            "publication_doi": source["doi"],
            "license": source["license"],
        }
        mapping = CONFIRMATORY_DIR / "mappings" / f"{source['source_id']}_mapping.yaml"
        if mapping.is_file():
            entry["mapping_path"] = mapping.relative_to(REPO_ROOT).as_posix()
            # The gold labels win over schema aliases (a schema alias may be wrong for these data).
            entry["override_standard_aliases"] = True
        sources.append(entry)
    header = "# Generated from sources.yaml by `python -m opendv.analysis.confirmatory manifest`.\n"
    body = yaml.safe_dump({"sources": sources}, sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    return path


LABELS_PATH = CONFIRMATORY_DIR / "gold_labels.csv"
LABELLED_KINDS = ("dv", "non_dv")


def write_mappings(
    labels_path: Path = LABELS_PATH, out_dir: Path = CONFIRMATORY_DIR / "mappings"
) -> dict[str, list[str]]:
    """Mapping YAML per source from the frozen gold labels; returns the column names left out as conflicts.

    Mapping files are flat per source, so a column name labelled differently in
    two tables of one source cannot be labelled and is left out.
    """
    import pandas as pd

    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    labelled = labels[labels["kind"].isin(LABELLED_KINDS)]
    out_dir.mkdir(parents=True, exist_ok=True)
    conflicts: dict[str, list[str]] = {}
    for source_id, group in labelled.groupby("source_id", sort=True):
        targets = group.groupby("column", sort=False)["label"].agg(lambda s: sorted(set(s)))
        clash = sorted(column for column, values in targets.items() if len(values) > 1)
        conflicts[source_id] = clash
        mapping = {column: values[0] for column, values in targets.items() if len(values) == 1}
        header = (
            f"# Gold labels of the confirmatory shortlist test for {source_id}, generated from\n"
            "# analyses/confirmatory/gold_labels.csv (docs/preregistration_confirmatory_shortlist.md).\n"
        )
        body = yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True, width=1000)
        (out_dir / f"{source_id}_mapping.yaml").write_text(header + body, encoding="utf-8")
    return conflicts


def _mcnemar(new_correct, base_correct) -> dict[str, Any]:
    from scipy.stats import binomtest

    better = int((new_correct & ~base_correct).sum())
    worse = int((~new_correct & base_correct).sum())
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    return {"new_only": better, "baseline_only": worse, "p_two_sided": float(p)}


def analyze(run_dirs: dict[str, Path], baseline: str = "definition", primary: str = "dense_h") -> dict[str, Any]:
    """Pre-registered analysis of the E1 runs, one output directory per shortlist."""
    import pandas as pd

    key = ["source_id", "dataset_id", "alias"]
    items, summaries = {}, {}
    for name, run_dir in run_dirs.items():
        frame = pd.read_csv(run_dir / "llm_mapping_evaluation_items.csv", low_memory=False)
        items[name] = frame[frame["experiment"] == "E1"].set_index(key).sort_index()
        summaries[name] = json.loads((run_dir / "llm_mapping_evaluation_summary.json").read_text(encoding="utf-8"))
    base = items[baseline]
    positives = base.index[base["label_kind"] == "dv"]

    def correct(frame):
        pos = frame.loc[positives]
        return (pos["final_checked"] == pos["gold"]).to_numpy()

    result: dict[str, Any] = {"n_dv_columns": int(len(positives)), "shortlists": {}, "tests": {}}
    for name, frame in items.items():
        if not frame.index.equals(base.index):
            raise ValueError(f"{name}: E1 items differ from {baseline}")
        pos = frame.loc[positives]
        reached = pos["pre_llm_rejection"].isna()
        e1 = summaries[name]["results"]["E1"]
        result["shortlists"][name] = {
            "correct": int(correct(frame).sum()),
            "gold_in_shortlist": int(pos["gold_in_shortlist"].astype(bool).sum()),
            "precision_of_accepted_dv_mappings": e1["checked"]["precision_of_accepted_dv_mappings"],
            "abstained_on_dv_columns": e1["checked"]["abstained_on_dv_columns"],
            "curated_non_dv_mapped_to_a_dv": e1["checked"].get("curated_non_dv_mapped_to_a_dv"),
            "true_mappings_lost_to_check_among_reached": int(
                ((pos["final_unchecked"] == pos["gold"]) & pos["final_checked"].isna() & reached).sum()
            ),
            "screened_before_llm": int((~reached).sum()),
            "correct_by_source": {
                str(s): int(v) for s, v in
                (pos["final_checked"] == pos["gold"]).groupby(level="source_id").sum().items()
            },
        }
    base_correct = correct(base)
    others = [name for name in items if name != baseline]
    for name in others:
        result["tests"][f"{name}_vs_{baseline}"] = _mcnemar(correct(items[name]), base_correct)
    # Holm over the comparisons with the baseline (H1 primary, H2 secondary).
    ordered = sorted(result["tests"].items(), key=lambda kv: kv[1]["p_two_sided"])
    running = 0.0
    for rank, (_, test) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * test["p_two_sided"]))
        test["p_holm"] = running
    h1 = result["tests"][f"{primary}_vs_{baseline}"]
    prec = {
        name: (stats["precision_of_accepted_dv_mappings"] or {}).get("rate")
        for name, stats in result["shortlists"].items()
    }
    result["decision"] = {
        "more_correct": result["shortlists"][primary]["correct"] > result["shortlists"][baseline]["correct"],
        "p_below_0.05": h1["p_two_sided"] < 0.05,
        "precision_at_least_baseline": (prec[primary] or 0) >= (prec[baseline] or 0),
    }
    result["decision"]["switch_default_to_hierarchical"] = all(result["decision"].values())
    return result


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "manifest", "mappings", "analyze"])
    parser.add_argument("--runs", nargs="*", default=[], metavar="NAME=DIR",
                        help="analyze: E1 output directories, e.g. definition=out/def dense_h=out/dh mixed_h=out/mh")
    parser.add_argument("--output", default=str(CONFIRMATORY_DIR / "results.json"))
    args = parser.parse_args(argv)
    if args.command == "fetch":
        problems = {k: v for k, v in fetch().items() if v}
        for source_id, issues in problems.items():
            for issue in issues:
                print(f"[DROP?] {source_id}: {issue}")
        print("all files verified" if not problems else f"{len(problems)} source(s) with problems")
    elif args.command == "manifest":
        print(write_manifest())
    elif args.command == "mappings":
        conflicts = write_mappings()
        print(json.dumps({k: v for k, v in conflicts.items() if v}, indent=2) or "no conflicts")
    else:
        runs = {name: Path(path) for name, path in (item.split("=", 1) for item in args.runs)}
        result = analyze(runs)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
