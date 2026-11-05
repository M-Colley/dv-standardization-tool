"""Confirmatory test of the hierarchical shortlist (docs/preregistration_confirmatory_shortlist.md).

``fetch`` downloads exactly the files listed in ``analyses/confirmatory/sources.yaml``
into ``data/raw/confirmatory/<source_id>/`` and verifies each against its
published checksum (SHA-256 for OSF, MD5 for Zenodo, the git blob hash for
files pinned to a GitHub commit). ``manifest`` writes the batch manifest
(``local_path`` sources with the frozen gold-label mappings).
``posthoc`` re-analyses E1 runs made after the pre-registered test, with
header matching applied first; its results are not part of that test.

    python -m opendv.analysis.confirmatory fetch
    python -m opendv.analysis.confirmatory manifest
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import zipfile
import zlib
from pathlib import Path, PurePosixPath
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


def member_ok(content: bytes, member: dict[str, Any]) -> bool:
    """True when an archive member matches its pinned size and CRC-32."""
    return len(content) == int(member["size"]) and f"{zlib.crc32(content):08x}" == member["crc32"]


def _member_target(target_dir: Path, archive: str, member: str) -> Path | None:
    """Where a member is written (``<archive stem>/<member path>``); ``None`` if it would leave that folder."""
    root = (target_dir / PurePosixPath(archive).stem).resolve()
    target = (root / member).resolve()
    return target if target.is_relative_to(root) else None


RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
RETRIES = 6


def _wait(attempt: int, retry_after: str | None = None) -> None:
    """Back off before a retry: the server's Retry-After if given, else 10, 20, 40, ... s."""
    import time

    seconds = float(retry_after) if retry_after and retry_after.isdigit() else 10.0 * 2**attempt
    time.sleep(min(seconds, 300.0))


def _download(client: httpx.Client, url: str) -> tuple[int, bytes]:
    """(status, body) of a GET, streamed into memory; rate limits and server errors are retried."""
    status = 0
    for attempt in range(RETRIES):
        with client.stream("GET", url) as response:
            status = response.status_code
            if status == 200:
                buffer = io.BytesIO()
                for chunk in response.iter_bytes(1 << 20):
                    buffer.write(chunk)
                return 200, buffer.getvalue()
            retry_after = response.headers.get("retry-after")
        if status not in RETRY_STATUS:
            break
        if attempt + 1 < RETRIES:
            _wait(attempt, retry_after)
    return status, b""


def _range_session():
    """A requests session for remotezip that retries rate limits and server errors."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=RETRIES, backoff_factor=10, status_forcelist=sorted(RETRY_STATUS),
                  allowed_methods=["GET", "HEAD"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_members(client: httpx.Client, spec: dict[str, Any], target_dir: Path, issues: list[str]) -> None:
    """Write the pinned members of a zip archive, read in memory (never extracted wholesale).

    ``fetch: whole`` downloads the archive and checks it against its published checksum;
    ``fetch: range`` (archives over 1 GB) reads only the members by HTTP range requests.
    Each member is checked against its pinned size and CRC-32 either way.
    """
    targets = {m["path"]: _member_target(target_dir, spec["path"], m["path"]) for m in spec["members"]}
    pending = [m for m in spec["members"]
               if not (targets[m["path"]] and targets[m["path"]].is_file()
                       and member_ok(targets[m["path"]].read_bytes(), m))]
    if not pending:
        return
    if spec.get("fetch") == "range":
        from remotezip import RemoteZip  # optional dependency: pip install opendv-hci[fetch]

        archive: zipfile.ZipFile = RemoteZip(spec["url"], support_suffix_range=False, timeout=120,
                                             session=_range_session())
    else:
        status, content = _download(client, spec["url"])
        if status != 200:
            issues.append(f"{spec['path']}: HTTP {status}")
            return
        if not checksum_ok(content, spec):
            issues.append(f"{spec['path']}: checksum or size mismatch")
            return
        archive = zipfile.ZipFile(io.BytesIO(content))
    with archive:
        for member in pending:
            target = targets[member["path"]]
            if target is None:
                issues.append(f"{spec['path']}!{member['path']}: path leaves the source folder")
                continue
            if archive.getinfo(member["path"]).file_size != int(member["size"]):
                issues.append(f"{spec['path']}!{member['path']}: size differs from the pinned size")
                continue
            content = archive.read(member["path"])  # zipfile checks the CRC-32 while reading
            if not member_ok(content, member):
                issues.append(f"{spec['path']}!{member['path']}: CRC-32 or size mismatch")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def fetch(data_dir: Path = DATA_DIR, sources_path: Path = SOURCES_PATH) -> dict[str, list[str]]:
    """Download and verify every listed file; return ``{source_id: [problems]}``.

    Sources marked ``dropped`` are skipped. A file with ``members`` is a zip archive of
    which only those members are written, under ``<source_id>/<archive stem>/``.
    """
    problems: dict[str, list[str]] = {}
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for source in load_sources(sources_path):
            if source.get("dropped"):
                continue
            target_dir = data_dir / source["source_id"]
            issues = problems.setdefault(source["source_id"], [])
            for spec in source["files"]:
                try:
                    if spec.get("members"):
                        _fetch_members(client, spec, target_dir, issues)
                        continue
                    target = target_dir / spec["path"]
                    if target.is_file() and checksum_ok(target.read_bytes(), spec):
                        continue
                    status, content = _download(client, spec["url"])
                except Exception as error:  # a file that fails is reported, and the others are still fetched
                    issues.append(f"{spec['path']}: {type(error).__name__}: {error}"[:300])
                    continue
                if status != 200:
                    issues.append(f"{spec['path']}: HTTP {status}")
                    continue
                if not checksum_ok(content, spec):
                    issues.append(f"{spec['path']}: checksum or size mismatch")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                logger.info("%s: %s (%d bytes)", source["source_id"], spec["path"], len(content))
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


def _e1_items(run_dir: Path):
    import pandas as pd

    frame = pd.read_csv(run_dir / "llm_mapping_evaluation_items.csv", low_memory=False)
    return frame[frame["experiment"] == "E1"].set_index(["source_id", "dataset_id", "alias"]).sort_index()


def header_match_answers(items, data_dir: Path = DATA_DIR) -> dict[tuple[str, str, str], str]:
    """DV that header matching gives each E1 item, as for a study without a mapping file.

    A table's columns that no standard alias maps go through
    :func:`opendv.header_matching.match_headers` with the standard aliases only,
    so neither the gold labels nor the study's mapping file are used.
    """
    from opendv._paths import SCHEMAS_DIR
    from opendv.header_matching import match_headers
    from opendv.run_batch_standardization import _load_any_table, load_schema

    aliases = load_schema(str(SCHEMAS_DIR / "standard_dv_mapping.yaml"))["mapping"]
    known = {str(alias).lower() for alias in aliases}
    answers: dict[tuple[str, str, str], str] = {}
    for (source_id, dataset_id), group in items.reset_index().groupby(["source_id", "dataset_id"]):
        columns = [str(c) for c in _load_any_table(data_dir / source_id / dataset_id).columns]
        matches = match_headers(columns, [c for c in columns if c.lower() not in known], aliases)
        for alias in group["alias"].astype(str):
            if alias in matches:
                answers[(source_id, dataset_id, alias)] = matches[alias].dv
    return answers


def posthoc(
    run_dirs: dict[str, Path],
    preregistered_dirs: dict[str, Path],
    header_answers: dict[tuple[str, str, str], str] | None = None,
    baseline: str = "definition",
    dv_ids: set[str] | None = None,
) -> dict[str, Any]:
    """E1 runs after the pre-registered test: the pre-registered analysis, the
    change against the pre-registered run of each shortlist, and the accuracy
    when header matching answers first and the LLM answers the rest.

    The combined rates come from the evaluator's own ``summarize`` over the
    schema's DV ids, so they are defined as the LLM-only rates are."""
    import pandas as pd

    from opendv._paths import SCHEMAS_DIR
    from opendv.analysis.evaluate_llm_mapping import summarize
    from opendv.run_batch_standardization import SchemaFamilies

    result = analyze(run_dirs, baseline=baseline)
    items = {name: _e1_items(run_dir) for name, run_dir in run_dirs.items()}
    if header_answers is None:
        header_answers = header_match_answers(items[baseline])
    if dv_ids is None:
        families = SchemaFamilies.load(SCHEMAS_DIR / "standard_dv_mapping.yaml")
        dv_ids = {canonical for canonical, domain in families.canonical_domain_lookup.items() if domain == "dv"}
    header = pd.Series(header_answers, dtype=object).reindex(items[baseline].index)
    combined_correct = {}
    for name, frame in items.items():
        pos = frame["label_kind"] == "dv"
        llm_correct = (frame["final_checked"] == frame["gold"])[pos]
        stats = result["shortlists"][name]
        before = _e1_items(preregistered_dirs[name])
        common = llm_correct.index.intersection(before.index[before["label_kind"] == "dv"])
        before_correct = (before["final_checked"] == before["gold"]).loc[common]
        stats["vs_preregistered_run"] = {
            "dv_columns_in_both": int(len(common)),
            "correct_before": int(before_correct.sum()),
            "correct_now": int(llm_correct.loc[common].sum()),
            **_mcnemar(llm_correct.loc[common].to_numpy(), before_correct.to_numpy()),
        }
        answer = header.combine_first(frame["final_checked"])
        combined_correct[name] = (answer == frame["gold"])[pos]
        rates = summarize(frame.assign(final_checked=answer).reset_index(), dv_ids)["E1"]["checked"]
        stats["with_header_matching"] = {
            "correct": int(combined_correct[name].sum()),
            "header_matched_dv_columns": int(header[pos].notna().sum()),
            "header_matched_correct": int((header == frame["gold"])[pos].sum()),
            "header_matched_non_dv_columns": int(header[~pos].notna().sum()),
            "precision_of_accepted_dv_mappings": rates["precision_of_accepted_dv_mappings"],
            "abstained_on_dv_columns": rates["abstained_on_dv_columns"],
            "curated_non_dv_mapped_to_a_dv": rates.get("curated_non_dv_mapped_to_a_dv"),
            **{f"vs_llm_only_{k}": v for k, v in _mcnemar(
                combined_correct[name].to_numpy(), llm_correct.to_numpy()).items()},
        }
    return result


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "manifest", "mappings", "analyze", "posthoc"])
    parser.add_argument("--runs", nargs="*", default=[], metavar="NAME=DIR",
                        help="analyze: E1 output directories, e.g. definition=out/def dense_h=out/dh mixed_h=out/mh")
    parser.add_argument("--preregistered-runs", nargs="*", default=[], metavar="NAME=DIR",
                        help="posthoc: the pre-registered E1 output directories, by the same names")
    parser.add_argument("--output", default=str(CONFIRMATORY_DIR / "results.json"))
    parser.add_argument("--sources", default=str(SOURCES_PATH),
                        help="fetch: list of pinned files (also used for analyses/development_sources/sources.yaml)")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="fetch: where <source_id>/ folders are written")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        problems = {k: v for k, v in fetch(Path(args.data_dir), Path(args.sources)).items() if v}
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
        if args.command == "posthoc":
            before = {name: Path(path) for name, path in (item.split("=", 1) for item in args.preregistered_runs)}
            result = posthoc(runs, before)
        else:
            result = analyze(runs)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
