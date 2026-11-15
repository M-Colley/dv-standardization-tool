#!/usr/bin/env python3
"""Schema stability of the LLM's column-to-DV mappings.

Schema 3.3 added one DV (``touch_offset``) and changed 19 LLM mappings in five
studies whose curated mappings did not change. This module measures how often
an LLM answer changes when the DV schema changes in ways unrelated to the
column, and compares conditions that might reduce it. The design is fixed in
``docs/schema_stability_design.md``.

It is a thin layer over the leave-alias-out benchmark
(:mod:`opendv.analysis.evaluate_llm_mapping`): every perturbation writes a DV
schema file and reruns the benchmark's E1 step (``run_e1``) on a fixed item
list with ``--schema`` pointing at it. Answers to prompts that repeat exactly
are taken from a response cache (decoding is greedy, or sampled with a fixed
seed); the ``identity`` perturbation reruns the reference without the cache to
check that a repeated prompt gets the same answer.

Subcommands::

    python -m opendv.analysis.schema_stability plan     # draws -> design.json
    python -m opendv.analysis.schema_stability dry-run  # prompts only, no LLM
    python -m opendv.analysis.schema_stability run --condition e2b_definition ...
    python -m opendv.analysis.schema_stability analyze
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import logging
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from opendv._paths import SCHEMAS_DIR

logger = logging.getLogger("schema_stability")

MASTER_SEED = 20261115
BENCHMARK_SEED = 20261017  # the committed benchmark's item seed
BASE_SCHEMA = SCHEMAS_DIR / "standard_dv_mapping.yaml"
DESIGN_DIR = Path("analyses/schema_stability")
OUTPUT_DIR = Path("analyses/output_llm_evaluation/schema_stability")
WORK_DIR = Path("data/processed/schema_stability")

EDIT_FAMILIES = ("add", "remove", "reword")
K_LEVELS = (1, 5)
# Draws per perturbation set (pre-specified in docs/schema_stability_design.md).
DEFAULT_DRAWS = {"edit": 5, "schema_order": 2, "candidate_order": 3, "sample_seed": 2}
ITEM_KEY = ["half", "source_id", "dataset_id", "alias"]
ANALYSIS_KINDS = ("dv", "curated_non_dv")  # blocklisted columns never reach the LLM in the pipeline


# ── Perturbations ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Perturbation:
    """One change of the DV schema or of how the prompt lists the shortlist."""

    name: str
    family: str  # base, identity, add, remove, reword, schema_order, candidate_order, sample_seed, real_3_2
    k: int = 0
    draw: int = 0
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    reworded: tuple[str, ...] = ()
    permutation_seed: int | None = None  # schema_order: seed of the permutation of the DV list
    shuffle_seed: int | None = None  # candidate_order: seed of the prompt's shortlist order
    sample_seed_offset: int = 0  # sample_seed: added to the condition's sampling seed
    use_cache: bool = True  # identity: False, every prompt is generated again

    @property
    def edits_schema(self) -> bool:
        return bool(self.added or self.removed or self.reworded or self.permutation_seed is not None)

    @property
    def schema_name(self) -> str:
        return self.name if self.edits_schema else "base"


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng([MASTER_SEED, *keys])


def plan_perturbations(
    unrelated: list[str],
    add_pool: list[str],
    rewordable: list[str],
    draws: dict[str, int] | None = None,
    k_levels: Iterable[int] = K_LEVELS,
) -> list[Perturbation]:
    """Every perturbation of the design, in run order.

    ``unrelated`` are the schema DVs a perturbation may remove (never a gold
    label, never named by a curated mapping), ``add_pool`` the ids of the DVs it
    may add, ``rewordable`` the DVs with a fixed rewording (a subset of
    ``unrelated``). Each set draws without replacement from its own seeded
    generator, so adding draws or k levels never changes the existing ones.
    """
    draws = {**DEFAULT_DRAWS, **(draws or {})}
    plan = [Perturbation("base", "base"), Perturbation("identity", "identity", use_cache=False)]
    sources = {"add": sorted(add_pool), "remove": list(unrelated), "reword": [d for d in unrelated if d in rewordable]}
    for family_index, family in enumerate(EDIT_FAMILIES, start=1):
        for k in k_levels:
            for draw in range(1, draws["edit"] + 1):
                pool = sources[family]
                chosen = _rng(family_index, k, draw).choice(len(pool), size=k, replace=False)
                ids = tuple(sorted(pool[i] for i in chosen))
                plan.append(Perturbation(f"{family}_k{k}_d{draw}", family, k, draw, **{
                    "add": {"added": ids}, "remove": {"removed": ids}, "reword": {"reworded": ids},
                }[family]))
    for draw in range(1, draws["schema_order"] + 1):
        plan.append(Perturbation(f"schema_order_d{draw}", "schema_order", draw=draw, permutation_seed=draw))
    for draw in range(1, draws["candidate_order"] + 1):
        plan.append(Perturbation(f"candidate_order_d{draw}", "candidate_order", draw=draw, shuffle_seed=draw))
    for draw in range(1, draws["sample_seed"] + 1):
        plan.append(Perturbation(f"sample_seed_d{draw}", "sample_seed", draw=draw, sample_seed_offset=draw))
    plan.append(Perturbation("real_3_2", "real_3_2", k=1, removed=("touch_offset",)))
    return plan


def apply_perturbation(
    schema: dict[str, Any],
    perturbation: Perturbation,
    add_pool: dict[str, dict[str, Any]],
    rewordings: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """A perturbed copy of the schema (the input is not changed).

    Removed DVs leave the ``dvs`` list; reworded DVs get the rewording's
    ``label`` and ``notes`` (nothing else); an added DV is inserted after the
    last DV of its cluster, as a maintainer would add it (at the end if the
    cluster has none); ``permutation_seed`` then permutes the whole list.
    """
    out = copy.deepcopy(schema)
    dvs = [entry for entry in out.get("dvs") or [] if entry.get("id") not in set(perturbation.removed)]
    for entry in dvs:
        if entry.get("id") in perturbation.reworded:
            rewording = rewordings[entry["id"]]
            entry["label"] = rewording["label"]
            if "notes" in rewording:
                entry["notes"] = rewording["notes"]
    for dv_id in perturbation.added:
        entry = copy.deepcopy(add_pool[dv_id])
        same_cluster = [i for i, other in enumerate(dvs) if other.get("cluster") == entry.get("cluster")]
        dvs.insert(same_cluster[-1] + 1 if same_cluster else len(dvs), entry)
    if perturbation.permutation_seed is not None:
        order = _rng(99, perturbation.permutation_seed).permutation(len(dvs))
        dvs = [dvs[i] for i in order]
    out["dvs"] = dvs
    return out


def check_perturbation(
    schema: dict[str, Any], perturbation: Perturbation, protected: set[str]
) -> None:
    """Raise if a perturbation touches a protected DV (gold labels, curated targets) or an unknown id."""
    ids = {str(entry.get("id")) for entry in schema.get("dvs") or []}
    if perturbation.family == "real_3_2":
        return  # removes touch_offset, a gold DV: its items are left out of this perturbation
    touched = set(perturbation.removed) | set(perturbation.reworded)
    if touched & protected:
        raise ValueError(f"{perturbation.name} touches protected DVs {sorted(touched & protected)}")
    if touched - ids:
        raise ValueError(f"{perturbation.name} names DVs missing from the schema: {sorted(touched - ids)}")
    if set(perturbation.added) & ids:
        raise ValueError(f"{perturbation.name} adds DVs already in the schema: {sorted(set(perturbation.added) & ids)}")


def protected_dvs(gold: Iterable[str], mapping_files: Iterable[Path], dv_ids: set[str]) -> set[str]:
    """Gold labels of the benchmark items plus every DV id a curated mapping file names."""
    named: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and node in dv_ids:
            named.add(node)

    for path in mapping_files:
        walk(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    return {str(g) for g in gold if isinstance(g, str)} | named


def unrelated_dvs(schema: dict[str, Any], protected: set[str]) -> list[str]:
    """Schema DV ids, in schema order, that a perturbation may remove or reword."""
    return [str(e["id"]) for e in schema.get("dvs") or [] if isinstance(e, dict) and str(e.get("id")) not in protected]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def write_schema(schema: dict[str, Any], path: Path) -> str:
    """Write a schema file; returns the SHA-256 of its text."""
    text = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True, width=1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Conditions and the benchmark halves ──────────────────────────────────────


@dataclass(frozen=True)
class Condition:
    """How the LLM is asked; every condition runs the same perturbations."""

    name: str
    model: str = "google/gemma-4-E2B-it"
    shortlist: str = "definition"
    candidate_order: str = "rank"
    samples: int = 1
    temperature: float = 1.0
    sample_seed: int = MASTER_SEED

    def applies(self, perturbation: Perturbation) -> bool:
        if perturbation.family == "candidate_order":
            return self.candidate_order == "rank"
        if perturbation.family == "sample_seed":
            return self.samples > 1
        return True


CONDITIONS = {
    c.name: c
    for c in (
        Condition("e2b_definition"),
        Condition("e2b_dense_h", shortlist="dense_h"),
        Condition("e4b_definition", model="google/gemma-4-E4B-it"),
        Condition("e2b_definition_sc5", samples=5),
        Condition("e2b_definition_alpha", candidate_order="alphabetical"),
    )
}
# Planned paired comparisons (condition, reference); Holm over those that ran.
COMPARISONS = (
    ("e2b_dense_h", "e2b_definition"),
    ("e4b_definition", "e2b_definition"),
    ("e2b_definition_sc5", "e2b_definition"),
    ("e2b_definition_alpha", "e2b_definition"),
)


@dataclass(frozen=True)
class Half:
    """One of the benchmark's two runs without the LLM (its items and their sources)."""

    name: str
    batch_dir: Path
    manifest: Path
    cache_dir: Path


def default_halves(root: Path = Path("data/processed")) -> list[Half]:
    """The committed benchmark's runs without the LLM (``det23l``, ``det16l``; docs/reproducibility.md)."""
    return [
        Half("first_23", root / "det23l", root / "manifest23.yaml", root / "cache"),
        Half("added_16", root / "det16l", root / "manifest16.yaml", root / "cache16"),
    ]


def parse_half(spec: str) -> Half:
    """``NAME=BATCH_DIR,MANIFEST,CACHE_DIR``."""
    name, _, rest = spec.partition("=")
    parts = rest.split(",")
    if not name or len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--half expects NAME=BATCH_DIR,MANIFEST,CACHE_DIR, got {spec!r}")
    return Half(name, Path(parts[0]), Path(parts[1]), Path(parts[2]))


def benchmark_items(half: Half, dv_ids: set[str]) -> pd.DataFrame:
    """The committed benchmark's E1 items of one half (``--max-per-source 1000``, its seed)."""
    from opendv.analysis.evaluate_llm_mapping import gold_items

    meta_view = pd.read_csv(half.batch_dir / "meta_view.csv", low_memory=False)
    return gold_items(meta_view, dv_ids, 1000, BENCHMARK_SEED)


def base_dv_ids(schema_path: Path = BASE_SCHEMA) -> set[str]:
    from opendv.run_batch_standardization import SchemaFamilies

    families = SchemaFamilies.load(schema_path)
    return {c for c, domain in families.canonical_domain_lookup.items() if domain == "dv"}


def curated_mapping_files(halves: Iterable[Half]) -> list[Path]:
    files: set[Path] = set()
    for half in halves:
        for source in load_yaml(half.manifest).get("sources") or []:
            if source.get("mapping_path"):
                files.add(Path(str(source["mapping_path"])))
    return sorted(files)


# ── Design ───────────────────────────────────────────────────────────────────


def build_design(halves: list[Half], draws: dict[str, int] | None = None) -> dict[str, Any]:
    """The pre-specified perturbations with everything that defines them."""
    schema = load_yaml(BASE_SCHEMA)
    dv_ids = base_dv_ids()
    gold: set[str] = set()
    for half in halves:
        gold |= set(benchmark_items(half, dv_ids)["gold"].dropna())
    protected = protected_dvs(gold, curated_mapping_files(halves), dv_ids)
    unrelated = unrelated_dvs(schema, protected)
    add_pool = {e["id"]: e for e in load_yaml(DESIGN_DIR / "added_dv_pool.yaml")["dvs"]}
    rewordings = load_yaml(DESIGN_DIR / "reworded_definitions.yaml")["dvs"]
    if set(rewordings) != set(unrelated):
        raise ValueError(
            "the rewordings must cover exactly the unrelated DVs: "
            f"missing {sorted(set(unrelated) - set(rewordings))}, extra {sorted(set(rewordings) - set(unrelated))}"
        )
    plan = plan_perturbations(unrelated, list(add_pool), list(rewordings), draws)
    for perturbation in plan:
        check_perturbation(schema, perturbation, protected)
    return {
        "master_seed": MASTER_SEED,
        "base_schema_version": str(schema.get("version")),
        "n_schema_dvs": len(schema.get("dvs") or []),
        "gold_dvs": sorted(gold),
        "protected_dvs": sorted(protected),
        "unrelated_dvs": unrelated,
        "add_pool": sorted(add_pool),
        "draws": {**DEFAULT_DRAWS, **(draws or {})},
        "k_levels": list(K_LEVELS),
        "perturbations": [asdict(p) for p in plan],
    }


def perturbations_from_design(design: dict[str, Any]) -> list[Perturbation]:
    return [
        Perturbation(**{k: tuple(v) if isinstance(v, list) else v for k, v in record.items()})
        for record in design["perturbations"]
    ]


def schema_file(perturbation: Perturbation, work_dir: Path) -> Path:
    if not perturbation.edits_schema:
        return BASE_SCHEMA
    return work_dir / "schemas" / f"{perturbation.schema_name}.yaml"


def materialize_schema(perturbation: Perturbation, work_dir: Path) -> Path:
    path = schema_file(perturbation, work_dir)
    if path != BASE_SCHEMA:
        add_pool = {e["id"]: e for e in load_yaml(DESIGN_DIR / "added_dv_pool.yaml")["dvs"]}
        rewordings = load_yaml(DESIGN_DIR / "reworded_definitions.yaml")["dvs"]
        write_schema(apply_perturbation(load_yaml(BASE_SCHEMA), perturbation, add_pool, rewordings), path)
    return path


# ── Running ──────────────────────────────────────────────────────────────────


def load_response_cache(path: Path) -> dict[str, str | None]:
    cache: dict[str, str | None] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    cache[record["key"]] = record["answer"]
    return cache


def append_response_cache(path: Path, cache: dict[str, str | None], known: set[str]) -> None:
    new = [key for key in cache if key not in known]
    if not new:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for key in new:
            handle.write(json.dumps({"key": key, "answer": cache[key]}) + "\n")
    known.update(new)


def install_cpu_ple_pipeline(model_id: str, weights_dir: Path, tokenizer_dir: Path) -> None:
    """Load a Gemma "E" model with its per-layer embedding table in CPU memory.

    gemma-4-E4B-it is 8.0 B parameters (16 GB in bf16), more than the 16 GB GPU
    holds; 2.8 B of them are the per-layer embeddings (PLE), a table that is only
    looked up. The model is loaded on the CPU, everything but that table moves
    to the GPU (10.4 GB), and the lookups run on the CPU, their result moving to
    the GPU. A lookup is a gather and its scale a power of two, so the numbers
    are those of a model held wholly on the GPU. The pipeline is put into
    ``llm_utils``' single-slot cache, where ``_get_text_generation_pipeline``
    finds it. The model's memory estimate (``MODEL_MIN_MEMORY_GB``, 18 GB for
    the whole model) is set to the 12 GB this placement needs on the GPU, so the
    memory filter of ``select_local_model_candidates`` keeps it.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from opendv import llm_utils

    class CpuLookup(torch.nn.Module):
        def __init__(self, embedding: torch.nn.Module) -> None:
            super().__init__()
            self.__dict__["embedding"] = embedding  # not a submodule: .to() leaves it on the CPU

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids.to("cpu")).to(input_ids.device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.bfloat16, device_map="cpu")
    language_model = model.model.language_model
    table = language_model.embed_tokens_per_layer
    language_model.embed_tokens_per_layer = None
    model.to("cuda:0")
    language_model.embed_tokens_per_layer = CpuLookup(table)
    model.eval()
    llm_utils._clear_pipeline_cache()
    llm_utils._PIPELINE_CACHE[model_id] = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm_utils._MODELS_USED.setdefault(model_id, None)
    llm_utils.MODEL_MIN_MEMORY_GB[model_id] = 12.0


@contextmanager
def memoized_contexts() -> Iterator[None]:
    """Read each source's manuscript context once per process (it does not depend on the schema)."""
    from opendv.analysis import evaluate_llm_mapping as ev

    original = ev._source_context
    contexts: dict[tuple[str, str], str] = {}

    def cached(source: dict[str, Any], base_dir: Path, cache_dir: Path | None = None) -> str:
        key = (str(source.get("source_id")), str(cache_dir))
        if key not in contexts:
            contexts[key] = original(source, base_dir, cache_dir)
        return contexts[key]

    ev._source_context = cached
    try:
        yield
    finally:
        ev._source_context = original


@contextmanager
def no_llm() -> Iterator[None]:
    """Answer every prompt with "no answer" (the dry run counts prompts, not answers)."""
    from opendv import llm_utils

    original = llm_utils.deduce_standard_name_with_local_llm
    llm_utils.deduce_standard_name_with_local_llm = lambda *_a, **_k: None
    try:
        yield
    finally:
        llm_utils.deduce_standard_name_with_local_llm = original


def run_perturbation(
    condition: Condition,
    perturbation: Perturbation,
    halves: list[Half],
    items: dict[str, pd.DataFrame],
    work_dir: Path,
    response_cache: dict[str, str | None] | None,
) -> pd.DataFrame:
    """E1 answers of every item under one perturbation (both halves)."""
    from opendv.analysis.evaluate_llm_mapping import run_e1
    from opendv.run_batch_standardization import SchemaFamilies

    schema_path = materialize_schema(perturbation, work_dir)
    families = SchemaFamilies.load(schema_path)
    dv_ids = {c for c, domain in families.canonical_domain_lookup.items() if domain == "dv"}
    order = "shuffle" if perturbation.family == "candidate_order" else condition.candidate_order
    frames = []
    for half in halves:
        args = SimpleNamespace(
            manifest=str(half.manifest), batch_dir=str(half.batch_dir), cache_dir=str(half.cache_dir),
            max_per_source=1000, seed=BENCHMARK_SEED, models=[condition.model], shortlist=condition.shortlist,
            schema=None if schema_path == BASE_SCHEMA else str(schema_path),
            candidate_order=order, order_seed=perturbation.shuffle_seed or 0,
            samples=condition.samples, temperature=condition.temperature,
            sample_seed=condition.sample_seed + perturbation.sample_seed_offset,
        )
        frame = run_e1(args, families, dv_ids, items=items[half.name],
                       response_cache=response_cache if perturbation.use_cache else None)
        frames.append(frame.assign(half=half.name))
    return pd.concat(frames, ignore_index=True).assign(condition=condition.name, perturbation=perturbation.name)


ANSWER_COLUMNS = [
    "condition", "perturbation", "half", "source_id", "dataset_id", "alias", "label_kind", "gold",
    "shortlist", "shortlist_top_score", "prompt_sha256", "suggestion", "pre_llm_rejection",
    "post_llm_rejection", "final_unchecked", "final_checked", "llm_error", "seconds",
]


def run_condition(
    condition: Condition,
    design: dict[str, Any],
    halves: list[Half],
    work_dir: Path,
    families_filter: set[str] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Run every applicable perturbation not run yet; one CSV per perturbation under ``work_dir``."""
    dv_ids = base_dv_ids()
    items = {half.name: benchmark_items(half, dv_ids) for half in halves}
    run_dir = work_dir / (f"dry_run_{condition.name}" if dry_run else condition.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "response_cache.jsonl"
    cache = {} if dry_run else load_response_cache(cache_path)
    known = set(cache)
    written = []
    with memoized_contexts():
        for perturbation in perturbations_from_design(design):
            written += _run_one(condition, perturbation, halves, items, work_dir, run_dir, cache, cache_path, known,
                                families_filter, dry_run)
    return written


def _run_one(
    condition: Condition,
    perturbation: Perturbation,
    halves: list[Half],
    items: dict[str, pd.DataFrame],
    work_dir: Path,
    run_dir: Path,
    cache: dict[str, str | None],
    cache_path: Path,
    known: set[str],
    families_filter: set[str] | None,
    dry_run: bool,
) -> list[Path]:
    """Run one perturbation unless it does not apply, is filtered out or has run already."""
    if not condition.applies(perturbation) or (families_filter and perturbation.family not in families_filter):
        return []
    if dry_run and perturbation.family in ("identity", "sample_seed"):
        return []
    out = run_dir / f"{perturbation.name}.csv"
    if out.exists():
        return []
    started = time.perf_counter()
    n_before = len(cache)
    with no_llm() if dry_run else nullcontext():
        frame = run_perturbation(condition, perturbation, halves, items, work_dir, cache)
    if not dry_run:
        append_response_cache(cache_path, cache, known)
    errors = int(frame["llm_error"].notna().sum())
    # A run with failed generations is kept aside and redone on the next start; the answers
    # that did succeed are in the cache, so only the failed prompts are generated again.
    target = out.with_suffix(".errors.csv") if errors else out
    frame[[c for c in ANSWER_COLUMNS if c in frame]].to_csv(target, index=False)
    logger.info(
        "%s %s: %d items, %d generated, %d errors, %.0f s", condition.name, perturbation.name, len(frame),
        len(cache) - n_before if perturbation.use_cache else int(frame["prompt_sha256"].notna().sum()),
        errors, time.perf_counter() - started,
    )
    return [] if errors else [out]


# ── Analysis ─────────────────────────────────────────────────────────────────


def _answer(value: Any) -> str | None:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) or value == "" else str(value)


def flip_table(
    answers: pd.DataFrame, perturbations: dict[str, Perturbation], answer: str = "final_checked"
) -> pd.DataFrame:
    """One row per (item, perturbation) of one condition, compared with the condition's ``base``.

    ``valid`` is False where the perturbation removed the item's gold DV (only
    ``real_3_2``) or the DV the base answer named (``forced``: the answer must
    change, so it is no evidence of instability); item-runs with a generation
    error in either run are left out.
    """
    key = ITEM_KEY
    base = answers[answers["perturbation"] == "base"].set_index(key)
    rows = answers[answers["perturbation"] != "base"].join(
        base[[answer, "prompt_sha256", "shortlist", "llm_error"]], on=key, rsuffix="_base"
    )
    out = rows[key + ["perturbation", "label_kind", "gold"]].copy()
    # Plain lists with None for "no DV" (pandas would turn None into NaN, and NaN != NaN).
    before = [_answer(v) for v in rows[f"{answer}_base"]]
    after = [_answer(v) for v in rows[answer]]
    out["base_answer"] = pd.Series(before, index=rows.index, dtype=object)
    out["answer"] = pd.Series(after, index=rows.index, dtype=object)
    pert = rows["perturbation"].map(perturbations)
    removed = pert.map(lambda p: set(p.removed))
    out["family"] = pert.map(lambda p: p.family)
    out["k"] = pert.map(lambda p: p.k)
    out["forced"] = [b is not None and b in r for b, r in zip(before, removed, strict=True)]
    gold_removed = [isinstance(g, str) and g in r for g, r in zip(rows["gold"], removed, strict=True)]
    errors = rows["llm_error"].notna() | rows["llm_error_base"].notna()
    out["valid"] = ~out["forced"] & ~pd.Series(gold_removed, index=rows.index) & ~errors
    out["flip"] = [b != a for b, a in zip(before, after, strict=True)]
    out["prompt_changed"] = rows["prompt_sha256"].fillna("") != rows["prompt_sha256_base"].fillna("")
    out["set_changed"] = [
        set(str(a).split("|")) != set(str(b).split("|"))
        for a, b in zip(rows["shortlist"].fillna(""), rows["shortlist_base"].fillna(""), strict=True)
    ]
    touched = pert.map(lambda p: set(p.added) | set(p.reworded))
    out["to_touched_dv"] = [a is not None and a in t for a, t in zip(after, touched, strict=True)]
    out["from_touched_dv"] = [b is not None and b in t for b, t in zip(before, touched, strict=True)]
    out["error"] = errors
    return out.reset_index(drop=True)


def cluster_bootstrap(
    values: pd.Series, clusters: pd.Series, n_boot: int = 10_000, seed: int = MASTER_SEED
) -> tuple[float, float, float]:
    """Mean and percentile 95% CI, resampling clusters (studies) with replacement."""
    values = values.astype(float)
    if values.empty:
        return (np.nan, np.nan, np.nan)
    frame = pd.DataFrame({"v": values.to_numpy(), "c": clusters.to_numpy()})
    grouped = frame.groupby("c")["v"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_flip_test(
    differences: pd.Series, clusters: pd.Series, n_perm: int = 100_000, seed: int = MASTER_SEED
) -> float:
    """Two-sided p of a mean paired difference: the signs of whole clusters are flipped at random."""
    frame = pd.DataFrame({"d": differences.astype(float).to_numpy(), "c": clusters.to_numpy()})
    totals = frame.groupby("c")["d"].sum().to_numpy()
    observed = abs(totals.sum())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(totals)))
    null = np.abs(signs @ totals)
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + n_perm))


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


SETS = {
    # name -> (families, k or None)
    **{f"{family}_k{k}": ({family}, k) for family in EDIT_FAMILIES for k in K_LEVELS},
    "schema_edits": (set(EDIT_FAMILIES), None),  # primary endpoint
    "schema_order": ({"schema_order"}, None),
    "candidate_order": ({"candidate_order"}, None),
    "identity": ({"identity"}, None),
    "sample_seed": ({"sample_seed"}, None),
    "real_3_2": ({"real_3_2"}, None),
}


def select_set(flips: pd.DataFrame, name: str) -> pd.DataFrame:
    families, k = SETS[name]
    chosen = flips["family"].isin(families)
    if k is not None:
        chosen &= flips["k"] == k
    return flips[chosen & flips["label_kind"].isin(ANALYSIS_KINDS)]


def per_item_rates(flips: pd.DataFrame) -> pd.DataFrame:
    """Per item: valid draws and the share of them that flipped."""
    valid = flips[flips["valid"]]
    grouped = valid.groupby(ITEM_KEY, sort=True)
    out = grouped.agg(n=("flip", "size"), flips=("flip", "sum"), label_kind=("label_kind", "first"))
    out["rate"] = out["flips"] / out["n"]
    return out.reset_index()


def summarize_set(flips: pd.DataFrame, n_boot: int) -> dict[str, Any]:
    """Flip rate and its mechanism for one perturbation set of one condition."""
    valid = flips[flips["valid"]]
    rates = per_item_rates(flips)
    mean, low, high = cluster_bootstrap(rates["rate"], rates["source_id"], n_boot)
    changed = valid[valid["prompt_changed"]]
    flipped = valid[valid["flip"]]
    order_only = valid[valid["prompt_changed"] & ~valid["set_changed"]]
    none_base = valid[valid["base_answer"].isna()]
    dv_base = valid[valid["base_answer"].notna()]
    return {
        "n_items": int(len(rates)),
        "n_item_runs": int(len(valid)),
        "n_forced": int(flips["forced"].sum()),
        "n_errors": int(flips["error"].sum()),
        "flip_rate": mean, "flip_rate_ci_low": low, "flip_rate_ci_high": high,
        "items_ever_flipped": int((rates["flips"] > 0).sum()),
        "prompt_changed_rate": float(valid["prompt_changed"].mean()) if len(valid) else np.nan,
        "shortlist_set_changed_rate": float(valid["set_changed"].mean()) if len(valid) else np.nan,
        "flip_rate_given_prompt_changed": float(changed["flip"].mean()) if len(changed) else np.nan,
        "flip_rate_given_order_only_change": float(order_only["flip"].mean()) if len(order_only) else np.nan,
        "flips_with_unchanged_prompt": int((valid["flip"] & ~valid["prompt_changed"]).sum()),
        "flips_to_touched_dv": int(flipped["to_touched_dv"].sum()),
        "flips_from_touched_dv": int(flipped["from_touched_dv"].sum()),
        "none_to_dv_rate": float(none_base["answer"].notna().mean()) if len(none_base) else np.nan,
        "n_none_base_runs": int(len(none_base)),
        "dv_to_none_rate": float(dv_base["answer"].isna().mean()) if len(dv_base) else np.nan,
        "dv_to_other_dv_rate": float(
            (dv_base["answer"].notna() & (dv_base["answer"] != dv_base["base_answer"])).mean()
        ) if len(dv_base) else np.nan,
        "n_dv_base_runs": int(len(dv_base)),
        **_gold_transitions(valid),
    }


def _gold_transitions(valid: pd.DataFrame) -> dict[str, Any]:
    """On DV columns: how often a correct reference answer is lost, and a wrong one corrected."""
    dv = valid[valid["label_kind"] == "dv"]
    right = dv[dv["base_answer"] == dv["gold"]]
    wrong = dv[dv["base_answer"] != dv["gold"]]
    return {
        "correct_to_wrong_rate": float((right["answer"] != right["gold"]).mean()) if len(right) else np.nan,
        "n_correct_base_runs": int(len(right)),
        "wrong_to_correct_rate": float((wrong["answer"] == wrong["gold"]).mean()) if len(wrong) else np.nan,
        "n_wrong_base_runs": int(len(wrong)),
    }


def accuracy_change(flips: pd.DataFrame, n_boot: int) -> dict[str, Any]:
    """Change of DV accuracy and of curated non-DV columns mapped to a DV (every draw, forced ones included)."""
    out: dict[str, Any] = {}
    no_error = flips[~flips["error"]]
    for kind, metric in (("dv", "accuracy"), ("curated_non_dv", "non_dv_mapped")):
        part = no_error[no_error["label_kind"] == kind]
        if part.empty:
            continue
        if kind == "dv":
            # real_3_2 removes a gold DV; its items cannot be answered correctly and are left out.
            part = part[~part.apply(lambda r: r["gold"] in ("touch_offset",) and r["family"] == "real_3_2", axis=1)]
            hit_base = (part["base_answer"] == part["gold"]).astype(float)
            hit = (part["answer"] == part["gold"]).astype(float)
        else:
            hit_base = part["base_answer"].notna().astype(float)
            hit = part["answer"].notna().astype(float)
        per_item = pd.DataFrame({
            **{c: part[c] for c in ITEM_KEY}, "delta": hit - hit_base, "base": hit_base, "pert": hit,
        }).groupby(ITEM_KEY, sort=True).mean().reset_index()
        mean, low, high = cluster_bootstrap(per_item["delta"], per_item["source_id"], n_boot)
        out |= {
            f"{metric}_base": float(per_item["base"].mean()), f"{metric}_perturbed": float(per_item["pert"].mean()),
            f"{metric}_change": mean, f"{metric}_change_ci_low": low, f"{metric}_change_ci_high": high,
            f"{metric}_n_items": int(len(per_item)),
        }
    return out


def read_answers(work_dir: Path, condition: str) -> pd.DataFrame:
    paths = [p for p in sorted((work_dir / condition).glob("*.csv")) if not p.name.endswith(".errors.csv")]
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    if not frames:
        return pd.DataFrame()
    answers = pd.concat(frames, ignore_index=True)
    for column in ("gold", "suggestion", "final_unchecked", "final_checked", "shortlist", "prompt_sha256", "llm_error"):
        if column in answers:
            answers[column] = answers[column].astype(object).where(answers[column].notna(), None)
    return answers


def analyze(design: dict[str, Any], work_dir: Path, output_dir: Path, n_boot: int = 10_000) -> dict[str, Any]:
    perturbations = {p.name: p for p in perturbations_from_design(design)}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, per_column, item_rates = [], [], {}
    conditions_run = []
    for name in CONDITIONS:
        answers = read_answers(work_dir, name)
        if answers.empty or "base" not in set(answers["perturbation"]):
            continue
        conditions_run.append(name)
        with gzip.open(output_dir / f"answers_{name}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            answers[[c for c in ANSWER_COLUMNS if c in answers]].to_csv(handle, index=False)
        for answer in ("final_checked", "final_unchecked"):
            flips = flip_table(answers, perturbations, answer)
            for set_name in SETS:
                chosen = select_set(flips, set_name)
                if chosen.empty:
                    continue
                rows.append({
                    "condition": name, "answer": answer.removeprefix("final_"), "set": set_name,
                    "n_perturbations": int(chosen["perturbation"].nunique()),
                    **summarize_set(chosen, n_boot), **accuracy_change(chosen, n_boot),
                })
            primary = per_item_rates(select_set(flips, "schema_edits"))
            item_rates[(name, answer)] = primary
            if answer == "final_checked":
                base = answers[answers["perturbation"] == "base"].set_index(ITEM_KEY)
                table = primary.join(base[["gold", "final_checked"]], on=ITEM_KEY)
                per_column.append(table.assign(condition=name).rename(columns={"final_checked": "base_answer"}))
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "flip_rates.csv", index=False)
    reference = reference_accuracy(work_dir, conditions_run)
    reference.to_csv(output_dir / "reference_accuracy.csv", index=False)
    if per_column:
        pd.concat(per_column, ignore_index=True).to_csv(output_dir / "per_column_flip_rates.csv", index=False)
    comparisons = compare_conditions(item_rates, conditions_run, n_boot)
    comparisons.to_csv(output_dir / "comparisons.csv", index=False)
    result = {
        "conditions": conditions_run,
        "n_boot": n_boot,
        "primary_endpoint": "schema_edits flip rate of the checked answer (add, remove, reword; k = 1 and 5)",
        "reference_accuracy": reference.to_dict(orient="records"),
        "flip_rates": summary.to_dict(orient="records"),
        "comparisons": comparisons.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    return result


def reference_accuracy(work_dir: Path, conditions: list[str]) -> pd.DataFrame:
    """Accuracy of each condition's reference answers (Wilson 95% intervals), as in the benchmark summary."""
    from opendv.analysis.evaluate_llm_mapping import wilson

    rows = []
    for name in conditions:
        base = read_answers(work_dir, name)
        base = base[base["perturbation"] == "base"]
        dv = base[base["label_kind"] == "dv"]
        non_dv = base[base["label_kind"] == "curated_non_dv"]
        for answer in ("final_checked", "final_unchecked"):
            correct = wilson(int((dv[answer] == dv["gold"]).sum()), len(dv))
            mapped = wilson(int(non_dv[answer].notna().sum()), len(non_dv))
            rows.append({
                "condition": name, "answer": answer.removeprefix("final_"),
                "dv_accuracy_k": correct["k"], "dv_n": correct["n"], "dv_accuracy": correct["rate"],
                "dv_accuracy_ci_low": correct["ci95"][0], "dv_accuracy_ci_high": correct["ci95"][1],
                "non_dv_mapped_k": mapped["k"], "non_dv_n": mapped["n"], "non_dv_mapped": mapped["rate"],
                "llm_errors": int(base["llm_error"].notna().sum()),
            })
    return pd.DataFrame(rows)


def compare_conditions(
    item_rates: dict[tuple[str, str], pd.DataFrame], conditions_run: list[str], n_boot: int
) -> pd.DataFrame:
    rows = []
    for answer in ("final_checked", "final_unchecked"):
        for condition, reference in COMPARISONS:
            if condition not in conditions_run or reference not in conditions_run:
                continue
            a, b = item_rates[(condition, answer)], item_rates[(reference, answer)]
            paired = a.merge(b, on=ITEM_KEY, suffixes=("", "_ref"))
            difference = paired["rate"] - paired["rate_ref"]
            mean, low, high = cluster_bootstrap(difference, paired["source_id"], n_boot)
            rows.append({
                "answer": answer.removeprefix("final_"), "condition": condition, "reference": reference,
                "n_items": int(len(paired)), "n_studies": int(paired["source_id"].nunique()),
                "flip_rate": float(paired["rate"].mean()), "flip_rate_reference": float(paired["rate_ref"].mean()),
                "difference": mean, "difference_ci_low": low, "difference_ci_high": high,
                "p_sign_flip": sign_flip_test(difference, paired["source_id"]),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = np.nan
        for _, group in out.groupby("answer"):
            out.loc[group.index, "p_holm"] = holm(group["p_sign_flip"].tolist())
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    raise TypeError(type(value))


def dry_run_summary(design: dict[str, Any], work_dir: Path, condition: str) -> pd.DataFrame:
    """Share of item prompts each perturbation changes (no LLM answers involved)."""
    perturbations = {p.name: p for p in perturbations_from_design(design)}
    answers = read_answers(work_dir, f"dry_run_{condition}")
    flips = flip_table(answers, perturbations, "final_checked")
    flips = flips[flips["label_kind"].isin(ANALYSIS_KINDS)]
    return flips.groupby("perturbation", sort=False).agg(
        family=("family", "first"), k=("k", "first"), prompt_changed=("prompt_changed", "mean"),
        set_changed=("set_changed", "mean"), n=("prompt_changed", "size"),
    ).reset_index()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock", "evaluate_llm_mapping", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "dry-run", "run", "analyze"):
        p = sub.add_parser(command)
        p.add_argument("--half", type=parse_half, action="append", default=None,
                       help="NAME=BATCH_DIR,MANIFEST,CACHE_DIR (default: det23l and det16l of data/processed).")
        p.add_argument("--work-dir", type=Path, default=WORK_DIR)
        p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
        if command in ("run", "dry-run"):
            p.add_argument("--condition", choices=sorted(CONDITIONS), required=command == "run",
                           default="e2b_definition")
        if command == "run":
            p.add_argument("--cpu-ple", nargs=2, type=Path, metavar=("WEIGHTS_DIR", "TOKENIZER_DIR"), default=None,
                           help="Load the condition's model with its per-layer embeddings on the CPU "
                                "(install_cpu_ple_pipeline), weights and tokenizer from these snapshot folders.")
            p.add_argument("--families", default=None, help="Comma-separated families to run (default: all).")
        if command == "analyze":
            p.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args(argv)
    halves = args.half or default_halves()
    design_path = args.output_dir / "design.json"
    if args.command == "plan":
        design = build_design(halves)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
        for perturbation in perturbations_from_design(design):
            materialize_schema(perturbation, args.work_dir)
        print(f"{len(design['perturbations'])} perturbations, {len(design['unrelated_dvs'])} unrelated DVs")
        return
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if args.command == "dry-run":
        run_condition(CONDITIONS[args.condition], design, halves, args.work_dir, dry_run=True)
        table = dry_run_summary(design, args.work_dir, args.condition)
        table.to_csv(args.work_dir / f"dry_run_{args.condition}.csv", index=False)
        print(table.to_string(index=False))
        return
    if args.command == "run":
        families = set(args.families.split(",")) if args.families else None
        condition = CONDITIONS[args.condition]
        if args.cpu_ple:
            install_cpu_ple_pipeline(condition.model, *args.cpu_ple)
        run_condition(condition, design, halves, args.work_dir, families)
        from opendv import llm_utils

        if llm_utils.generation_errors():
            sys.exit(f"{len(llm_utils.generation_errors())} generation errors; rerun the affected perturbations")
        return
    analyze(design, args.work_dir, args.output_dir, args.n_boot)


__all__ = [
    "CONDITIONS", "Condition", "Perturbation", "analyze", "apply_perturbation", "build_design",
    "check_perturbation", "cluster_bootstrap", "flip_table", "holm", "plan_perturbations",
    "protected_dvs", "sign_flip_test", "unrelated_dvs",
]


if __name__ == "__main__":
    main()
