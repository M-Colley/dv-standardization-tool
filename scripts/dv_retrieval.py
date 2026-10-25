"""Definition-aware candidate retrieval for LLM DV mapping.

The LLM picks from a shortlist, so the shortlist bounds its accuracy. Ranking
canonical DVs by fuzzy similarity of a column name to their *aliases* misses
the right DV for most real dataset columns (``Acceptance1`` vs. ``AOA1``,
``unsafeJudgement`` vs. ``situational_awareness``). This module ranks DVs by
character n-gram TF-IDF similarity to their *definitions* from
``schemas/standard_dv_mapping.yaml``: the mean of a *name* similarity (id,
label, aliases, instruments) and a *full-definition* similarity (names plus
notes, cluster and unit). Long notes alone dilute a DV's own name
("MentalDemand" would rank ``sart_demand`` first); the blend keeps names
decisive while notes still contribute. On held-out real columns it raised
recall@8 from 42% to 83% (``analyses/evaluate_shortlist_retrieval.py``; the
blend weight was chosen on those items, so confirm on new data).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from scripts.llm_plausibility import name_tokens, normalize_name

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"
DEFAULT_DV_SCHEMA_PATH = SCHEMAS_DIR / "standard_dv_mapping.yaml"
DEFAULT_CLUSTERS_PATH = SCHEMAS_DIR / "thematic_clusters.yaml"


NAME_WEIGHT = 0.5


@dataclass(frozen=True)
class DvDefinition:
    dv_id: str
    aliases: tuple[str, ...]
    names: str  # id tokens, label, instruments
    context: str = ""  # notes, cluster description, unit

    def name_text(self, exclude_alias: str | None = None) -> str:
        held_out = normalize_name(exclude_alias) if exclude_alias else None
        aliases = (a for a in self.aliases if normalize_name(a) != held_out)
        return self.names + " " + " ".join(" ".join(name_tokens(a)) for a in aliases)

    def full_text(self, exclude_alias: str | None = None) -> str:
        return self.name_text(exclude_alias) + " " + self.context


def load_dv_definitions(
    schema_path: Path = DEFAULT_DV_SCHEMA_PATH,
    clusters_path: Path = DEFAULT_CLUSTERS_PATH,
) -> list[DvDefinition]:
    schema = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8")) or {}
    clusters: dict[str, str] = {}
    if Path(clusters_path).is_file():
        cluster_data = yaml.safe_load(Path(clusters_path).read_text(encoding="utf-8")) or {}
        clusters = {
            str(c.get("id")): f"{c.get('label', '')} {c.get('description', '')}"
            for c in cluster_data.get("clusters") or []
            if isinstance(c, dict)
        }
    definitions = []
    for entry in schema.get("dvs") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        measurement = entry.get("measurement") or {}
        definitions.append(DvDefinition(
            dv_id=str(entry["id"]),
            aliases=tuple(str(a) for a in entry.get("aliases") or [] if isinstance(a, str)),
            names=" ".join([
                " ".join(name_tokens(entry["id"])),
                str(entry.get("label", "")),
                " ".join(str(i) for i in entry.get("instruments") or []),
            ]),
            context=" ".join([
                str(entry.get("notes", "")),
                clusters.get(str(entry.get("cluster", "")), ""),
                str(measurement.get("primary_unit", "")),
            ]),
        ))
    return definitions


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)


class DefinitionRetriever:
    """Rank canonical DVs for a column name by similarity to their definitions."""

    def __init__(self, definitions: list[DvDefinition], name_weight: float = NAME_WEIGHT) -> None:
        self._definitions = definitions
        self._ids = [d.dv_id for d in definitions]
        self._name_weight = name_weight
        self._fitted = self._fit(None)

    def _fit(self, exclude_alias: str | None):
        fitted = []
        for texts in (
            [d.name_text(exclude_alias) for d in self._definitions],
            [d.full_text(exclude_alias) for d in self._definitions],
        ):
            vectorizer = _vectorizer()
            fitted.append((vectorizer, vectorizer.fit_transform(texts)))
        return fitted

    def rank(self, column_name: str, k: int = 8, exclude_alias: str | None = None) -> list[tuple[str, float]]:
        """Top-*k* ``(dv_id, cosine)`` pairs.

        ``exclude_alias`` removes that alias from every definition first
        (leave-alias-out evaluation); this refits the vectorizer, so use it
        for evaluation only.
        """
        (name_vec, name_mat), (full_vec, full_mat) = (
            self._fitted if exclude_alias is None else self._fit(exclude_alias)
        )
        query = " ".join(name_tokens(column_name))
        scores = (
            self._name_weight * linear_kernel(name_vec.transform([query]), name_mat).ravel()
            + (1.0 - self._name_weight) * linear_kernel(full_vec.transform([query]), full_mat).ravel()
        )
        order = scores.argsort(kind="stable")[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in order]


# Query/document prefixes that some embedding models are trained with.
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
}

Encoder = Callable[[list[str]], np.ndarray]


def sentence_transformer_encoder(model_name: str, device: str | None = None) -> Encoder:
    """Encoder backed by a sentence-transformers model (normalized embeddings).

    Loads only from the local Hugging Face cache when ``HF_HUB_OFFLINE=1``;
    otherwise the model is downloaded on first use.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False))

    return encode


class DenseRetriever:
    """Rank canonical DVs by cosine similarity of text embeddings.

    Same definitions and leave-alias-out protocol as :class:`DefinitionRetriever`;
    documents are the full definitions, queries the tokenized column name.
    """

    def __init__(
        self,
        definitions: list[DvDefinition],
        encoder: Encoder,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self._definitions = definitions
        self._ids = [d.dv_id for d in definitions]
        self._encode = encoder
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._matrix = self._encode_documents(None)

    @classmethod
    def from_model(cls, definitions: list[DvDefinition], model_name: str, **kwargs: Any) -> DenseRetriever:
        query_prefix, document_prefix = MODEL_PREFIXES.get(model_name, ("", ""))
        return cls(definitions, sentence_transformer_encoder(model_name, **kwargs), query_prefix, document_prefix)

    def _encode_documents(self, exclude_alias: str | None, only: list[int] | None = None) -> np.ndarray:
        indices = range(len(self._definitions)) if only is None else only
        texts = [self._document_prefix + self._definitions[i].full_text(exclude_alias) for i in indices]
        return self._encode(texts)

    def rank(self, column_name: str, k: int = 8, exclude_alias: str | None = None) -> list[tuple[str, float]]:
        matrix = self._matrix
        if exclude_alias is not None:
            # Re-embed only the definitions that list the held-out alias.
            held_out = normalize_name(exclude_alias)
            affected = [
                i for i, d in enumerate(self._definitions)
                if any(normalize_name(alias) == held_out for alias in d.aliases)
            ]
            if affected:
                matrix = matrix.copy()
                matrix[affected] = self._encode_documents(exclude_alias, affected)
        query = self._encode([self._query_prefix + " ".join(name_tokens(column_name))])[0]
        scores = matrix @ query
        order = scores.argsort(kind="stable")[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in order]


@lru_cache(maxsize=4)
def get_retriever(schema_path: str = str(DEFAULT_DV_SCHEMA_PATH)) -> DefinitionRetriever:
    return DefinitionRetriever(load_dv_definitions(Path(schema_path)))


def definition_shortlist(column_name: str, max_candidates: int = 8, **kwargs: Any) -> list[str]:
    return [dv_id for dv_id, _ in get_retriever(**kwargs).rank(column_name, k=max_candidates)]


# Best dense model in analyses/evaluate_shortlist_retrieval.py (real columns:
# recall@8 103/106 vs. 88/106 for TF-IDF definitions).
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=2)
def get_dense_retriever(model_name: str = DEFAULT_DENSE_MODEL, device: str = "cpu") -> DenseRetriever:
    return DenseRetriever.from_model(load_dv_definitions(), model_name, device=device)


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> list[str]:
    """Fuse rankings by summed 1 / (k + rank) (Cormack et al., 2009)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def dense_shortlist(column_name: str, max_candidates: int = 8, **kwargs: Any) -> list[str]:
    """Dense and TF-IDF definition rankings fused. Fusion keeps exact lexical
    matches the embedding model can miss ("MentalDemand" -> mental_demand) and
    gives the highest recall@8 in analyses/evaluate_shortlist_retrieval.py."""
    depth = max(max_candidates * 2, 16)
    dense = [dv for dv, _ in get_dense_retriever(**kwargs).rank(column_name, k=depth)]
    lexical = [dv for dv, _ in get_retriever().rank(column_name, k=depth)]
    return reciprocal_rank_fusion(lexical, dense)[:max_candidates]


# Dense-only candidates that replace the lowest TF-IDF ranks in the mixed
# shortlist (fixed before the end-to-end evaluation; see research agenda 2.4).
N_DENSE_EXTRA = 2


def mix_rankings(lexical: list[str], dense: list[str], n: int, n_dense: int = N_DENSE_EXTRA) -> list[str]:
    """Top ``n - n_dense`` TF-IDF candidates in their order, then the best dense
    candidates not among them.

    Fusion (``dense_shortlist``) raised recall but filled the list with
    near-synonyms, and a small model then chose worse. This keeps the
    lexical list, which the model handles well, and adds only what dense
    retrieval finds beyond it.
    """
    n_dense = max(0, min(n_dense, n))
    keep = list(dict.fromkeys(lexical))[: n - n_dense]
    extras = [dv for dv in dict.fromkeys(dense) if dv not in keep][: n - len(keep)]
    return keep + extras


def mixed_shortlist(
    column_name: str, max_candidates: int = 8, n_dense: int = N_DENSE_EXTRA, **kwargs: Any
) -> list[str]:
    depth = max(max_candidates * 2, 16)
    dense = [dv for dv, _ in get_dense_retriever(**kwargs).rank(column_name, k=depth)]
    lexical = [dv for dv, _ in get_retriever().rank(column_name, k=depth)]
    return mix_rankings(lexical, dense, max_candidates, n_dense)


# ── Construct hierarchy ──────────────────────────────────────────────────────
# Offering a scale and one of its own items in the same shortlist invites a
# small model to pick the scale for an item column ("SUS3" -> usability);
# that caused 18 of the 19 answers the mixed shortlist lost (research agenda
# 2.4). The hierarchy filter keeps only the higher-ranked of the two.

_ITEM_ID = re.compile(r"^(?P<construct>.+)_item_\d+$")


@lru_cache(maxsize=4)
def construct_hierarchy(schema_path: str = str(DEFAULT_DV_SCHEMA_PATH)) -> dict[str, frozenset[str]]:
    """DV id -> the DV ids it must not share a shortlist with (its scales and its items).

    Scale/item pairs come from the instrument registry (a score's DV and its
    item DVs), from ``subitem_of`` in the DV schema, and from the
    ``<construct>_item_NN`` naming convention.
    """
    from scripts import instruments

    schema = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8")) or {}
    entries = [e for e in schema.get("dvs") or [] if isinstance(e, dict) and e.get("id")]
    ids = {str(e["id"]) for e in entries}
    pairs: set[tuple[str, str]] = set()
    for entry in entries:
        dv_id = str(entry["id"])
        parent = str(entry.get("subitem_of") or "")
        match = _ITEM_ID.match(dv_id)
        if not parent and match and match.group("construct") in ids:
            parent = match.group("construct")
        if parent and parent != dv_id:
            pairs.add((parent, dv_id))
    for scale, items in instruments.construct_subitems(instruments.load_instruments()).items():
        pairs.update((scale, item) for item in items if item in ids and item != scale)
    related: dict[str, set[str]] = defaultdict(set)
    for parent, child in pairs:
        related[parent].add(child)
        related[child].add(parent)
    return {dv: frozenset(others) for dv, others in related.items()}


def apply_hierarchy(
    ranking: Iterable[str], n: int, related: dict[str, frozenset[str]] | None = None
) -> list[str]:
    """First ``n`` candidates of ``ranking``, skipping any whose scale or item is
    already listed (the higher-ranked of the two wins; later ones fill up)."""
    related = construct_hierarchy() if related is None else related
    out: list[str] = []
    for dv in dict.fromkeys(ranking):
        if any(dv in related.get(kept, ()) for kept in out):
            continue
        out.append(dv)
        if len(out) == n:
            break
    return out


def hierarchical_shortlist(column_name: str, max_candidates: int = 8, **kwargs: Any) -> list[str]:
    """Dense + TF-IDF fusion (as ``dense_shortlist``) with the hierarchy filter."""
    depth = max(max_candidates * 3, 24)
    dense = [dv for dv, _ in get_dense_retriever(**kwargs).rank(column_name, k=depth)]
    lexical = [dv for dv, _ in get_retriever().rank(column_name, k=depth)]
    return apply_hierarchy(reciprocal_rank_fusion(lexical, dense), max_candidates)


__all__ = [
    "DefinitionRetriever",
    "DenseRetriever",
    "DvDefinition",
    "apply_hierarchy",
    "construct_hierarchy",
    "definition_shortlist",
    "hierarchical_shortlist",
    "dense_shortlist",
    "get_dense_retriever",
    "get_retriever",
    "load_dv_definitions",
    "mix_rankings",
    "mixed_shortlist",
    "reciprocal_rank_fusion",
    "sentence_transformer_encoder",
]
