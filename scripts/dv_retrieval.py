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

from collections.abc import Callable
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


__all__ = [
    "DefinitionRetriever",
    "DenseRetriever",
    "DvDefinition",
    "definition_shortlist",
    "get_retriever",
    "load_dv_definitions",
    "sentence_transformer_encoder",
]
