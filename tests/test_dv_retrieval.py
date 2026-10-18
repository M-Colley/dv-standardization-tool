"""Tests for definition-aware candidate retrieval (scripts/dv_retrieval.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import dv_retrieval  # noqa: E402
from scripts.llm_coordinator import _augment_mapping_with_llm_deductions  # noqa: E402


class DefinitionRetrieverTests(unittest.TestCase):
    def test_definitions_cover_every_schema_dv(self):
        definitions = dv_retrieval.load_dv_definitions()
        ids = {d.dv_id for d in definitions}
        self.assertIn("mental_demand", ids)
        self.assertGreaterEqual(len(ids), 150)

    def test_ranks_the_matching_dv_first_for_plain_names(self):
        retriever = dv_retrieval.get_retriever()
        self.assertEqual(retriever.rank("MentalDemand", k=1)[0][0], "mental_demand")
        self.assertIn("trust_rating", [dv for dv, _ in retriever.rank("TrustScore", k=5)])

    def test_exclude_alias_removes_the_held_out_alias_everywhere(self):
        definitions = [
            dv_retrieval.DvDefinition("a_dv", ("zzqx",), "alpha"),
            dv_retrieval.DvDefinition("b_dv", (), "beta"),
        ]
        retriever = dv_retrieval.DefinitionRetriever(definitions)
        self.assertEqual(retriever.rank("zzqx", k=1)[0][0], "a_dv")
        held_out = dict(retriever.rank("zzqx", k=2, exclude_alias="zzqx"))
        self.assertEqual(held_out["a_dv"], 0.0)

    def test_coordinator_uses_definition_candidates_when_asked(self):
        seen: list[list[str]] = []

        def fake_llm(raw_column_name, canonical_candidates, **_kwargs):
            seen.append(list(canonical_candidates))
            return None

        mapping = {"nasa_tlx_mental": "mental_demand", "trust": "trust_rating"}
        with mock.patch("scripts.llm_coordinator.deduce_standard_name_with_local_llm", side_effect=fake_llm), \
                mock.patch("scripts.llm_coordinator._select_llm_candidate_shortlist",
                           return_value=(["trust_rating", "mental_demand"], 90.0)):
            _augment_mapping_with_llm_deductions(mapping, ["MentalDemandScore"], REPO_ROOT, shortlist="definition")
        self.assertEqual(len(seen[0]), 2)
        self.assertEqual(seen[0][0], "mental_demand")


def _trigram_encoder(calls: list[int]):
    """Deterministic stand-in for a sentence model: hashed character trigrams."""
    import numpy as np

    def encode(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        out = np.zeros((len(texts), 256))
        for row, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for i in range(len(padded) - 2):
                out[row, hash(padded[i:i + 3]) % 256] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    return encode


class DenseRetrieverTests(unittest.TestCase):
    DEFINITIONS = [
        dv_retrieval.DvDefinition("mental_demand", ("tlx_mental",), "mental demand Mental Demand NASA-TLX"),
        dv_retrieval.DvDefinition("trust_rating", ("trust_score",), "trust rating Trust"),
        dv_retrieval.DvDefinition("usability", ("sus_score",), "usability System Usability Scale"),
    ]

    def test_ranks_by_embedding_similarity(self):
        retriever = dv_retrieval.DenseRetriever(self.DEFINITIONS, _trigram_encoder([]))
        self.assertEqual(retriever.rank("MentalDemand", k=1)[0][0], "mental_demand")
        self.assertEqual(retriever.rank("trustScore", k=1)[0][0], "trust_rating")

    def test_leave_alias_out_reembeds_only_affected_definitions(self):
        calls: list[int] = []
        retriever = dv_retrieval.DenseRetriever(self.DEFINITIONS, _trigram_encoder(calls))
        self.assertEqual(calls, [3])  # all definitions embedded once
        retriever.rank("sus_score", k=3, exclude_alias="sus_score")
        self.assertEqual(calls[1:], [1, 1])  # one definition re-embedded, then the query

    def test_reciprocal_rank_fusion(self):
        fused = dv_retrieval.reciprocal_rank_fusion(["a", "b", "c"], ["b", "c", "a"])
        self.assertEqual(fused[0], "b")
        self.assertEqual(sorted(fused), ["a", "b", "c"])

    def test_coordinator_dense_option_uses_fused_shortlist(self):
        seen: list[list[str]] = []

        def fake_llm(raw_column_name, canonical_candidates, **_kwargs):
            seen.append(list(canonical_candidates))
            return None

        with mock.patch("scripts.llm_coordinator.deduce_standard_name_with_local_llm", side_effect=fake_llm), \
                mock.patch("scripts.llm_coordinator._select_llm_candidate_shortlist",
                           return_value=(["trust_rating", "mental_demand"], 90.0)), \
                mock.patch("scripts.dv_retrieval.dense_shortlist", return_value=["usability", "trust_rating"]):
            _augment_mapping_with_llm_deductions({"trust": "trust_rating"}, ["SUS"], REPO_ROOT, shortlist="dense")
        self.assertEqual(seen, [["usability", "trust_rating"]])

    def test_model_prefixes_are_applied(self):
        seen: list[str] = []

        def encoder(texts):
            seen.extend(texts)
            return _trigram_encoder([])(texts)

        retriever = dv_retrieval.DenseRetriever(self.DEFINITIONS, encoder, "query: ", "passage: ")
        retriever.rank("TrustScore", k=1)
        self.assertTrue(all(t.startswith("passage: ") for t in seen[:3]))
        self.assertEqual(seen[-1], "query: trust score")  # name_tokens lower-cases


if __name__ == "__main__":
    unittest.main()
