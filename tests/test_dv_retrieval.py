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


if __name__ == "__main__":
    unittest.main()
