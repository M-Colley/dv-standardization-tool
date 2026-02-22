import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.llm_utils import (
    _extract_dois,
    _fetch_pdf_text_from_doi,
    collect_repository_context,
    select_local_model_candidates,
    deduce_standard_name_with_local_llm,
)


class LLMUtilsTests(unittest.TestCase):
    def test_extract_dois_deduplicates_and_cleans(self):
        text = "See doi:10.1145/12345.67890, and DOI 10.1145/12345.67890."
        dois = _extract_dois(text)
        self.assertEqual(dois, ["10.1145/12345.67890"])

    def test_collect_repository_context_reads_readme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("Project about trust ratings.", encoding="utf-8")

            context = collect_repository_context(root)

            self.assertIn("README", context)
            self.assertIn("trust ratings", context)

    def test_collect_repository_context_reads_pdf_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("Readme text", encoding="utf-8")
            (root / "paper.pdf").write_bytes(b"%PDF-1.4")

            with mock.patch(
                "scripts.llm_utils._extract_text_from_pdf",
                return_value="PDF context on response time",
            ):
                context = collect_repository_context(root)

            self.assertIn("PDF:paper.pdf", context)
            self.assertIn("response time", context)

    def test_collect_repository_context_fetches_doi_pdf_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text(
                "Paper DOI: 10.1145/3706598.3713762",
                encoding="utf-8",
            )

            with mock.patch(
                "scripts.llm_utils._fetch_pdf_text_from_doi",
                return_value="Online PDF context about user trust.",
            ) as mocked:
                context = collect_repository_context(root)

            mocked.assert_called_once_with("10.1145/3706598.3713762")
            self.assertIn("DOI_PDF:10.1145/3706598.3713762", context)
            self.assertIn("user trust", context)

    def test_fetch_pdf_text_from_doi_follows_html_pdf_link(self):
        html = b'<html><body><a href="/paper.pdf">PDF</a></body></html>'
        pdf = b"%PDF-1.4 dummy"

        with mock.patch(
            "scripts.llm_utils._http_get",
            side_effect=[(html, "text/html"), (pdf, "application/pdf")],
        ):
            with mock.patch(
                "scripts.llm_utils._extract_text_from_pdf_bytes",
                return_value="Recovered PDF text",
            ):
                text = _fetch_pdf_text_from_doi("10.1145/3706598.3713762")

        self.assertIn("Recovered PDF text", text)


    def test_select_local_model_candidates_uses_env_override(self):
        with mock.patch.dict("os.environ", {"DV_LLM_MODELS": "foo/bar,baz/qux"}, clear=False):
            selected = select_local_model_candidates()

        self.assertEqual(selected, ["foo/bar", "baz/qux"])

    def test_select_local_model_candidates_filters_by_memory(self):
        with mock.patch.dict("os.environ", {"DV_LLM_MODELS": ""}, clear=False):
            with mock.patch("scripts.llm_utils._detect_available_memory_gb", return_value=5.0):
                selected = select_local_model_candidates(
                    preferred_models=[
                        "Qwen/Qwen2.5-3B-Instruct",
                        "Qwen/Qwen2.5-1.5B-Instruct",
                    ]
                )

        self.assertEqual(selected, ["Qwen/Qwen2.5-1.5B-Instruct"])

    def test_deduce_standard_name_with_local_llm_returns_matching_candidate(self):
        def _fake_pipeline(prompt, max_new_tokens, do_sample):
            return [{"generated_text": prompt + "task_completion_time"}]

        with mock.patch("scripts.llm_utils._get_text_generation_pipeline", return_value=_fake_pipeline):
            inferred = deduce_standard_name_with_local_llm(
                raw_column_name="Duration",
                canonical_candidates=["task_completion_time", "trust_rating"],
            )

        self.assertEqual(inferred, "task_completion_time")


if __name__ == "__main__":
    unittest.main()
