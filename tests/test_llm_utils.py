import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.llm_utils import (
    _extract_dois,
    _fetch_pdf_text_from_doi,
    _select_pdf_context_excerpt,
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

    def test_select_pdf_context_excerpt_prefers_measurement_relevant_blocks(self):
        pdf_text = """
        Title of Paper

        Abstract
        We study trust and workload in automated driving.

        Measures
        Participants completed a trust rating questionnaire and the NASA-TLX workload scale.
        We also recorded response time and task completion time.

        References
        Smith et al. 2020; Doe et al. 2021; Roe et al. 2022; Poe et al. 2023.
        """

        excerpt = _select_pdf_context_excerpt(pdf_text, max_chars=500)

        self.assertIn("trust rating questionnaire", excerpt)
        self.assertIn("NASA-TLX", excerpt)
        self.assertNotIn("Smith et al. 2020", excerpt)

    def test_select_pdf_context_excerpt_falls_back_to_full_text(self):
        pdf_text = "Alpha Beta Gamma Delta Epsilon"

        excerpt = _select_pdf_context_excerpt(pdf_text, max_chars=500)

        self.assertEqual(excerpt, "Alpha Beta Gamma Delta Epsilon")

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

    def test_collect_repository_context_surfaces_explicit_doi_pdf_and_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with mock.patch(
                "scripts.llm_utils._fetch_pdf_text_from_doi",
                return_value="DOI manuscript text about trust.",
            ):
                with mock.patch(
                    "scripts.llm_utils._fetch_pdf_text_from_url",
                    return_value="Direct PDF appendix text about workload.",
                ):
                    context = collect_repository_context(
                        root,
                        explicit_dois=["10.1145/3706598.3713762"],
                        explicit_pdf_urls=["https://example.com/paper.pdf"],
                        extra_context=["Study context: trust and workload ratings."],
                    )

        self.assertIn("[CONTEXT_SUMMARY]", context)
        self.assertIn("Source notes: Study context: trust and workload ratings.", context)
        self.assertIn("Detected DOIs: 10.1145/3706598.3713762", context)
        self.assertIn("Fetched DOI PDFs: 10.1145/3706598.3713762", context)
        self.assertIn("Explicit PDF URLs: https://example.com/paper.pdf", context)
        self.assertIn("Fetched explicit PDFs: https://example.com/paper.pdf", context)
        self.assertIn("SOURCE_CONTEXT", context)
        self.assertIn("DOI manuscript text about trust.", context)
        self.assertIn("Direct PDF appendix text about workload.", context)

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
            with mock.patch("scripts.llm_utils._detect_cuda_memory_gb", return_value=None):
                with mock.patch("scripts.llm_utils._detect_available_memory_gb", return_value=64.0):
                    selected = select_local_model_candidates()

        self.assertEqual(selected, ["foo/bar", "baz/qux"])

    def test_select_local_model_candidates_filters_by_memory(self):
        with mock.patch.dict("os.environ", {"DV_LLM_MODELS": ""}, clear=False):
            with mock.patch("scripts.llm_utils._detect_cuda_memory_gb", return_value=None):
                with mock.patch("scripts.llm_utils._detect_available_memory_gb", return_value=5.0):
                    selected = select_local_model_candidates(
                        preferred_models=[
                            "Qwen/Qwen2.5-3B-Instruct",
                            "Qwen/Qwen2.5-1.5B-Instruct",
                        ]
                    )

        self.assertEqual(selected, ["Qwen/Qwen2.5-1.5B-Instruct"])

    def test_select_local_model_candidates_returns_empty_when_models_exceed_cuda_memory(self):
        with mock.patch.dict("os.environ", {"DV_LLM_MODELS": ""}, clear=False):
            with mock.patch("scripts.llm_utils._detect_cuda_memory_gb", return_value=4.0):
                selected = select_local_model_candidates(
                    preferred_models=[
                        "google/gemma-4-E4B-it",
                        "Qwen/Qwen2.5-3B-Instruct",
                    ]
                )

        self.assertEqual(selected, [])

    def test_select_local_model_candidates_filters_env_override_by_cuda_memory(self):
        with mock.patch.dict("os.environ", {"DV_LLM_MODELS": "google/gemma-4-E4B-it"}, clear=False):
            with mock.patch("scripts.llm_utils._detect_cuda_memory_gb", return_value=4.0):
                selected = select_local_model_candidates()

        self.assertEqual(selected, [])

    def test_deduce_standard_name_with_local_llm_returns_matching_candidate(self):
        def _fake_pipeline(prompt, max_new_tokens, do_sample, return_full_text):
            self.assertIn("Match the measured human outcome or construct", prompt)
            self.assertIn("task_completion_time | label=Task Completion Time", prompt)
            self.assertFalse(return_full_text)
            return [{"generated_text": "task_completion_time"}]

        with mock.patch(
            "scripts.llm_utils.select_local_model_candidates",
            return_value=["fake/test-model"],
        ):
            with mock.patch(
                "scripts.llm_utils._get_text_generation_pipeline",
                return_value=_fake_pipeline,
            ):
                inferred = deduce_standard_name_with_local_llm(
                    raw_column_name="Duration",
                    canonical_candidates=["task_completion_time", "trust_rating"],
                )

        self.assertEqual(inferred, "task_completion_time")

    def test_deduce_standard_name_with_local_llm_uses_precomputed_context(self):
        def _fake_pipeline(prompt, max_new_tokens, do_sample, return_full_text):
            self.assertIn("cached context", prompt)
            return [{"generated_text": "task_completion_time"}]

        with mock.patch(
            "scripts.llm_utils.select_local_model_candidates",
            return_value=["fake/test-model"],
        ):
            with mock.patch("scripts.llm_utils.collect_repository_context") as mocked_context:
                with mock.patch(
                    "scripts.llm_utils._get_text_generation_pipeline",
                    return_value=_fake_pipeline,
                ):
                    inferred = deduce_standard_name_with_local_llm(
                        raw_column_name="Duration",
                        canonical_candidates=["task_completion_time", "trust_rating"],
                        repository_context="cached context",
                    )

        mocked_context.assert_not_called()
        self.assertEqual(inferred, "task_completion_time")


class PipelineCacheTests(unittest.TestCase):
    """Cover the single-slot pipeline cache and its helpers.

    Holding two transformers pipelines simultaneously can pin tens of GB of
    model weights in RAM/VRAM and trigger OOM, so the cache is intentionally
    sized to one entry. These tests verify that:

    * The same model id returns the cached pipeline (no re-load).
    * Switching to a different model id evicts the previous pipeline first.
    * The public _clear_pipeline_cache helper drops cached state and is safe
      to call when nothing is cached (no AttributeError or KeyError).
    """

    def setUp(self):
        # Each test starts with an empty cache so failures cannot leak across
        # tests via the module-level dict.
        from scripts import llm_utils
        llm_utils._PIPELINE_CACHE.clear()

    def tearDown(self):
        from scripts import llm_utils
        llm_utils._PIPELINE_CACHE.clear()

    def test_get_text_generation_pipeline_returns_cached_entry_without_reload(self):
        """Cache hit on the same model id must return the same pipeline
        object without invoking transformers.pipeline a second time.

        We pre-populate the cache so the test does not require the real
        transformers import to be patchable from here (the function uses a
        lazy `from transformers import pipeline` import, which is awkward
        to intercept reliably across Python versions/platforms)."""
        from scripts import llm_utils

        sentinel_pipeline = mock.MagicMock(name="cached-pipeline")
        llm_utils._PIPELINE_CACHE["model-A"] = sentinel_pipeline

        # Sanity guard: if we accidentally fell through to importing
        # transformers, this fake import would explode.
        import sys
        fake_transformers = mock.MagicMock(name="should-not-be-called")
        fake_transformers.pipeline.side_effect = AssertionError(
            "transformers.pipeline must not be called on a cache hit",
        )
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            result = llm_utils._get_text_generation_pipeline("model-A")

        self.assertIs(result, sentinel_pipeline)
        # Cache must still hold exactly the one entry we seeded.
        self.assertEqual(set(llm_utils._PIPELINE_CACHE.keys()), {"model-A"})

    def test_get_text_generation_pipeline_evicts_previous_model_on_miss(self):
        """Loading a different model id must evict the previously cached
        pipeline and call _release_torch_memory before allocating the new
        one — that is the whole point of the single-slot cache pattern."""
        from scripts import llm_utils

        prior_pipeline = mock.MagicMock(name="prior-pipeline")
        new_pipeline = mock.MagicMock(name="new-pipeline")
        llm_utils._PIPELINE_CACHE["model-A"] = prior_pipeline

        # Stub the lazy `from transformers import pipeline` import by
        # injecting a fake module into sys.modules.
        import sys
        fake_transformers = mock.MagicMock(name="transformers-stub")
        fake_transformers.pipeline.return_value = new_pipeline

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with mock.patch.object(llm_utils, "_release_torch_memory") as release_mock:
                result = llm_utils._get_text_generation_pipeline("model-B")

        self.assertIs(result, new_pipeline)
        self.assertEqual(set(llm_utils._PIPELINE_CACHE.keys()), {"model-B"})
        # The torch cleanup helper must run during eviction so VRAM held by
        # the prior pipeline becomes recoverable.
        release_mock.assert_called_once()
        fake_transformers.pipeline.assert_called_once()

    def test_clear_pipeline_cache_drops_cached_pipeline(self):
        from scripts import llm_utils

        llm_utils._PIPELINE_CACHE["model-A"] = mock.MagicMock(name="cached")

        with mock.patch.object(llm_utils, "_release_torch_memory") as release_mock:
            llm_utils._clear_pipeline_cache()

        self.assertEqual(llm_utils._PIPELINE_CACHE, {})
        release_mock.assert_called_once()

    def test_clear_pipeline_cache_is_safe_when_empty(self):
        from scripts import llm_utils

        # Calling clear on an empty cache must not raise; it should still ask
        # torch to release any lingering CUDA blocks (cheap best-effort op).
        with mock.patch.object(llm_utils, "_release_torch_memory") as release_mock:
            llm_utils._clear_pipeline_cache()

        self.assertEqual(llm_utils._PIPELINE_CACHE, {})
        release_mock.assert_called_once()

    def test_release_torch_memory_is_no_op_without_torch(self):
        from scripts import llm_utils

        # When torch is not importable the helper must swallow the ImportError
        # silently — it is a best-effort cleanup, not a hard requirement.
        import builtins
        original_import = builtins.__import__

        def _raising_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_raising_import):
            try:
                llm_utils._release_torch_memory()
            except Exception as exc:  # pragma: no cover - safety net
                self.fail(f"_release_torch_memory raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
