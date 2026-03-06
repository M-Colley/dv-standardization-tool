"""Utilities for local LLM-assisted DV name deduction.

These helpers keep inference optional and offline-first by relying on local
`transformers` models when available. They can also enrich prompts with project
context extracted from README files and PDFs found in a source folder.
"""

from __future__ import annotations

import io
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib import request

DEFAULT_LOCAL_MODEL_CANDIDATES = [
    "Qwen/Qwen3.5-4B",
    "microsoft/Phi-4-mini-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

# Approximate memory recommendations (FP16-ish inference envelope, incl. headroom).
MODEL_MIN_MEMORY_GB = {
    "Qwen/Qwen3.5-4B": 10.0,
    "microsoft/Phi-4-mini-instruct": 8.0,
    "meta-llama/Llama-3.2-3B-Instruct": 8.0,
    "Qwen/Qwen2.5-3B-Instruct": 8.0,
    "Qwen/Qwen2.5-1.5B-Instruct": 4.0,
}
CUDA_MEMORY_HEADROOM_FACTOR = 0.9

README_PATTERNS = ("README", "README.md", "README.txt", "readme.md", "readme.txt")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
PDF_LINK_PATTERN = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
DEFAULT_HTTP_TIMEOUT_S = 12


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def _extract_text_from_pdf_bytes(pdf_content: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(io.BytesIO(pdf_content))
    except Exception:
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def _truncate_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _detect_available_memory_gb() -> float | None:
    """Best-effort memory detection for local model suitability checks."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return float(mem)
    except Exception:
        pass

    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        return None


def _detect_cuda_memory_gb() -> float | None:
    """Best-effort CUDA memory detection for strict GPU-fit checks."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return float(mem)
    except Exception:
        return None
    return None


def _estimate_model_min_memory_gb(model_name: str) -> float:
    """Estimate minimum memory for unknown model ids from a common `xB` pattern."""
    known = MODEL_MIN_MEMORY_GB.get(model_name)
    if known is not None:
        return known

    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    if not match:
        return 8.0

    params_in_billions = float(match.group(1))
    return max(4.0, params_in_billions * 2.5)


def select_local_model_candidates(
    preferred_models: list[str] | None = None,
) -> list[str]:
    """Return candidate models filtered by available memory.

    Override with `DV_LLM_MODELS` (comma-separated model ids) for explicit control.
    """
    override = os.getenv("DV_LLM_MODELS", "").strip()
    candidates = (
        [m.strip() for m in override.split(",") if m.strip()]
        if override
        else (preferred_models or DEFAULT_LOCAL_MODEL_CANDIDATES)
    )
    if not candidates:
        return []

    cuda_memory = _detect_cuda_memory_gb()
    available_memory = (
        cuda_memory * CUDA_MEMORY_HEADROOM_FACTOR
        if cuda_memory is not None
        else _detect_available_memory_gb()
    )
    if available_memory is None:
        # Unknown machine profile: keep full ordered fallback list.
        return list(candidates)

    filtered = [
        model_name
        for model_name in candidates
        if available_memory >= _estimate_model_min_memory_gb(model_name)
    ]
    return filtered


def _extract_dois(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in DOI_PATTERN.findall(text or ""):
        cleaned = match.strip().rstrip(".,);]")
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(cleaned)
    return ordered


def _http_get(url: str, timeout_s: int = DEFAULT_HTTP_TIMEOUT_S) -> tuple[bytes, str]:
    req = request.Request(
        url,
        headers={
            "User-Agent": "OpenDV-HCI/1.0 (+https://github.com)",
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        },
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        content_type = resp.headers.get("Content-Type", "")
        payload = resp.read()
    return payload, content_type


def _resolve_pdf_url(base_url: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    return request.urljoin(base_url, href)


def _fetch_pdf_text_from_doi(doi: str, max_chars: int = 3000) -> str:
    doi_url = f"https://doi.org/{doi}"

    try:
        landing_payload, landing_content_type = _http_get(doi_url)
    except Exception:
        return ""

    if "application/pdf" in landing_content_type.lower():
        return _truncate_text(_extract_text_from_pdf_bytes(landing_payload), max_chars)

    landing_text = landing_payload.decode("utf-8", errors="ignore")
    candidates = PDF_LINK_PATTERN.findall(landing_text)

    for href in candidates[:5]:
        pdf_url = _resolve_pdf_url(doi_url, href)
        try:
            pdf_payload, pdf_content_type = _http_get(pdf_url)
        except Exception:
            continue
        if "application/pdf" not in pdf_content_type.lower() and not pdf_url.lower().endswith(".pdf"):
            continue
        text = _extract_text_from_pdf_bytes(pdf_payload)
        if text:
            return _truncate_text(text, max_chars)

    return ""


def collect_repository_context(source_root: str | Path, max_chars: int = 6000) -> str:
    """Collect textual context from README + PDFs in a folder tree.

    Also attempts DOI-based online PDF retrieval when a DOI appears in local
    context so model prompting can include manuscript context where possible.

    Args:
        source_root: Folder to scan.
        max_chars: Maximum total characters returned.
    """
    root = Path(source_root)
    if not root.exists() or not root.is_dir():
        return ""

    snippets: list[str] = []

    # Prefer top-level README variants first for concise project context.
    for name in README_PATTERNS:
        candidate = root / name
        if candidate.is_file():
            content = _safe_read_text(candidate)
            if content:
                snippets.append(f"[README:{candidate.name}]\n{content}")
            break

    # Include PDFs from top-level and one level deep (common artifact layout).
    pdf_files = sorted({*root.glob("*.pdf"), *root.glob("*/*.pdf")})
    for pdf in pdf_files:
        content = _extract_text_from_pdf(pdf)
        if content:
            snippets.append(f"[PDF:{pdf.name}]\n{content}")

    doi_search_text = "\n\n".join(snippets)
    for doi in _extract_dois(doi_search_text)[:3]:
        remote_pdf_text = _fetch_pdf_text_from_doi(doi)
        if remote_pdf_text:
            snippets.append(f"[DOI_PDF:{doi}]\n{remote_pdf_text}")

    if not snippets:
        return ""

    return _truncate_text("\n\n".join(snippets), max_chars=max_chars)


@lru_cache(maxsize=4)
def _get_text_generation_pipeline(model_name: str):
    from transformers import pipeline  # type: ignore

    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device_map="auto",
    )


def deduce_standard_name_with_local_llm(
    raw_column_name: str,
    canonical_candidates: Iterable[str],
    source_root: str | Path | None = None,
    preferred_models: list[str] | None = None,
    repository_context: str | None = None,
) -> str | None:
    """Infer the most appropriate canonical DV id from a local LLM.

    Returns one candidate from `canonical_candidates` when possible.
    """
    candidates = [str(c).strip() for c in canonical_candidates if str(c).strip()]
    if not raw_column_name or not candidates:
        return None

    model_list = select_local_model_candidates(preferred_models=preferred_models)
    context = repository_context
    if context is None:
        context = collect_repository_context(source_root) if source_root else ""

    prompt = (
        "You standardize dependent variable names for HCI datasets. "
        "Choose exactly one canonical identifier from the provided list.\n\n"
        f"Raw column name: {raw_column_name}\n"
        f"Canonical identifiers: {', '.join(candidates)}\n"
    )
    if context:
        prompt += f"\nRepository context (README/PDF excerpts):\n{context}\n"
    prompt += "\nRespond with only one canonical identifier from the list."

    for model_name in model_list:
        try:
            generator = _get_text_generation_pipeline(model_name)
            output = generator(prompt, max_new_tokens=24, do_sample=False)
            text = output[0].get("generated_text", "")
        except Exception:
            continue

        tail = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        answer = tail.splitlines()[0].strip().strip("` ")

        if answer in candidates:
            return answer

        lowered_map = {c.lower(): c for c in candidates}
        normalized = re.sub(r"[^a-z0-9_]+", "", answer.lower())
        for key, original in lowered_map.items():
            if normalized == re.sub(r"[^a-z0-9_]+", "", key):
                return original

    return None

