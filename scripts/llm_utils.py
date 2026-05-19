"""Utilities for local LLM-assisted DV name deduction.

These helpers keep inference optional and offline-first by relying on local
`transformers` models when available. They can also enrich prompts with project
context extracted from README files and PDFs found in a source folder.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import httpx
import yaml

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL_CANDIDATES = [
    "google/gemma-4-E4B-it",
    "google/gemma-4-E2B-it",
]

# Approximate memory recommendations (FP16-ish inference envelope, incl. headroom).
MODEL_MIN_MEMORY_GB = {
    "google/gemma-4-E4B-it": 18.0,
    "google/gemma-4-E2B-it": 10.0
}
CUDA_MEMORY_HEADROOM_FACTOR = 0.9

README_PATTERNS = ("README", "README.md", "README.txt", "readme.md", "readme.txt")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
PDF_LINK_PATTERN = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
DEFAULT_HTTP_TIMEOUT_S = 12
PDF_SECTION_HINTS = (
    "abstract",
    "introduction",
    "background",
    "method",
    "methods",
    "methodology",
    "study",
    "experiment",
    "participants",
    "materials",
    "measures",
    "measurements",
    "dependent variable",
    "dependent variables",
    "outcome",
    "outcomes",
    "results",
    "discussion",
    "questionnaire",
    "survey",
    "procedure",
    "evaluation",
)
PDF_MEASUREMENT_HINTS = (
    "measure",
    "metric",
    "variable",
    "rating",
    "score",
    "scale",
    "likert",
    "questionnaire",
    "survey",
    "response time",
    "completion time",
    "reaction time",
    "duration",
    "accuracy",
    "performance",
    "trust",
    "workload",
    "usability",
    "satisfaction",
    "acceptance",
    "mental demand",
    "effort",
    "comfort",
    "preference",
    "confidence",
    "nasa-tlx",
    "sus",
)
PDF_LOW_VALUE_HINTS = (
    "references",
    "bibliography",
    "acknowledgment",
    "acknowledgement",
)
DEFAULT_DV_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        logger.debug("pypdf not available; skipping PDF extraction for %s", path)
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        logger.warning("Failed to read PDF '%s': %s", path, exc)
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
    dropped_chars = len(text) - max_chars + 3
    logger.info(
        "Truncating context text from %d to %d chars (%d chars dropped).",
        len(text), max_chars, dropped_chars,
    )
    return text[: max_chars - 3] + "..."


def _normalize_multiline_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _split_long_block(block: str, target_chars: int = 900) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", block).strip())
        if sentence.strip()
    ]
    if not sentences:
        return []

    if len(sentences) == 1:
        words = sentences[0].split()
        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0
        for word in words:
            word_len = len(word) + (1 if current_words else 0)
            if current_words and current_len + word_len > target_chars:
                chunks.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len += word_len
        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    chunks = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence) + (1 if current else 0)
        if current and current_len + sentence_len > target_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += sentence_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_text_into_blocks(text: str, target_chars: int = 900) -> list[str]:
    normalized = _normalize_multiline_text(text)
    if not normalized:
        return []

    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]
    if not raw_blocks:
        return []

    blocks: list[str] = []
    max_block_chars = int(target_chars * 1.35)
    for block in raw_blocks:
        compact = re.sub(r"\s+", " ", block).strip()
        if not compact:
            continue
        if len(compact) <= max_block_chars:
            blocks.append(compact)
            continue
        blocks.extend(_split_long_block(compact, target_chars=target_chars))
    return blocks


def _score_pdf_block(block: str, index: int) -> int:
    lower = block.lower()
    score = 0

    if index == 0:
        score += 1

    if len(block) <= 140 and any(hint in lower for hint in PDF_SECTION_HINTS):
        score += 4

    score += sum(3 for hint in PDF_SECTION_HINTS if hint in lower)
    score += sum(1 for hint in PDF_MEASUREMENT_HINTS if hint in lower)

    if re.search(r"\b(n\s*=\s*\d+|participants?|questionnaire|survey|likert|scale|measures?)\b", lower):
        score += 3
    if re.search(r"\b(dependent variable|outcome|metric|score|rating|time|duration|accuracy)\b", lower):
        score += 2

    if any(hint in lower for hint in PDF_LOW_VALUE_HINTS):
        score -= 4
    if re.search(r"\bet al\.\b", lower) and lower.count(";") >= 3:
        score -= 2

    return score


def _select_pdf_context_excerpt(text: str, max_chars: int = 3000) -> str:
    """Prefer measurement-relevant PDF sections, else fall back to full text."""
    normalized = _normalize_multiline_text(text)
    if not normalized:
        return ""

    blocks = _split_text_into_blocks(normalized)
    if not blocks:
        return _truncate_text(normalized, max_chars)

    ranked = sorted(
        ((_score_pdf_block(block, index), index, block) for index, block in enumerate(blocks)),
        key=lambda item: (-item[0], item[1]),
    )
    positive = [item for item in ranked if item[0] > 0]
    if not positive:
        return _truncate_text(normalized, max_chars)

    selected: list[tuple[int, str]] = []
    skipped_blocks = 0
    used_chars = 0
    for _, index, block in positive:
        separator_len = 2 if selected else 0
        projected_len = used_chars + separator_len + len(block)
        if selected and projected_len > max_chars:
            skipped_blocks += 1
            continue
        selected.append((index, block))
        used_chars = projected_len
        if used_chars >= max_chars or len(selected) >= 4:
            skipped_blocks += len(positive) - len(selected) - skipped_blocks
            break

    if skipped_blocks > 0:
        logger.info(
            "PDF context selection: used %d of %d relevant blocks (%d skipped due to size limit).",
            len(selected), len(positive), skipped_blocks,
        )

    if not selected:
        return _truncate_text(normalized, max_chars)

    selected.sort(key=lambda item: item[0])
    return _truncate_text("\n\n".join(block for _, block in selected), max_chars)


def _dedupe_case_insensitive(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(text)
    return ordered


@lru_cache(maxsize=1)
def _load_candidate_schema_metadata(schema_path: str = str(DEFAULT_DV_SCHEMA_PATH)) -> dict[str, dict[str, object]]:
    path = Path(schema_path)
    if not path.is_file():
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}

    metadata: dict[str, dict[str, object]] = {}
    for entry in data.get("dvs", []):
        if not isinstance(entry, dict):
            continue
        canonical = str(entry.get("id", "")).strip()
        if not canonical:
            continue
        measurement = entry.get("measurement") if isinstance(entry.get("measurement"), dict) else {}
        metadata[canonical] = {
            "label": str(entry.get("label", "")).strip(),
            "cluster": str(entry.get("cluster", "")).strip(),
            "category": str(measurement.get("category", "")).strip(),
            "direction": str(measurement.get("direction", "")).strip(),
            "aliases": [str(alias).strip() for alias in (entry.get("aliases") or []) if str(alias).strip()],
            "instruments": [
                str(instrument).strip()
                for instrument in entry.get("instruments", [])
                if str(instrument).strip()
            ],
            "notes": str(entry.get("notes", "")).strip(),
        }
    return metadata


def _format_candidate_reference(candidates: Iterable[str]) -> str:
    metadata = _load_candidate_schema_metadata()
    lines: list[str] = []

    for candidate in candidates:
        record = metadata.get(candidate)
        if not record:
            lines.append(f"- {candidate}")
            continue

        parts = [candidate]
        label = str(record.get("label", "")).strip()
        cluster = str(record.get("cluster", "")).strip()
        category = str(record.get("category", "")).strip()
        direction = str(record.get("direction", "")).strip()
        aliases = [str(alias) for alias in record.get("aliases", [])][:3]
        instruments = [str(instrument) for instrument in record.get("instruments", [])][:2]
        notes = _truncate_text(str(record.get("notes", "")).strip(), 120) if record.get("notes") else ""

        if label:
            parts.append(f"label={label}")
        if cluster:
            parts.append(f"cluster={cluster}")
        if category:
            parts.append(f"category={category}")
        if direction:
            parts.append(f"direction={direction}")
        if aliases:
            parts.append(f"aliases={', '.join(aliases)}")
        if instruments:
            parts.append(f"instruments={', '.join(instruments)}")
        if notes:
            parts.append(f"notes={notes}")

        lines.append("- " + " | ".join(parts))

    return "\n".join(lines)


def _summarize_items(label: str, values: Iterable[str], max_items: int = 4) -> str | None:
    items = _dedupe_case_insensitive(values)
    if not items:
        return None

    preview = items[:max_items]
    suffix = f", +{len(items) - max_items} more" if len(items) > max_items else ""
    return f"{label}: {', '.join(preview)}{suffix}"


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
    headers = {
        "User-Agent": "OpenDV-HCI/1.0 (+https://github.com)",
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        return response.content, content_type


def _resolve_pdf_url(base_url: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    return urljoin(base_url, href)


def _fetch_pdf_text_from_url(url: str, max_chars: int = 3000) -> str:
    try:
        landing_payload, landing_content_type = _http_get(url)
    except Exception:
        return ""

    if "application/pdf" in landing_content_type.lower():
        return _select_pdf_context_excerpt(
            _extract_text_from_pdf_bytes(landing_payload),
            max_chars=max_chars,
        )

    landing_text = landing_payload.decode("utf-8", errors="ignore")
    candidates = PDF_LINK_PATTERN.findall(landing_text)

    for href in candidates[:5]:
        pdf_url = _resolve_pdf_url(url, href)
        try:
            pdf_payload, pdf_content_type = _http_get(pdf_url)
        except Exception:
            continue
        if "application/pdf" not in pdf_content_type.lower() and not pdf_url.lower().endswith(".pdf"):
            continue
        text = _extract_text_from_pdf_bytes(pdf_payload)
        if text:
            return _select_pdf_context_excerpt(text, max_chars=max_chars)

    return ""


def _fetch_pdf_text_from_doi(doi: str, max_chars: int = 3000) -> str:
    return _fetch_pdf_text_from_url(f"https://doi.org/{doi}", max_chars=max_chars)


def collect_repository_context(
    source_root: str | Path,
    max_chars: int = 6000,
    explicit_dois: Iterable[str] | None = None,
    explicit_pdf_urls: Iterable[str] | None = None,
    extra_context: Iterable[str] | None = None,
) -> str:
    """Collect textual context from README + PDFs in a folder tree.

    Also attempts DOI-based online PDF retrieval when a DOI appears in local
    context so model prompting can include manuscript context where possible.

    Args:
        source_root: Folder to scan.
        max_chars: Maximum total characters returned.
        explicit_dois: DOI values supplied externally (e.g., manifest metadata).
        explicit_pdf_urls: PDF URLs supplied externally.
        extra_context: Additional free-text context to pass to the prompt.
    """
    root = Path(source_root)
    snippets: list[str] = []
    readme_names: list[str] = []
    local_pdf_names: list[str] = []
    fetched_doi_pdfs: list[str] = []
    fetched_explicit_pdfs: list[str] = []
    supplied_notes = _dedupe_case_insensitive(extra_context or [])

    if supplied_notes:
        snippets.append("[SOURCE_CONTEXT]\n" + "\n".join(supplied_notes))

    if root.exists() and root.is_dir():
        # Prefer top-level README variants first for concise project context.
        for name in README_PATTERNS:
            candidate = root / name
            if candidate.is_file():
                content = _safe_read_text(candidate)
                if content:
                    readme_names.append(candidate.name)
                    snippets.append(f"[README:{candidate.name}]\n{content}")
                break

        # Include PDFs from top-level and one level deep (common artifact layout).
        pdf_files = sorted({*root.glob("*.pdf"), *root.glob("*/*.pdf")})
        for pdf in pdf_files:
            content = _select_pdf_context_excerpt(_extract_text_from_pdf(pdf))
            if content:
                local_pdf_names.append(str(pdf.relative_to(root).as_posix()))
                snippets.append(f"[PDF:{pdf.name}]\n{content}")

    discovered_dois: list[str] = []
    for candidate in explicit_dois or []:
        discovered_dois.extend(_extract_dois(str(candidate)))
    discovered_dois.extend(_extract_dois("\n\n".join(snippets)))
    discovered_dois = _dedupe_case_insensitive(discovered_dois)

    for doi in discovered_dois[:3]:
        remote_pdf_text = _fetch_pdf_text_from_doi(doi)
        if remote_pdf_text:
            fetched_doi_pdfs.append(doi)
            snippets.append(f"[DOI_PDF:{doi}]\n{remote_pdf_text}")

    explicit_pdf_urls = _dedupe_case_insensitive(explicit_pdf_urls or [])
    for pdf_url in explicit_pdf_urls[:3]:
        remote_pdf_text = _fetch_pdf_text_from_url(pdf_url)
        if remote_pdf_text:
            fetched_explicit_pdfs.append(pdf_url)
            snippets.append(f"[REMOTE_PDF:{pdf_url}]\n{remote_pdf_text}")

    summary_lines = ["[CONTEXT_SUMMARY]"]
    for line in (
        _summarize_items("Source notes", supplied_notes, max_items=2),
        _summarize_items("README files", readme_names),
        _summarize_items("Local PDFs", local_pdf_names),
        _summarize_items("Detected DOIs", discovered_dois),
        _summarize_items("Fetched DOI PDFs", fetched_doi_pdfs),
        _summarize_items("Explicit PDF URLs", explicit_pdf_urls, max_items=2),
        _summarize_items("Fetched explicit PDFs", fetched_explicit_pdfs, max_items=2),
    ):
        if line:
            summary_lines.append(line)

    if len(summary_lines) == 1 and not snippets:
        return ""

    summary_text = "\n".join(summary_lines)
    if not snippets:
        return _truncate_text(summary_text, max_chars=max_chars)

    remaining_chars = max_chars - len(summary_text) - 2
    if remaining_chars <= 0:
        return _truncate_text(summary_text, max_chars=max_chars)

    body_text = _truncate_text("\n\n".join(snippets), max_chars=remaining_chars)
    return f"{summary_text}\n\n{body_text}"


# Single-slot pipeline cache. We deliberately do NOT use lru_cache(maxsize>1)
# here: each cached transformers pipeline pins gigabytes of model weights in
# RAM/VRAM and never releases them. On a 32 GB host, holding two candidate
# models simultaneously (e.g. two gemma-4 variants tried in fallback order)
# is enough to OOM. Instead we keep at most one pipeline live and explicitly
# release the previous one when a different model is requested.
_PIPELINE_CACHE: dict[str, Any] = {}


def _release_torch_memory() -> None:
    """Best-effort release of CUDA memory held by torch tensors.

    No-op when torch or CUDA is unavailable.
    """
    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must never raise
        pass


def _clear_pipeline_cache() -> None:
    """Drop the cached generation pipeline and reclaim its memory.

    Public for callers that finish a batch run and want to free model RAM
    before kicking off subsequent work. Safe to call when nothing is cached.
    """
    if not _PIPELINE_CACHE:
        _release_torch_memory()
        return
    _PIPELINE_CACHE.clear()
    gc.collect()
    _release_torch_memory()


def _get_text_generation_pipeline(model_name: str):
    """Return a (single-slot cached) text-generation pipeline for `model_name`.

    Loading a new model evicts and frees the previously cached pipeline
    before instantiating the next one — see `_PIPELINE_CACHE` comment for
    rationale. The cache key is the model id; the function signature is
    preserved for backwards-compatible mocking in tests.
    """
    cached = _PIPELINE_CACHE.get(model_name)
    if cached is not None:
        return cached

    # Different model requested — drop the previous pipeline first so its
    # weights are eligible for GC before we allocate the next one.
    if _PIPELINE_CACHE:
        _PIPELINE_CACHE.clear()
        gc.collect()
        _release_torch_memory()

    from transformers import pipeline  # type: ignore

    new_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device_map="auto",
    )
    _PIPELINE_CACHE[model_name] = new_pipeline
    return new_pipeline


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

    candidate_reference = _format_candidate_reference(candidates)
    prompt = (
        "You standardize HCI dependent variable columns into canonical identifiers.\n"
        "Choose exactly one identifier from the candidate list. Do not invent new identifiers.\n\n"
        "Decision rules:\n"
        "1. Match the measured human outcome or construct, not superficial token overlap.\n"
        "2. Use README, DOI, PDF, questionnaire, and instrument evidence when available.\n"
        "3. Prefer construct-level outcomes such as trust, workload, usability, safety, time, accuracy, or acceptance.\n"
        "4. Treat identifiers, timestamps, frame counters, coordinates, raw sensor channels, scenario codes, and logging fields as weak evidence. "
        "Do not map them to a candidate unless the source context clearly says they operationalize that construct.\n"
        "5. A clock timestamp is not a task duration unless the study context explicitly says it is an outcome measure.\n"
        "6. If multiple candidates seem plausible, choose the one best supported by construct wording, instrument names, and manuscript context.\n\n"
        f"Raw column name: {raw_column_name}\n"
        "Candidate reference:\n"
        f"{candidate_reference}\n"
    )
    if context:
        prompt += (
            "\nSource context:\n"
            "Use the evidence below to infer what the column actually measures. "
            "Pay special attention to DOI, PDF, manuscript, README, instrument, and measure descriptions.\n"
            f"\nRepository context:\n{context}\n"
        )
    prompt += "\nRespond with only one canonical identifier from the candidate reference above."

    for model_name in model_list:
        try:
            generator = _get_text_generation_pipeline(model_name)
            tokenizer = getattr(generator, "tokenizer", None)
            chat_template = getattr(tokenizer, "chat_template", None) if tokenizer else None
            if chat_template:
                model_input = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                model_input = prompt
            output = generator(
                model_input,
                max_new_tokens=64,
                do_sample=False,
                return_full_text=False,
            )
            text = output[0].get("generated_text", "")
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM deduction failed for model '%s': %s", model_name, exc)
            continue

        answer = _extract_candidate_from_llm_output(text, candidates)
        if answer:
            logger.debug(
                "LLM deduction: '%s' -> '%s' (model=%s, raw=%r)",
                raw_column_name, answer, model_name, text[:120],
            )
            return answer
        logger.debug(
            "LLM deduction produced no matching candidate for '%s' (model=%s, raw=%r)",
            raw_column_name, model_name, text[:160],
        )

    return None


def _extract_candidate_from_llm_output(text: str, candidates: list[str]) -> str | None:
    """Find the best candidate ID inside an LLM response.

    Models often wrap the answer ("Best match: `trust_rating`", "The answer is
    trust_rating because..."), so substring search beats first-line equality.
    When several candidates appear, prefer the longest match — it is the most
    specific and least likely to be a token-overlap artefact (e.g. ``trust`` vs
    ``trust_rating``).
    """
    if not text:
        return None

    body = text.strip()
    if not body:
        return None

    body_lower = body.lower()
    matches: list[tuple[int, str]] = []
    for candidate in candidates:
        needle = candidate.lower()
        if not needle:
            continue
        position = body_lower.find(needle)
        if position != -1:
            matches.append((len(candidate), candidate))

    if matches:
        matches.sort(key=lambda item: (-item[0], item[1]))
        return matches[0][1]

    first_line = body.splitlines()[0].strip().strip("` ")
    if first_line in candidates:
        return first_line

    normalized_answer = re.sub(r"[^a-z0-9_]+", "", first_line.lower())
    for candidate in candidates:
        if normalized_answer == re.sub(r"[^a-z0-9_]+", "", candidate.lower()):
            return candidate

    return None

