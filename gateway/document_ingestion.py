"""Safe inbound document-to-Markdown preprocessing for messaging attachments.

This module intentionally wraps MarkItDown as a *local cached-file* converter.
It never accepts user-supplied URLs/URIs, and callers should only pass files
that were already downloaded into Hermes' attachment cache.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_MB = 25
DEFAULT_MAX_MARKDOWN_CHARS = 60_000
DEFAULT_TIMEOUT_SECONDS = 45

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".log",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}

MARKITDOWN_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".epub",
    ".msg",
}

ARCHIVE_EXTENSIONS = {".zip"}


@dataclass(frozen=True)
class DocumentIngestionDecision:
    enabled: bool
    mode: str
    max_file_mb: int
    max_markdown_chars: int
    timeout_seconds: int
    save_markdown_cache: bool
    allow_archives: bool


@dataclass(frozen=True)
class DocumentIngestionResult:
    status: str  # converted | skipped | failed
    display_name: str
    original_path: str
    agent_path: str
    markdown: str = ""
    markdown_path: Optional[str] = None
    reason: str = ""
    truncated: bool = False

    def context_note(self) -> str:
        if self.status == "converted":
            md_path = f" Markdown cache: {self.markdown_path}." if self.markdown_path else ""
            trunc = " The included Markdown was truncated." if self.truncated else ""
            return (
                f"[The user sent a document: '{self.display_name}'. "
                f"It was converted to Markdown with MarkItDown. "
                f"Original file: {self.agent_path}.{md_path}{trunc} "
                "The converted content is included below.]"
            )
        if self.status == "failed":
            return (
                f"[The user sent a document: '{self.display_name}'. "
                f"The file is saved at: {self.agent_path}. "
                f"Automatic MarkItDown conversion failed: {self.reason}. "
                "Ask the user what they'd like you to do with it.]"
            )
        return (
            f"[The user sent a document: '{self.display_name}'. "
            f"The file is saved at: {self.agent_path}. "
            "Ask the user what they'd like you to do with it.]"
        )

    def prompt_block(self) -> str:
        note = self.context_note()
        if self.status != "converted":
            return note
        return (
            f"{note}\n\n"
            f"--- BEGIN MARKITDOWN: {self.display_name} ---\n"
            f"{self.markdown}\n"
            f"--- END MARKITDOWN: {self.display_name} ---"
        )


def _cfg_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _cfg_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def decision_from_config(config: Optional[Mapping[str, Any]]) -> DocumentIngestionDecision:
    cfg = dict(config or {})
    mode = str(cfg.get("mode", "auto") or "auto").strip().lower()
    if mode not in {"auto", "manual", "off"}:
        mode = "auto"
    enabled = _cfg_bool(cfg.get("enabled", True), True) and mode != "off"
    return DocumentIngestionDecision(
        enabled=enabled,
        mode=mode,
        max_file_mb=_cfg_int(cfg.get("max_file_mb"), DEFAULT_MAX_FILE_MB),
        max_markdown_chars=_cfg_int(cfg.get("max_markdown_chars"), DEFAULT_MAX_MARKDOWN_CHARS),
        timeout_seconds=_cfg_int(cfg.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS),
        save_markdown_cache=_cfg_bool(cfg.get("save_markdown_cache", True), True),
        allow_archives=_cfg_bool(cfg.get("allow_archives", False), False),
    )


def display_name_from_cache_path(path: str) -> str:
    basename = os.path.basename(path)
    parts = basename.split("_", 2)
    display = parts[2] if len(parts) >= 3 else basename
    return re.sub(r"[^\w.\- ]", "_", display)


def should_use_markitdown(
    *,
    path: str,
    mime_type: str = "",
    user_text: str = "",
    config: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, str]:
    """Return whether a cached local attachment should be converted."""
    decision = decision_from_config(config)
    if not decision.enabled:
        return False, "disabled"

    ext = Path(path).suffix.lower()
    mime = (mime_type or "").lower()

    if ext in TEXT_EXTENSIONS or mime.startswith("text/plain"):
        return False, "plain text bypass"
    if ext in ARCHIVE_EXTENSIONS and not decision.allow_archives:
        return False, "archives disabled"
    if decision.mode == "manual":
        trigger = " ".join((user_text or "").lower().split())
        if not any(word in trigger for word in ("read", "summar", "analy", "convert", "读取", "总结", "分析", "转换", "看看")):
            return False, "manual mode without request"

    if ext in MARKITDOWN_EXTENSIONS or (decision.allow_archives and ext in ARCHIVE_EXTENSIONS):
        return True, "supported document"

    guessed, _ = mimetypes.guess_type(path)
    guessed = (guessed or mime or "").lower()
    if guessed in {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/msword",
        "application/vnd.ms-powerpoint",
        "application/vnd.ms-excel",
        "text/html",
        "application/epub+zip",
    }:
        return True, "supported MIME"

    return False, "unsupported type"


def _load_markitdown_class():
    try:
        from markitdown import MarkItDown  # type: ignore
        return MarkItDown
    except ImportError as exc:
        try:
            from tools.lazy_deps import ensure

            ensure("attachment.markitdown")
            from markitdown import MarkItDown  # type: ignore
            return MarkItDown
        except Exception as lazy_exc:  # pragma: no cover - exact lazy failure text is env-dependent
            raise RuntimeError(f"MarkItDown is not installed: {lazy_exc}") from exc


def _convert_local_sync(path: str) -> str:
    MarkItDown = _load_markitdown_class()
    md = MarkItDown(enable_plugins=False)
    # Security invariant: only local cached paths reach this function. Do not
    # switch to convert()/convert_uri(); those accept URLs/file:/data: URIs.
    result = md.convert_local(path)
    markdown = getattr(result, "markdown", "")
    return str(markdown or "")


async def convert_document_attachment(
    *,
    path: str,
    agent_path: str,
    mime_type: str = "",
    user_text: str = "",
    config: Optional[Mapping[str, Any]] = None,
) -> DocumentIngestionResult:
    """Convert a single cached local document to Markdown when policy allows."""
    display = display_name_from_cache_path(path)
    should_convert, reason = should_use_markitdown(
        path=path,
        mime_type=mime_type,
        user_text=user_text,
        config=config,
    )
    if not should_convert:
        return DocumentIngestionResult("skipped", display, path, agent_path, reason=reason)

    decision = decision_from_config(config)
    local = Path(path)
    try:
        if not local.is_file():
            return DocumentIngestionResult("failed", display, path, agent_path, reason="local cached file not found")
        size = local.stat().st_size
    except OSError as exc:
        return DocumentIngestionResult("failed", display, path, agent_path, reason=f"cannot stat file: {exc}")

    max_bytes = decision.max_file_mb * 1024 * 1024
    if size > max_bytes:
        return DocumentIngestionResult(
            "skipped",
            display,
            path,
            agent_path,
            reason=f"file exceeds {decision.max_file_mb} MB limit",
        )

    try:
        markdown = await asyncio.wait_for(
            asyncio.to_thread(_convert_local_sync, str(local)),
            timeout=decision.timeout_seconds,
        )
    except asyncio.TimeoutError:
        return DocumentIngestionResult("failed", display, path, agent_path, reason="conversion timed out")
    except Exception as exc:
        logger.info("MarkItDown conversion failed for %s: %s", path, exc)
        return DocumentIngestionResult("failed", display, path, agent_path, reason=str(exc) or exc.__class__.__name__)

    if not markdown.strip():
        return DocumentIngestionResult("failed", display, path, agent_path, reason="empty Markdown output")

    markdown_path: Optional[str] = None
    if decision.save_markdown_cache:
        try:
            out = local.with_suffix(local.suffix + ".md")
            out.write_text(markdown, encoding="utf-8")
            try:
                from tools.credential_files import to_agent_visible_cache_path

                markdown_path = to_agent_visible_cache_path(str(out))
            except Exception:
                markdown_path = str(out)
        except OSError as exc:
            logger.debug("Failed to save MarkItDown cache for %s: %s", path, exc)

    truncated = False
    if len(markdown) > decision.max_markdown_chars:
        markdown = markdown[: decision.max_markdown_chars].rstrip() + "\n\n[... MarkItDown output truncated ...]"
        truncated = True

    return DocumentIngestionResult(
        "converted",
        display,
        path,
        agent_path,
        markdown=markdown,
        markdown_path=markdown_path,
        truncated=truncated,
    )
