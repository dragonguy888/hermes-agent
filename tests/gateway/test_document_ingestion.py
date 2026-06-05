"""Tests for MarkItDown-backed inbound document ingestion."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import cast

import pytest

from gateway.document_ingestion import (
    convert_document_attachment,
    should_use_markitdown,
)
from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource
from gateway.run import GatewayConfig, GatewayRunner


class _FakeMarkItDown:
    calls: list[str] = []
    output = "# Converted\n\nhello from document"
    fail: Exception | None = None

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def convert_local(self, path: str):
        self.__class__.calls.append(path)
        if self.__class__.fail:
            raise self.__class__.fail
        return types.SimpleNamespace(markdown=self.__class__.output)

    def convert_uri(self, uri: str):  # pragma: no cover - must never be called
        raise AssertionError(f"convert_uri must not be called: {uri}")

    def convert(self, source):  # pragma: no cover - must never be called
        raise AssertionError(f"convert must not be called: {source}")


@pytest.fixture(autouse=True)
def fake_markitdown(monkeypatch):
    _FakeMarkItDown.calls = []
    _FakeMarkItDown.output = "# Converted\n\nhello from document"
    _FakeMarkItDown.fail = None
    module = types.ModuleType("markitdown")
    setattr(module, "MarkItDown", _FakeMarkItDown)
    monkeypatch.setitem(sys.modules, "markitdown", module)
    yield _FakeMarkItDown


def test_policy_uses_markitdown_for_supported_documents():
    use, reason = should_use_markitdown(path="report.pdf", mime_type="application/pdf")
    assert use is True
    assert reason == "supported document"


@pytest.mark.parametrize("filename", ["notes.txt", "data.json", "README.md", "table.csv"])
def test_policy_bypasses_plain_text(filename):
    use, reason = should_use_markitdown(path=filename, mime_type="text/plain")
    assert use is False
    assert reason == "plain text bypass"


def test_policy_disables_archives_by_default():
    use, reason = should_use_markitdown(path="bundle.zip", mime_type="application/zip")
    assert use is False
    assert reason == "archives disabled"


@pytest.mark.asyncio
async def test_convert_document_attachment_uses_convert_local_and_saves_markdown(tmp_path, fake_markitdown):
    source = tmp_path / "doc_abc_report.pdf"
    source.write_bytes(b"%PDF fake")

    result = await convert_document_attachment(
        path=str(source),
        agent_path="/agent/doc_abc_report.pdf",
        mime_type="application/pdf",
        config={"max_markdown_chars": 1000},
    )

    assert result.status == "converted"
    assert fake_markitdown.calls == [str(source)]
    assert "hello from document" in result.markdown
    assert result.markdown_path == str(source.with_suffix(".pdf.md"))
    assert Path(cast(str, result.markdown_path)).read_text(encoding="utf-8") == _FakeMarkItDown.output
    assert "convert_uri" not in result.prompt_block()
    assert "--- BEGIN MARKITDOWN: report.pdf ---" in result.prompt_block()


@pytest.mark.asyncio
async def test_convert_document_attachment_truncates_prompt_but_saves_full_markdown(tmp_path, fake_markitdown):
    source = tmp_path / "doc_abc_long.pdf"
    source.write_bytes(b"%PDF fake")
    fake_markitdown.output = "x" * 200

    result = await convert_document_attachment(
        path=str(source),
        agent_path=str(source),
        mime_type="application/pdf",
        config={"max_markdown_chars": 50},
    )

    assert result.status == "converted"
    assert result.truncated is True
    assert "[... MarkItDown output truncated ...]" in result.markdown
    assert Path(cast(str, result.markdown_path)).read_text(encoding="utf-8") == "x" * 200


@pytest.mark.asyncio
async def test_convert_document_attachment_degrades_on_failure(tmp_path, fake_markitdown):
    source = tmp_path / "doc_abc_broken.pdf"
    source.write_bytes(b"%PDF fake")
    fake_markitdown.fail = RuntimeError("boom")

    result = await convert_document_attachment(
        path=str(source),
        agent_path="/agent/broken.pdf",
        mime_type="application/pdf",
    )

    assert result.status == "failed"
    assert "boom" in result.reason
    assert "Automatic MarkItDown conversion failed" in result.prompt_block()
    assert "Ask the user what they'd like you to do" in result.prompt_block()


@pytest.mark.asyncio
async def test_prepare_inbound_message_includes_markitdown_for_document(tmp_path, monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=True)
    runner.adapters = {}
    setattr(runner, "_model", "test-model")
    setattr(runner, "_base_url", "")
    runner._has_setup_skill = lambda: False

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm")
    doc = tmp_path / "doc_abc_report.pdf"
    doc.write_bytes(b"%PDF fake")
    event = MessageEvent(
        text="请分析这个",
        source=source,
        message_type=MessageType.DOCUMENT,
        media_urls=[str(doc)],
        media_types=["application/pdf"],
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"attachments": {"markitdown": {"enabled": True}}})

    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert result is not None
    assert "It was converted to Markdown with MarkItDown" in result
    assert "--- BEGIN MARKITDOWN: report.pdf ---" in result
    assert "hello from document" in result


@pytest.mark.asyncio
async def test_prepare_inbound_message_keeps_text_documents_on_bypass(tmp_path, monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=True)
    runner.adapters = {}
    setattr(runner, "_model", "test-model")
    setattr(runner, "_base_url", "")
    runner._has_setup_skill = lambda: False

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm")
    doc = tmp_path / "doc_abc_notes.txt"
    doc.write_text("hello", encoding="utf-8")
    event = MessageEvent(
        text="看这个",
        source=source,
        message_type=MessageType.DOCUMENT,
        media_urls=[str(doc)],
        media_types=["text/plain"],
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"attachments": {"markitdown": {"enabled": True}}})

    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert result is not None
    assert "text document" in result
    assert "MarkItDown" not in result
