import importlib.util
import time
from pathlib import Path

import pytest


PLUGIN_PATH = Path("/Users/jeffphoon/.hermes/hermes-agent/plugins/memory/hermes-memory/__init__.py")
spec = importlib.util.spec_from_file_location("hermes_memory_provider", PLUGIN_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
HermesMemoryProvider = module.HermesMemoryProvider


@pytest.fixture()
def provider(monkeypatch, tmp_path):
    repo = tmp_path / "demo-project"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(repo))
    monkeypatch.setenv("HERMES_MEMORY_BASE_URL", "http://127.0.0.1:8790")
    p = HermesMemoryProvider()
    p.initialize("session-123", platform="telegram", user_id="jeff")
    return p


def test_provider_rejects_remote_base_url_by_default(monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_BASE_URL", "http://example.com:8790")
    monkeypatch.delenv("HERMES_MEMORY_ALLOW_REMOTE", raising=False)
    p = HermesMemoryProvider()
    assert p.is_available() is False


def test_prefetch_uses_project_user_global_scopes(provider, monkeypatch):
    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"context": "remembered deployment constraint"}

    monkeypatch.setattr(provider, "_post_json", fake_post)
    result = provider.prefetch("deployment history")

    assert captured["path"] == "/memories/recall"
    assert captured["payload"]["scopes"][0].startswith("project:demo-project-")
    assert captured["payload"]["scopes"][1:] == ["user:jeff", "global"]
    assert "remembered deployment constraint" in result


def test_queue_prefetch_populates_cache(provider, monkeypatch):
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"context": "cached memory"}

    monkeypatch.setattr(provider, "_post_json", fake_post)
    provider.queue_prefetch("deployment history")
    provider._prefetch_thread.join(timeout=2)
    result = provider.prefetch("deployment history")

    assert calls[0][0] == "/memories/recall"
    assert "cached memory" in result


def test_on_memory_write_mirrors_user_memory_to_user_scope(provider, monkeypatch):
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"created": 1}

    monkeypatch.setattr(provider, "_post_json", fake_post)
    provider.on_memory_write("add", "user", "User prefers concise Chinese replies.")

    assert calls
    path, payload = calls[0]
    assert path == "/memories/ingest"
    assert payload["scope"] == "user:jeff"
    assert payload["items"][0]["kind"] == "preference"


def test_non_primary_context_skips_writes(monkeypatch, tmp_path):
    repo = tmp_path / "demo-project"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(repo))
    p = HermesMemoryProvider()
    p.initialize("session-123", platform="cli", user_id="jeff", agent_context="cron")
    calls = []
    monkeypatch.setattr(p, "_post_json", lambda path, payload: calls.append((path, payload)) or {"created": 1})

    p.on_memory_write("add", "user", "User prefers concise Chinese replies.")
    p.sync_turn("hi", "Verified fix: deployment now requires CHANGELOG update.")
    time.sleep(0.1)
    assert calls == []


def test_sync_turn_only_ingests_durable_content(provider, monkeypatch):
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"created": 1}

    monkeypatch.setattr(provider, "_post_json", fake_post)
    provider.sync_turn("hi", "Thanks!")
    assert calls == []

    provider.sync_turn("what changed", "Verified fix: deployment now requires CHANGELOG update before rollout.")
    provider._sync_thread.join(timeout=2)
    assert len(calls) == 1
    assert calls[0][0] == "/memories/ingest"
    assert calls[0][1]["scope"].startswith("project:demo-project-")
    assert calls[0][1]["items"][0]["kind"] == "solution"
