import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from agent.memory_manager import MemoryManager

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugins" / "memory" / "hermes-memory" / "__init__.py"
SPEC = spec_from_file_location("hermes_memory_plugin", PLUGIN_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
_project_scope_from_cwd = MODULE._project_scope_from_cwd


DURABLE_KEYWORD_REPLY = (
    "Fixed the failing integration test suite after the deploy — the root cause was "
    "a stale cache entry that must be cleared before release. Redeployed the service "
    "to staging and verified 190/190 tests green across the full suite, which is "
    "required before this can ship to production. "
    "修复了失败的测试用例，部署到预发布环境后已验证 190/190 全部通过，"
    "这是发布前必须确认的约束条件。"
)


def _make_primary_provider():
    provider = MODULE.HermesMemoryProvider()
    provider._agent_context = "primary"
    provider._project_scope = "project:test-project"
    provider._session_id = "test-session"
    provider._platform = "cli"
    return provider


def _join_sync_thread(provider) -> None:
    thread = getattr(provider, "_sync_thread", None)
    if thread is not None:
        thread.join(timeout=2)


def _provider_with_post_capture(monkeypatch):
    provider = _make_primary_provider()
    calls = []

    def fake_post_json(path, payload):
        calls.append((path, payload))
        return {}

    monkeypatch.setattr(provider, "_post_json", fake_post_json)
    return provider, calls


# --- T1: sync_turn is a no-op -------------------------------------------------


def test_sync_turn_does_not_ingest_long_durable_keyword_response(monkeypatch):
    provider = _make_primary_provider()
    calls = []
    monkeypatch.setattr(provider, "_ingest", lambda **kwargs: calls.append(kwargs))

    provider.sync_turn(user_content="please check the tests", assistant_content=DURABLE_KEYWORD_REPLY, session_id="s1")
    _join_sync_thread(provider)

    assert calls == []


def test_sync_turn_makes_no_http_request(monkeypatch):
    provider = _make_primary_provider()
    calls = []
    monkeypatch.setattr(provider, "_post_json", lambda path, payload: calls.append((path, payload)))

    for content in (DURABLE_KEYWORD_REPLY, "just a short reply", ""):
        provider.sync_turn(user_content="", assistant_content=content, session_id="s1")
        _join_sync_thread(provider)

    assert calls == []


def test_sync_turn_writes_nothing_for_short_durable_statement(monkeypatch):
    provider = _make_primary_provider()
    calls = []
    monkeypatch.setattr(provider, "_ingest", lambda **kwargs: calls.append(kwargs))

    provider.sync_turn(
        user_content="",
        assistant_content="The Hermes memory service must bind to 127.0.0.1:8790 under launchd.",
        session_id="s1",
    )
    _join_sync_thread(provider)

    assert calls == []


def test_sync_turn_no_op_in_every_agent_context(monkeypatch):
    for ctx in ("primary", "subagent", "cron", "flush"):
        provider = _make_primary_provider()
        provider._agent_context = ctx
        calls = []
        monkeypatch.setattr(provider, "_ingest", lambda **kwargs: calls.append(kwargs))

        provider.sync_turn(user_content="", assistant_content=DURABLE_KEYWORD_REPLY, session_id="s1")
        _join_sync_thread(provider)

        assert calls == [], f"agent_context={ctx} unexpectedly ingested: {calls}"


def test_sync_turn_back_to_back_turns_write_nothing(monkeypatch):
    provider = _make_primary_provider()
    calls = []
    monkeypatch.setattr(provider, "_ingest", lambda **kwargs: calls.append(kwargs))

    provider.sync_turn(user_content="", assistant_content=DURABLE_KEYWORD_REPLY, session_id="s1")
    provider.sync_turn(user_content="", assistant_content=DURABLE_KEYWORD_REPLY, session_id="s1")
    _join_sync_thread(provider)

    assert calls == []


def test_looks_durable_removed():
    assert not hasattr(MODULE.HermesMemoryProvider, "_looks_durable")


def test_sync_thread_attribute_removed():
    provider = MODULE.HermesMemoryProvider()
    assert not hasattr(provider, "_sync_thread")


# --- T2: explicit-write provenance -------------------------------------------


def test_on_memory_write_add_still_ingests(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    provider.on_memory_write("add", "memory", "The runtime binds 127.0.0.1:8790 under launchd.")

    assert len(calls) == 1
    path, payload = calls[0]
    assert path == "/memories/ingest"
    assert payload["items"][0]["content"] == "The runtime binds 127.0.0.1:8790 under launchd."


def test_on_memory_write_replace_still_ingests(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    provider.on_memory_write("replace", "memory", "Updated durable fact.")

    assert len(calls) == 1
    assert calls[0][1]["items"][0]["content"] == "Updated durable fact."


def test_on_memory_write_uses_explicit_source_and_existing_trust(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    provider.on_memory_write("add", "memory", "Durable fact content.")

    item = calls[0][1]["items"][0]
    assert item["source"] == "hermes-explicit"
    assert item["importance"] == 0.85
    assert item["confidence"] == 0.9


def test_on_memory_write_user_target_writes_user_scope(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    provider._user_scope = "user:jeff"
    provider.on_memory_write("add", "user", "I prefer dark mode in the editor.")

    payload = calls[0][1]
    assert payload["scope"] == "user:jeff"
    assert payload["items"][0]["kind"] == "preference"


def test_on_memory_write_accepts_metadata_kwarg(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    provider.on_memory_write(
        "add", "memory", "Durable fact.", metadata={"session_id": "s1", "tool_name": "memory"}
    )

    item = calls[0][1]["items"][0]
    assert item["metadata"]["sessionId"] == provider._session_id
    assert "session_id" not in item["metadata"]
    assert item["metadata"]["tool_name"] == "memory"


def test_on_memory_write_does_not_forward_old_text_into_metadata(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    mgr = MemoryManager()
    mgr.add_provider(provider)
    secret = "api_key=sk-this-must-not-be-forwarded"

    mgr.notify_memory_tool_write(
        {"success": True},
        {"target": "memory", "action": "replace", "content": "sanitized fact", "old_text": secret},
        build_metadata=lambda: {"tool_name": "memory", "session_id": "core-session"},
    )

    assert len(calls) == 1
    assert secret not in json.dumps(calls[0][1])
    assert "old_text" not in calls[0][1]["items"][0]["metadata"]


def test_extra_metadata_cannot_override_authoritative_provenance(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    provider.on_memory_write(
        "add",
        "memory",
        "Durable fact.",
        metadata={"writePath": "spoofed", "platform": "evil", "sessionId": "attacker", "tool_name": "memory"},
    )

    metadata = calls[0][1]["items"][0]["metadata"]
    assert metadata["writePath"] == "on_memory_write"
    assert metadata["platform"] == provider._platform
    assert metadata["sessionId"] == provider._session_id


def test_on_memory_write_batched_operations_each_ingest(monkeypatch):
    provider = _make_primary_provider()
    calls = []
    monkeypatch.setattr(provider, "_ingest", lambda **kwargs: calls.append(kwargs))

    mgr = MemoryManager()
    mgr.add_provider(provider)
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {
            "target": "memory",
            "operations": [
                {"action": "add", "content": "fact one"},
                {"action": "add", "content": "fact two"},
                {"action": "replace", "content": "fact three", "old_text": "old"},
            ],
        },
    )

    assert [c["content"] for c in calls] == ["fact one", "fact two", "fact three"]
    assert all(c["source"] == "hermes-explicit" for c in calls)
    assert all(c["scope"] == "project:test-project" for c in calls)


def test_on_memory_write_ignores_remove(monkeypatch):
    provider, calls = _provider_with_post_capture(monkeypatch)
    provider.on_memory_write("remove", "memory", "")

    assert calls == []


def test_project_scope_prefers_multica_project_id(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTICA_PROJECT_ID", "fc7a7002-d23f-4718-8ae7-4724652533c9")
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)

    scope = _project_scope_from_cwd(tmp_path)

    assert scope == "project:fc7a7002-d23f-4718-8ae7-4724652533c9"


def test_project_scope_uses_origin_url_not_worktree_path(monkeypatch, tmp_path):
    repo1 = tmp_path / "ws1" / "court-booking-management"
    repo2 = tmp_path / "ws2" / "court-booking-management"
    repo1.mkdir(parents=True)
    repo2.mkdir(parents=True)
    (repo1 / ".git").mkdir()
    (repo2 / ".git").mkdir()

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)

    def fake_run_git(args, cwd: Path):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/dragonguy888/court-booking-management.git"
        if args == ["rev-parse", "--git-common-dir"]:
            return "/stable/common.git"
        return ""

    monkeypatch.setattr(MODULE, "_run_git", fake_run_git)
    monkeypatch.setattr(MODULE, "_run_multica", lambda args, cwd: "")

    scope1 = _project_scope_from_cwd(repo1)
    scope2 = _project_scope_from_cwd(repo2)

    assert scope1 == scope2
    assert scope1 == "project:court-booking-management-498ec503"


def test_project_scope_discovers_single_repo_inside_workdir(monkeypatch, tmp_path):
    workdir = tmp_path / "task-workdir"
    repo = workdir / "court-booking-management"
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)

    def fake_run_git(args, cwd: Path):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/dragonguy888/court-booking-management.git"
        if args == ["rev-parse", "--git-common-dir"]:
            return "/stable/common.git"
        return ""

    monkeypatch.setattr(MODULE, "_run_git", fake_run_git)
    monkeypatch.setattr(MODULE, "_run_multica", lambda args, cwd: "")

    scope = _project_scope_from_cwd(workdir)

    assert scope == "project:court-booking-management-498ec503"


def test_project_scope_uses_multica_issue_context_when_repo_missing(monkeypatch, tmp_path):
    workdir = tmp_path / "task-workdir"
    ctx = workdir / ".agent_context"
    ctx.mkdir(parents=True)
    (ctx / "issue_context.md").write_text(
        "# Task Assignment\n\n**Issue ID:** c1fad34d-b41b-4a29-96ed-5f9a352c275c\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_ISSUE_ID", raising=False)

    def fake_run_multica(args, cwd: Path):
        assert args == ["issue", "get", "c1fad34d-b41b-4a29-96ed-5f9a352c275c", "--output", "json"]
        return '{"project_id": "fc7a7002-d23f-4718-8ae7-4724652533c9"}'

    monkeypatch.setattr(MODULE, "_run_multica", fake_run_multica)

    scope = _project_scope_from_cwd(workdir)

    assert scope == "project:fc7a7002-d23f-4718-8ae7-4724652533c9"


def test_project_scope_uses_single_multica_workspace_project_for_repo(monkeypatch, tmp_path):
    repo = tmp_path / "court-booking-management"
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()

    monkeypatch.delenv("HERMES_MEMORY_PROJECT_SCOPE", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_PROJECT_ID", raising=False)
    monkeypatch.delenv("MULTICA_ISSUE_ID", raising=False)

    def fake_run_git(args, cwd: Path):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/dragonguy888/court-booking-management.git"
        if args == ["rev-parse", "--git-common-dir"]:
            return "/stable/common.git"
        return ""

    def fake_run_multica(args, cwd: Path):
        if args == ["workspace", "get", "--output", "json"]:
            return '{"repos":[{"url":"https://github.com/dragonguy888/court-booking-management.git"}]}'
        if args == ["project", "list", "--output", "json"]:
            return '[{"id":"fc7a7002-d23f-4718-8ae7-4724652533c9"}]'
        return ""

    monkeypatch.setattr(MODULE, "_run_git", fake_run_git)
    monkeypatch.setattr(MODULE, "_run_multica", fake_run_multica)

    scope = _project_scope_from_cwd(repo)

    assert scope == "project:fc7a7002-d23f-4718-8ae7-4724652533c9"
