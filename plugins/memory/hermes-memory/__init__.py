from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


def _find_git_root(start: Path) -> Optional[Path]:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / '.git').exists():
            return parent
    return None


def _sanitize_scope_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    return cleaned or "default"


class HermesMemoryProvider(MemoryProvider):
    def __init__(self):
        self._base_url = os.getenv("HERMES_MEMORY_BASE_URL", "http://127.0.0.1:8790").rstrip("/")
        self._allow_remote = str(os.getenv("HERMES_MEMORY_ALLOW_REMOTE", "")).lower() in {"1", "true", "yes", "on"}
        self._session_id = ""
        self._platform = "cli"
        self._user_id = ""
        self._agent_context = "primary"
        self._project_scope: Optional[str] = None
        self._user_scope: Optional[str] = None
        self._global_scope = "global"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "hermes-memory"

    def is_available(self) -> bool:
        if not self._base_url:
            return False
        parsed = urlparse(self._base_url)
        host = (parsed.hostname or "").lower()
        if self._allow_remote:
            return True
        return host in {"127.0.0.1", "localhost", "::1"}

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._platform = kwargs.get("platform", "cli") or "cli"
        self._agent_context = kwargs.get("agent_context", "primary") or "primary"
        self._user_id = str(kwargs.get("user_id", "") or "")
        if self._user_id:
            self._user_scope = f"user:{_sanitize_scope_part(self._user_id)}"
        cwd = os.getenv("TERMINAL_CWD") or os.getcwd()
        root = _find_git_root(Path(cwd))
        if root:
            root_hash = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:8]
            self._project_scope = f"project:{_sanitize_scope_part(root.name)}-{root_hash}"

    def system_prompt_block(self) -> str:
        scopes = ", ".join(self._build_scopes())
        return (
            "# hermes-memory\n"
            f"External memory provider active at {self._base_url}. "
            f"Default recall scopes: {scopes}. "
            "This provider injects recalled durable context before each turn and mirrors durable memory writes after explicit memory updates."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query.strip():
            return ""
        with self._prefetch_lock:
            cached = self._prefetch_result
            self._prefetch_result = ""
        if cached:
            return cached
        return self._perform_prefetch(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query.strip():
            return None
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return None

        def _runner() -> None:
            result = self._perform_prefetch(query)
            with self._prefetch_lock:
                self._prefetch_result = result

        self._prefetch_thread = threading.Thread(target=_runner, daemon=True)
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return
        content = assistant_content.strip()
        if not content:
            return
        if not self._looks_durable(content):
            return
        if self._sync_thread and self._sync_thread.is_alive():
            return

        def _runner() -> None:
            self._ingest(scope=self._preferred_write_scope(), content=content, kind=self._classify_kind(content), summary=content[:280])

        self._sync_thread = threading.Thread(target=_runner, daemon=True)
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if self._agent_context != "primary":
            return
        if action not in {"add", "replace"}:
            return
        scope = self._user_scope if target == "user" and self._user_scope else self._preferred_write_scope()
        kind = "preference" if target == "user" else self._classify_kind(content)
        self._ingest(scope=scope, content=content, kind=kind, summary=content[:280])

    def shutdown(self) -> None:
        return None

    def _build_scopes(self) -> List[str]:
        scopes: List[str] = []
        if self._project_scope:
            scopes.append(self._project_scope)
        if self._user_scope:
            scopes.append(self._user_scope)
        scopes.append(self._global_scope)
        return scopes

    def _preferred_write_scope(self) -> str:
        return self._project_scope or self._user_scope or self._global_scope

    def _looks_durable(self, text: str) -> bool:
        durable_patterns = [
            r"\bprefer(?:s|ence)?\b",
            r"\blikes?\b",
            r"\bmust\b",
            r"\brequired\b",
            r"\bconstraint\b",
            r"\bverified\b",
            r"\broot cause\b",
            r"\bfix(?:ed)?\b",
            r"\bdeploy(?:ment)?\b",
            r"偏好|喜欢|必须|约束|修复|已验证|部署",
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in durable_patterns)

    def _classify_kind(self, text: str) -> str:
        lower = text.lower()
        if re.search(r"prefer|likes|偏好|喜欢", lower):
            return "preference"
        if re.search(r"verified|fix|root cause|修复|已验证", lower):
            return "solution"
        if re.search(r"must|required|constraint|必须|约束", lower):
            return "decision"
        if re.search(r"deploy|deployment|部署", lower):
            return "decision"
        return "fact"

    def _perform_prefetch(self, query: str) -> str:
        payload = {
            "query": query,
            "scopes": self._build_scopes(),
            "limit": 8,
            "budgetChars": 1200,
        }
        try:
            data = self._post_json("/memories/recall", payload)
        except Exception as exc:
            logger.debug("hermes-memory prefetch failed: %s", exc)
            return ""
        context = str(data.get("context", "") or "").strip()
        return f"## hermes-memory recall\n{context}" if context else ""

    def _ingest(self, *, scope: str, content: str, kind: str, summary: str) -> None:
        payload = {
            "scope": scope,
            "items": [{
                "kind": kind,
                "title": f"Hermes memory {self._platform}",
                "content": content,
                "summary": summary,
                "source": "hermes-runtime",
                "sourceRef": self._session_id,
                "importance": 0.85,
                "confidence": 0.9,
                "tags": ["hermes", self._platform, kind],
                "metadata": {
                    "sessionId": self._session_id,
                    "platform": self._platform,
                },
            }],
        }
        try:
            self._post_json("/memories/ingest", payload)
        except Exception as exc:
            logger.debug("hermes-memory ingest failed: %s", exc)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"HTTP request failed: {exc}") from exc


def register(ctx):
    ctx.register_memory_provider(HermesMemoryProvider())
