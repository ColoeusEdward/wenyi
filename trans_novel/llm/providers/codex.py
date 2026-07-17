"""Through the local Codex CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomllib landed in 3.11
    tomllib = None

from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...config import LLMConfig
from ..base import LLMClient, Messages
from ..tiers import resolve_tier
from ..usage import UsageSample, read_usage_int
from ._openai_compatible import ResolvedTier, resolve_provider_tiers

_JSON_MODE_INSTRUCTION = "Return only valid JSON, with no markdown fence or explanation."


class CodexTierOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning_effort: str = "high"


def _default_tiers() -> dict[str, ResolvedTier[CodexTierOptions]]:
    return {
        "strong": ResolvedTier("gpt-5.6-terra", CodexTierOptions(reasoning_effort="high")),
        "cheap": ResolvedTier("gpt-5.6-terra", CodexTierOptions(reasoning_effort="medium")),
        "fast": ResolvedTier("gpt-5.6-terra", CodexTierOptions(reasoning_effort="low")),
    }


def build_codex_prompt(messages: Messages, *, json_mode: bool = False) -> str:
    """Convert generic messages into one Codex exec prompt."""
    parts: list[str] = []
    for message in messages:
        content = str(message.get("content", ""))
        if content:
            parts.append(f"[{message.get('role', 'user').upper()}]\n{content}")
    if json_mode:
        parts.append(f"[OUTPUT FORMAT]\n{_JSON_MODE_INSTRUCTION}")
    return "\n\n".join(parts)


def normalize_codex_usage(usage: Any) -> UsageSample | None:
    if usage is None:
        return None
    prompt_tokens = read_usage_int(usage, "input_tokens")
    cache_hit_tokens = read_usage_int(usage, "cached_input_tokens")
    completion_tokens = read_usage_int(usage, "output_tokens")
    return UsageSample(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=read_usage_int(usage, "total_tokens") or prompt_tokens + completion_tokens,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=max(0, prompt_tokens - cache_hit_tokens),
    )


def parse_codex_events(output: str) -> tuple[str, UsageSample | None]:
    """Read the final text and usage from `codex exec --json` JSONL."""
    text: str | None = None
    usage: UsageSample | None = None
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as error:
            raise RuntimeError(f"codex CLI output contains non-JSONL data: {line[:500]!r}") from error
        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                text = item["text"]
        elif event.get("type") == "turn.completed":
            usage = normalize_codex_usage(event.get("usage"))
    if text is None:
        raise RuntimeError(f"codex CLI did not return an agent_message: {output[:500]!r}")
    return text, usage


class CodexClient(LLMClient):
    """Use locally authenticated `codex exec --json` requests."""

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.base_url or cfg.api_key_env:
            print("codex provider uses the local Codex CLI; llm.base_url / api_key_env are ignored.")
        self.tiers = resolve_provider_tiers(
            cfg.tiers, options_type=CodexTierOptions, defaults=_default_tiers()
        )
        self._cli_path: str | None = None
        self._cli_path_lock = threading.Lock()
        self._mcp_disable_cache: list[str] | None = None

    def _mcp_disable_args(self) -> list[str]:
        """Build per-server `-c mcp_servers.<name>.enabled=false` overrides.

        `-c mcp_servers={}` merges non-destructively and silently no-ops
        (openai/codex#16045), so each server configured in the user's
        ~/.codex/config.toml must be disabled by name. Translation calls never
        need MCP servers, and skipping them avoids respawning e.g. the
        playwright npx process on every ephemeral exec.
        """
        with self._cli_path_lock:
            if self._mcp_disable_cache is None:
                names: list[str] = []
                config_path = (
                    Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
                )
                if tomllib is not None:
                    try:
                        with open(config_path, "rb") as fh:
                            servers = tomllib.load(fh).get("mcp_servers")
                        if isinstance(servers, dict):
                            names = list(servers)
                    except (OSError, tomllib.TOMLDecodeError):
                        pass  # 读不到配置就不加覆盖，保持原行为
                self._mcp_disable_cache = [
                    arg
                    for name in names
                    for arg in ("--config", f"mcp_servers.{name}.enabled=false")
                ]
        return self._mcp_disable_cache

    def _ensure_cli_path(self) -> str:
        with self._cli_path_lock:
            if self._cli_path is None:
                path = self.cfg.cli_path or shutil.which("codex")
                if not path:
                    raise RuntimeError(
                        "codex CLI was not found. Install and log in to Codex CLI, or set llm.cli_path in config.yaml."
                    )
                self._cli_path = path
        return self._cli_path

    def complete(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> str:
        del max_tokens
        tier_config = resolve_tier(self.tiers, tier)
        argv = [
            self._ensure_cli_path(), "exec", "--ephemeral", "--skip-git-repo-check",
            # mcp_servers={} 会被非破坏性合并而静默失效（openai/codex#16045），
            # 必须按名逐个 enabled=false 才能真正阻止 MCP server 启动。
            "--config", "mcp_servers={}",
            *self._mcp_disable_args(),
            "--sandbox", "read-only", "--color", "never", "--json", "--model",
            tier_config.model, "--config",
            f'model_reasoning_effort="{tier_config.options.reasoning_effort}"', "-",
        ]
        prompt = build_codex_prompt(messages, json_mode=json_mode)

        @retry(
            stop=stop_after_attempt(self.cfg.max_retries + 1),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> str:
            proc = subprocess.run(
                argv, input=prompt, capture_output=True, text=True,
                timeout=self.cfg.timeout, encoding="utf-8",
            )
            if proc.returncode != 0:
                raise RuntimeError(f"codex CLI exited {proc.returncode}: {proc.stderr[:500]}")
            text, usage = parse_codex_events(proc.stdout)
            self.usage.record(tier, usage, stage)
            return text

        return _call()
