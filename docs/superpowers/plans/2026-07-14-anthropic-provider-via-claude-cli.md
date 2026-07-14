# Anthropic Provider via Local Claude Code CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `llm.provider: anthropic` call the local `claude` CLI (Claude Code, non-interactive `-p` mode) via `subprocess` instead of the `anthropic` Python SDK, so translation runs use the machine's logged-in Claude Code session instead of an API key.

**Architecture:** Rewrite `trans_novel/llm/providers/anthropic.py` internals only. `AnthropicClient` keeps its `LLMClient` interface (`complete()`), but instead of holding an `anthropic.Anthropic` SDK client it resolves a `claude` executable path once, and on every `complete()` call spawns `subprocess.run([...], input=<user text>, encoding="utf-8", timeout=cfg.timeout)` with `--safe-mode --no-session-persistence --output-format json --tools none --model <tier_model> [--effort <level>] --system-prompt <text> -p`, parses the JSON stdout, and reuses the existing `normalize_anthropic_usage()` on the `usage` field. Retry/backoff via the same `tenacity` decorator pattern already used in this file.

**Tech Stack:** Python 3.13, pydantic v2, tenacity, `subprocess` (stdlib), `unittest` (no pytest installed in this repo's venv — use `unittest`).

## Global Constraints

- Test runner: `.venv/Scripts/python.exe -m unittest tests.test_llm -v` (verified working; `pytest` is NOT installed in this project's venv, do not invoke `pytest` directly).
- `subprocess.run` calls to the `claude` CLI MUST pass `encoding="utf-8"` explicitly — Windows defaults to the system code page (GBK) and corrupts/crashes on Chinese output otherwise (reproduced during design).
- Every CLI invocation MUST include `--safe-mode --no-session-persistence --output-format json --tools none` — `--safe-mode` (not `--bare`) is required to preserve OAuth/subscription auth while skipping CLAUDE.md/skills/hooks/plugins.
- `LLMClient` abstract interface (`trans_novel/llm/base.py`), `factory.py`, and all `agents/*.py` / `pipeline/orchestrator.py` callers must NOT change — they only depend on `AnthropicClient.complete()`'s existing signature and behavior.
- `base_url`, `api_key_env`, and `tiers.<tier>.options.extra_body` are no longer consumed by this provider; if set, ignore them and print one informational log line (do not raise).
- Preserve the `anthropic>=0.80` dependency line in `pyproject.toml` (per user decision — do not remove it even though it's no longer imported).
- New optional config field: `llm.cli_path` (string) — explicit path to the `claude` executable, overriding `shutil.which("claude")`.

---

### Task 1: Add `cli_path` to `LLMConfig` and wire it through `Config.from_dict`

**Files:**
- Modify: `trans_novel/config.py:99-106` (`LLMConfig` class)
- Modify: `trans_novel/config.py:177-185` (`from_dict` construction of `LLMConfig`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `LLMConfig.cli_path: str | None` (new field, default `None`), read from `raw["llm"]["cli_path"]` in `Config.from_dict`.

- [ ] **Step 1: Read the current `LLMConfig` class and `from_dict` method to confirm exact context**

Already confirmed via `rtk read` — `LLMConfig` is:

```python
class LLMConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str | None = None
    api_key_env: str | None = None
    reasoning_style: ReasoningStyle = "none"
    timeout: int = 600
    max_retries: int = 4
    tiers: dict[str, TierConfig] = Field(default_factory=dict)
```

and the `from_dict` construction is:

```python
        llm = LLMConfig(
            provider=llm_raw.get("provider", "deepseek"),
            base_url=llm_raw.get("base_url"),
            api_key_env=llm_raw.get("api_key_env"),
            reasoning_style=llm_raw.get("reasoning_style", "none"),
            timeout=llm_raw.get("timeout", 600),
            max_retries=llm_raw.get("max_retries", 4),
            tiers=tiers,
        )
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_config.py` (open the file first to find the right `TestCase` class for `LLMConfig`/`from_dict` coverage — if none fits cleanly, add a new `TestCase` class at the end of the file, following the existing style of that file):

```python
    def test_llm_cli_path_defaults_to_none_and_can_be_set(self):
        from trans_novel.config import Config

        default_cfg = Config.from_dict({"llm": {"provider": "anthropic"}})
        self.assertIsNone(default_cfg.llm.cli_path)

        custom_cfg = Config.from_dict(
            {"llm": {"provider": "anthropic", "cli_path": r"C:\tools\claude.cmd"}}
        )
        self.assertEqual(custom_cfg.llm.cli_path, r"C:\tools\claude.cmd")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_config -v`
Expected: FAIL — `AttributeError: 'LLMConfig' object has no attribute 'cli_path'`

- [ ] **Step 4: Add the field to `LLMConfig`**

In `trans_novel/config.py`, change:

```python
class LLMConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str | None = None
    api_key_env: str | None = None
    reasoning_style: ReasoningStyle = "none"
    timeout: int = 600
    max_retries: int = 4
    tiers: dict[str, TierConfig] = Field(default_factory=dict)
```

to:

```python
class LLMConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str | None = None
    api_key_env: str | None = None
    reasoning_style: ReasoningStyle = "none"
    timeout: int = 600
    max_retries: int = 4
    tiers: dict[str, TierConfig] = Field(default_factory=dict)
    cli_path: str | None = None  # anthropic provider: 显式指定 claude CLI 可执行文件路径
```

- [ ] **Step 5: Wire it through `from_dict`**

Change:

```python
        llm = LLMConfig(
            provider=llm_raw.get("provider", "deepseek"),
            base_url=llm_raw.get("base_url"),
            api_key_env=llm_raw.get("api_key_env"),
            reasoning_style=llm_raw.get("reasoning_style", "none"),
            timeout=llm_raw.get("timeout", 600),
            max_retries=llm_raw.get("max_retries", 4),
            tiers=tiers,
        )
```

to:

```python
        llm = LLMConfig(
            provider=llm_raw.get("provider", "deepseek"),
            base_url=llm_raw.get("base_url"),
            api_key_env=llm_raw.get("api_key_env"),
            reasoning_style=llm_raw.get("reasoning_style", "none"),
            timeout=llm_raw.get("timeout", 600),
            max_retries=llm_raw.get("max_retries", 4),
            tiers=tiers,
            cli_path=llm_raw.get("cli_path"),
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_config -v`
Expected: PASS (this test, plus all pre-existing tests in the file still pass)

- [ ] **Step 7: Commit**

```bash
git add trans_novel/config.py tests/test_config.py
git commit -m "feat: add optional llm.cli_path config field"
```

---

### Task 2: Rewrite `anthropic.py` — CLI argv/stdin builder replacing `build_request_kwargs`

**Files:**
- Modify: `trans_novel/llm/providers/anthropic.py` (full rewrite of the request-building portion)
- Test: `tests/test_llm.py` (rewrite `TestAnthropicProvider` request-building tests)

**Interfaces:**
- Consumes: `ResolvedTier[AnthropicTierOptions]` from `trans_novel/llm/providers/_openai_compatible.py` (unchanged — `ResolvedTier` dataclass with `.model: str` and `.options: AnthropicTierOptions`).
- Consumes: `Messages = list[dict[str, str]]` from `trans_novel/llm/base.py` (unchanged).
- Produces: `build_cli_invocation(tier_config, messages, *, json_mode=False) -> tuple[list[str], str]` — returns `(extra_argv, stdin_text)` where `extra_argv` is the list of CLI arguments to append (model + optional `--effort`), and `stdin_text` is what gets piped to `-p` via stdin. The `--system-prompt` value is returned separately as the third tuple element: actual signature is `build_cli_invocation(tier_config, messages, *, json_mode=False) -> tuple[list[str], str, str]` = `(extra_argv, system_prompt_text, stdin_text)`.
- Produces: `AnthropicTierOptions` (unchanged: `thinking: bool = True`, `reasoning_effort: str = "high"`, `extra_body: dict[str, Any] = Field(default_factory=dict)` — `extra_body` kept in the model for backward config compatibility but no longer consumed by request building).
- Produces: `normalize_anthropic_usage(usage: Any) -> UsageSample | None` (unchanged, do not touch).
- Produces: `_default_tiers() -> dict[str, ResolvedTier[AnthropicTierOptions]]` (unchanged, do not touch).
- Produces: `_split_system(messages: Messages) -> tuple[str, list[dict[str, str]]]` (unchanged, do not touch — still used by `build_cli_invocation`).

- [ ] **Step 1: Write the failing tests replacing the old `build_request_kwargs` tests**

In `tests/test_llm.py`, the existing `TestAnthropicProvider` class (currently lines 328-459) has these tests that reference the old SDK-shaped `build_request_kwargs`:
- `test_splits_system_and_defaults_to_adaptive_thinking` (lines 334-349)
- `test_thinking_disabled_skips_effort_and_uses_default_max_tokens` (lines 350-366)
- `test_explicit_max_tokens_respected_and_floored_when_thinking` (lines 367-393)
- `test_json_mode_appends_instruction_without_mutating_input` (lines 394-406)
- `test_json_mode_without_system_message_still_injects_instruction` (lines 407-420)
- `test_extra_body_merges_over_generated_kwargs` (lines 421-437)

Replace ALL SIX of those tests (keep `test_usage_normalization_treats_cache_write_as_miss` and `test_usage_normalization_handles_missing_usage` at lines 438-459 untouched) with:

```python
    def test_splits_system_and_defaults_to_effort_flag(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.anthropic import (
            AnthropicTierOptions,
            build_cli_invocation,
        )

        tier = ResolvedTier(model="claude-opus-4-8", options=AnthropicTierOptions())
        extra_argv, system_prompt, stdin_text = build_cli_invocation(tier, self.messages)

        self.assertEqual(system_prompt, "你是翻译。")
        self.assertEqual(stdin_text, "x")
        self.assertIn("--model", extra_argv)
        self.assertEqual(extra_argv[extra_argv.index("--model") + 1], "claude-opus-4-8")
        self.assertIn("--effort", extra_argv)
        self.assertEqual(extra_argv[extra_argv.index("--effort") + 1], "high")

    def test_thinking_disabled_skips_effort_flag(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.anthropic import (
            AnthropicTierOptions,
            build_cli_invocation,
        )

        tier = ResolvedTier(
            model="claude-haiku-4-5",
            options=AnthropicTierOptions(thinking=False),
        )
        extra_argv, _, _ = build_cli_invocation(tier, self.messages)

        self.assertNotIn("--effort", extra_argv)

    def test_custom_reasoning_effort_is_passed_through(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.anthropic import (
            AnthropicTierOptions,
            build_cli_invocation,
        )

        tier = ResolvedTier(
            model="claude-opus-4-8",
            options=AnthropicTierOptions(reasoning_effort="xhigh"),
        )
        extra_argv, _, _ = build_cli_invocation(tier, self.messages)

        self.assertEqual(extra_argv[extra_argv.index("--effort") + 1], "xhigh")

    def test_json_mode_appends_instruction_without_mutating_input(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.anthropic import (
            AnthropicTierOptions,
            build_cli_invocation,
        )

        tier = ResolvedTier(model="m", options=AnthropicTierOptions(thinking=False))
        _, system_prompt, _ = build_cli_invocation(tier, self.messages, json_mode=True)

        self.assertIn("json", system_prompt)
        self.assertEqual(self.messages[0]["content"], "你是翻译。")

    def test_json_mode_without_system_message_still_injects_instruction(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.anthropic import (
            AnthropicTierOptions,
            build_cli_invocation,
        )

        tier = ResolvedTier(model="m", options=AnthropicTierOptions(thinking=False))
        _, system_prompt, _ = build_cli_invocation(
            tier, [{"role": "user", "content": "x"}], json_mode=True
        )

        self.assertIn("json", system_prompt)

    def test_multiple_non_system_messages_join_into_stdin_text(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.anthropic import (
            AnthropicTierOptions,
            build_cli_invocation,
        )

        tier = ResolvedTier(model="m", options=AnthropicTierOptions(thinking=False))
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        _, _, stdin_text = build_cli_invocation(tier, messages)

        self.assertEqual(stdin_text, "first\n\nsecond")
```

Also update the class docstring-adjacent `messages` fixture at the top of `TestAnthropicProvider` (lines 328-332) — it stays as-is:

```python
class TestAnthropicProvider(unittest.TestCase):
    messages = [
        {"role": "system", "content": "你是翻译。"},
        {"role": "user", "content": "x"},
    ]
```

(no change needed to the fixture itself).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_llm -v`
Expected: FAIL — `ImportError: cannot import name 'build_cli_invocation' from 'trans_novel.llm.providers.anthropic'`

- [ ] **Step 3: Replace the request-building portion of `anthropic.py`**

Read the current full file first (already captured above via `rtk read` — the file is 208 lines). Replace the section from `_JSON_MODE_INSTRUCTION = "Output must be valid json."` (line 35) through the end of `build_request_kwargs` (line 141, just before `class AnthropicClient`) with:

```python
_JSON_MODE_INSTRUCTION = "Output must be valid json."


class AnthropicTierOptions(BaseModel):
    """Anthropic 档位的专属请求选项。"""

    model_config = ConfigDict(extra="forbid")

    thinking: bool = True
    # low | medium | high | xhigh | max；xhigh/max 仅 Opus 档支持，
    # Haiku 4.5 不支持 effort（连同 thinking 一起发送会被拒绝），故只在
    # thinking=True 时随请求发出。CLI 模式下映射为 `--effort <value>`。
    reasoning_effort: str = "high"
    # SDK 时代遗留字段：CLI 模式下不再消费（那是 API 请求体覆盖，CLI 无此概念）。
    # 保留字段以兼容旧配置文件，配置了会被忽略（AnthropicClient 会打一条提示日志）。
    extra_body: dict[str, Any] = Field(default_factory=dict)


def _default_tiers() -> dict[str, ResolvedTier[AnthropicTierOptions]]:
    return {
        "strong": ResolvedTier(
            model="claude-opus-4-8",
            options=AnthropicTierOptions(),
        ),
        "cheap": ResolvedTier(
            model="claude-haiku-4-5",
            options=AnthropicTierOptions(thinking=False),
        ),
        "fast": ResolvedTier(
            model="claude-haiku-4-5",
            options=AnthropicTierOptions(thinking=False),
        ),
    }


def _split_system(messages: Messages) -> tuple[str, list[dict[str, str]]]:
    """把 system 角色消息提取为单独文本（Anthropic 要求 system 独立于 messages）。

    多条 system 消息按 Anthropic 官方 OpenAI 兼容层的做法用换行拼接，保持行为一致。
    """
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content", "")
            if content:
                system_parts.append(str(content))
        else:
            rest.append(dict(message))
    return "\n".join(system_parts), rest


def normalize_anthropic_usage(usage: Any) -> UsageSample | None:
    """把 Anthropic 用量字段换算成统一用量。

    Anthropic 的 input_tokens 只是未命中缓存的剩余部分，完整 prompt 大小是
    input_tokens + cache_creation_input_tokens + cache_read_input_tokens；
    据此把 cache_creation（写入，非命中）与 input_tokens 一并计入 miss，
    cache_read（命中）计入 hit，prompt_tokens 取两者之和以对齐其它 provider
    「prompt_tokens = hit + miss」的语义。

    CLI 模式下这个函数直接消费 `claude -p --output-format json` 响应体里的
    `usage` 字段——字段名与官方 Messages API 完全一致，无需额外转换。
    """
    if usage is None:
        return None
    input_tokens = read_usage_int(usage, "input_tokens")
    cache_creation = read_usage_int(usage, "cache_creation_input_tokens")
    cache_read = read_usage_int(usage, "cache_read_input_tokens")
    output_tokens = read_usage_int(usage, "output_tokens")
    cache_miss_tokens = input_tokens + cache_creation
    cache_hit_tokens = cache_read
    prompt_tokens = cache_miss_tokens + cache_hit_tokens
    return UsageSample(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
    )


def build_cli_invocation(
    tier_config: ResolvedTier[AnthropicTierOptions],
    messages: Messages,
    *,
    json_mode: bool = False,
) -> tuple[list[str], str, str]:
    """把通用 messages 转换成 `claude` CLI 的调用形状。

    返回 (extra_argv, system_prompt_text, stdin_text)：
    - extra_argv：追加到固定 CLI flags 后的档位专属参数（--model、可选 --effort）
    - system_prompt_text：喂给 `--system-prompt` 的文本（json_mode 时追加指令）
    - stdin_text：喂给 `-p` 的 stdin 内容（非 system 消息按顺序拼接；正常场景下
      调用方只传一条 system + 一条 user，多条消息只是兜底不炸）
    """
    system_text, chat_messages = _split_system(messages)
    if json_mode:
        system_text = (
            f"{system_text}\n\n{_JSON_MODE_INSTRUCTION}"
            if system_text
            else _JSON_MODE_INSTRUCTION
        )
    stdin_text = "\n\n".join(
        str(message.get("content", "")) for message in chat_messages
    )
    extra_argv: list[str] = ["--model", tier_config.model]
    if tier_config.options.thinking:
        extra_argv += ["--effort", tier_config.options.reasoning_effort]
    return extra_argv, system_text, stdin_text
```

Note: this deletes `deep_merge` usage from this file (it's still imported/used
by other providers via `_openai_compatible.py`, just no longer needed here) and
deletes `resolve_provider_tiers`'s import usage of `deep_merge` — check the
import line at the top of the file:

```python
from ._openai_compatible import ResolvedTier, deep_merge, resolve_provider_tiers
```

Change to (drop unused `deep_merge`):

```python
from ._openai_compatible import ResolvedTier, resolve_provider_tiers
```

Also remove the now-unused constants `_DEFAULT_MAX_TOKENS` and
`_THINKING_MIN_MAX_TOKENS` (lines 31-32) — CLI mode has no `max_tokens`
parameter to manage (the CLI manages its own output budget). Delete these two
lines entirely:

```python
_DEFAULT_MAX_TOKENS = 8192
_THINKING_MIN_MAX_TOKENS = 16000
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.test_llm -v`
Expected: PASS for all `TestAnthropicProvider` tests written in Step 1. (`AnthropicClient` itself is not yet updated to use `build_cli_invocation` — that's Task 3 — so `TestProviderFactory` tests may still reference the old class; they are not run/fixed until Task 3. It's fine if `AnthropicClient.complete()` is currently broken/unused at this point, as long as nothing currently calls it in this test run. Confirm this by running the full suite and checking only `TestAnthropicProvider`-prefixed tests plus pre-existing unrelated tests pass; `TestProviderFactory.test_builds_each_provider_from_its_own_module` merely instantiates `AnthropicClient(config.llm)` without calling `.complete()`, so it should still pass since `__init__` hasn't changed shape yet.)

- [ ] **Step 5: Commit**

```bash
git add trans_novel/llm/providers/anthropic.py tests/test_llm.py
git commit -m "refactor: replace anthropic SDK request-kwargs builder with CLI invocation builder"
```

---

### Task 3: Rewrite `AnthropicClient` to shell out to the `claude` CLI

**Files:**
- Modify: `trans_novel/llm/providers/anthropic.py` (the `AnthropicClient` class, currently lines 143-208)
- Test: `tests/test_llm.py` (new tests for `AnthropicClient.complete()`)

**Interfaces:**
- Consumes: `build_cli_invocation` and `normalize_anthropic_usage` from Task 2 (same file, same module — no cross-file import change).
- Consumes: `LLMConfig` from `trans_novel/config.py`, now including `.cli_path: str | None` from Task 1.
- Consumes: `resolve_tier` from `trans_novel/llm/tiers.py` (unchanged signature: `resolve_tier(tiers: dict, tier: str) -> ResolvedTier`).
- Produces: `AnthropicClient(cfg: LLMConfig)` — same constructor signature as before. `AnthropicClient.complete(messages, *, tier="strong", json_mode=False, max_tokens=None, stage=None) -> str` — same signature as `LLMClient.complete` abstract method (unchanged; `max_tokens` param accepted for interface compatibility but not forwarded to the CLI — CLI has no equivalent flag).
- Produces (new internal helper, module-private): `_resolve_cli_path(cli_path: str | None) -> str` — resolves `cli_path` if given, else `shutil.which("claude")`, else raises `RuntimeError` mentioning `cli_path`.

- [ ] **Step 1: Write the failing tests for `AnthropicClient.complete()`**

Add to `tests/test_llm.py`, right after the `test_usage_normalization_handles_missing_usage` test (currently the last test in `TestAnthropicProvider`, ending at line 459) and before `class TestProviderFactory` (currently line 461):

```python
    def test_complete_invokes_cli_and_returns_result_text(self):
        import json
        from unittest.mock import patch

        from trans_novel.config import LLMConfig
        from trans_novel.llm.providers.anthropic import AnthropicClient

        cfg = LLMConfig(tiers={"strong": {"model": "claude-opus-4-8"}})
        client = AnthropicClient(cfg)

        fake_result = {
            "is_error": False,
            "result": "你好",
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 5,
            },
        }

        captured = {}

        def fake_run(argv, *, input, capture_output, text, timeout, encoding):
            captured["argv"] = argv
            captured["input"] = input
            captured["timeout"] = timeout
            captured["encoding"] = encoding

            class Result:
                returncode = 0
                stdout = json.dumps(fake_result)
                stderr = ""

            return Result()

        with patch("shutil.which", return_value=r"C:\nodejs\claude.cmd"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            text = client.complete(
                [
                    {"role": "system", "content": "你是翻译。"},
                    {"role": "user", "content": "hello"},
                ],
                tier="strong",
                stage="Translator",
            )

        self.assertEqual(text, "你好")
        self.assertEqual(captured["encoding"], "utf-8")
        self.assertEqual(captured["input"], "hello")
        self.assertIn(r"C:\nodejs\claude.cmd", captured["argv"])
        self.assertIn("--safe-mode", captured["argv"])
        self.assertIn("--no-session-persistence", captured["argv"])
        self.assertIn("--tools", captured["argv"])
        self.assertIn("none", captured["argv"])
        self.assertIn("--system-prompt", captured["argv"])
        self.assertIn("你是翻译。", captured["argv"])
        self.assertIn("--model", captured["argv"])
        self.assertIn("claude-opus-4-8", captured["argv"])

        summary = client.usage_summary()
        self.assertEqual(summary["totals"]["prompt_tokens"], 10)
        self.assertEqual(summary["totals"]["completion_tokens"], 5)

    def test_complete_retries_on_is_error_then_succeeds(self):
        import json
        from unittest.mock import patch

        from trans_novel.config import LLMConfig
        from trans_novel.llm.providers.anthropic import AnthropicClient

        cfg = LLMConfig(tiers={"strong": {"model": "m"}}, max_retries=2)
        client = AnthropicClient(cfg)

        calls = {"count": 0}

        def fake_run(argv, *, input, capture_output, text, timeout, encoding):
            calls["count"] += 1

            class Result:
                returncode = 0
                stderr = ""

            result = Result()
            if calls["count"] == 1:
                result.stdout = json.dumps({"is_error": True, "result": ""})
            else:
                result.stdout = json.dumps(
                    {"is_error": False, "result": "ok", "usage": None}
                )
            return result

        with patch("shutil.which", return_value="/usr/bin/claude"), patch(
            "subprocess.run", side_effect=fake_run
        ), patch("time.sleep", return_value=None):
            text = client.complete([{"role": "user", "content": "x"}])

        self.assertEqual(text, "ok")
        self.assertEqual(calls["count"], 2)

    def test_complete_raises_when_cli_not_found(self):
        from unittest.mock import patch

        from trans_novel.config import LLMConfig
        from trans_novel.llm.providers.anthropic import AnthropicClient

        cfg = LLMConfig(tiers={"strong": {"model": "m"}})
        client = AnthropicClient(cfg)

        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cli_path"):
                client.complete([{"role": "user", "content": "x"}])

    def test_complete_uses_explicit_cli_path_over_which(self):
        import json
        from unittest.mock import patch

        from trans_novel.config import LLMConfig
        from trans_novel.llm.providers.anthropic import AnthropicClient

        cfg = LLMConfig(
            tiers={"strong": {"model": "m"}}, cli_path=r"D:\custom\claude.cmd"
        )
        client = AnthropicClient(cfg)

        captured = {}

        def fake_run(argv, *, input, capture_output, text, timeout, encoding):
            captured["argv"] = argv

            class Result:
                returncode = 0
                stdout = json.dumps(
                    {"is_error": False, "result": "ok", "usage": None}
                )
                stderr = ""

            return Result()

        with patch("shutil.which", return_value="/usr/bin/claude"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            client.complete([{"role": "user", "content": "x"}])

        self.assertIn(r"D:\custom\claude.cmd", captured["argv"])
        self.assertNotIn("/usr/bin/claude", captured["argv"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_llm -v`
Expected: FAIL — either `TypeError`/`AttributeError` because `AnthropicClient` still uses the old SDK-based `_ensure_client`/`complete`, or import errors, since `AnthropicClient.__init__` currently sets `self._client = None` for an SDK client and `complete()` calls `client.messages.create(**kwargs)`.

- [ ] **Step 3: Rewrite `AnthropicClient`**

Replace the entire `class AnthropicClient(LLMClient):` block (from `class AnthropicClient(LLMClient):` through the end of the file, currently lines 143-208) with:

```python
class AnthropicClient(LLMClient):
    """通过本机已登录的 `claude` CLI（Claude Code）调用 Claude 模型。

    不使用 anthropic SDK：每次 complete() 调用都 spawn 一个独立的
    `claude -p ... --output-format json` 子进程，解析其 JSON 输出得到回复
    文本和 usage 统计。鉴权完全依赖本机 `claude` 的登录态（OAuth/订阅），
    不需要配置 API key / base_url。
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.base_url:
            print(
                "提示：anthropic provider 已改用本机 claude CLI，"
                "llm.base_url 不再生效，已忽略。"
            )
        if cfg.api_key_env:
            print(
                "提示：anthropic provider 已改用本机 claude CLI，"
                "llm.api_key_env 不再生效，已忽略。"
            )
        self.tiers = resolve_provider_tiers(
            cfg.tiers,
            options_type=AnthropicTierOptions,
            defaults=_default_tiers(),
        )
        for name, tier in self.tiers.items():
            if tier.options.extra_body:
                print(
                    f"提示：anthropic provider 已改用本机 claude CLI，"
                    f"llm.tiers.{name}.options.extra_body 不再生效，已忽略。"
                )
        self._cli_path: str | None = None
        self._cli_path_lock = threading.Lock()

    def _ensure_cli_path(self) -> str:
        with self._cli_path_lock:
            if self._cli_path is None:
                import shutil

                path = self.cfg.cli_path or shutil.which("claude")
                if not path:
                    raise RuntimeError(
                        "找不到 claude CLI 可执行文件。请确认已安装并登录 "
                        "Claude Code（claude --version 可正常运行），或在 "
                        "config.yaml 的 llm.cli_path 显式指定可执行文件路径。"
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
        tier_config = resolve_tier(self.tiers, tier)
        extra_argv, system_prompt, stdin_text = build_cli_invocation(
            tier_config, messages, json_mode=json_mode
        )
        cli_path = self._ensure_cli_path()
        argv = (
            [cli_path]
            + [
                "--safe-mode",
                "--no-session-persistence",
                "--output-format",
                "json",
                "--tools",
                "none",
            ]
            + extra_argv
        )
        if system_prompt:
            argv += ["--system-prompt", system_prompt]
        argv += ["-p"]

        @retry(
            stop=stop_after_attempt(self.cfg.max_retries + 1),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> str:
            proc = subprocess.run(
                argv,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout,
                encoding="utf-8",
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"claude CLI 退出码非 0（{proc.returncode}）：{proc.stderr[:500]}"
                )
            try:
                data = json.loads(proc.stdout)
            except ValueError as error:
                raise RuntimeError(
                    f"claude CLI 输出不是合法 JSON：{proc.stdout[:500]!r}"
                ) from error
            if data.get("is_error"):
                raise RuntimeError(f"claude CLI 返回错误：{data!r}")
            sample = normalize_anthropic_usage(data.get("usage"))
            self.usage.record(tier, sample, stage)
            return data.get("result", "")

        return _call()
```

Add the two new stdlib imports at the top of the file (alongside the existing
`import os` / `import threading`):

```python
import json
import subprocess
```

(`os` is currently imported but only used by `DEFAULT_API_KEY_ENV`-adjacent
code that we're removing — check if `os` is still used anywhere else in the
file after this change; if not, remove the `import os` line. Also remove the
now-unused `DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"` constant, currently at
line 27, since API key env vars are no longer read by this provider.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.test_llm -v`
Expected: PASS — all tests in `TestAnthropicProvider` (from Task 2 and Task 3) and all pre-existing tests in the file (`TestProviderFactory`, etc.) pass. Total test count should be higher than the baseline 32 (baseline confirmed via `.venv/Scripts/python.exe -m unittest tests.test_llm -v` before this work started).

- [ ] **Step 5: Run the FULL test suite to confirm no regressions elsewhere**

Run: `.venv/Scripts/python.exe -m unittest discover tests -v 2>&1 | tail -40`
Expected: All tests OK, no failures/errors (confirms `agents/*`, `pipeline/*`, `cli.py` tests — which construct `AnthropicClient` indirectly via fakes/mocks or don't touch it at all — are unaffected).

- [ ] **Step 6: Commit**

```bash
git add trans_novel/llm/providers/anthropic.py tests/test_llm.py
git commit -m "feat: AnthropicClient shells out to local claude CLI instead of anthropic SDK"
```

---

### Task 4: Update project `config.yaml` and `docs/configuration.md`

**Files:**
- Modify: `D:/Software/wenyi-me/wenyi/config.yaml` (comment out `base_url`/`api_key_env` in the `llm` section)
- Modify: `docs/configuration.md` (rewrite the "Anthropic（Claude 原生接口）" section, currently around lines 90-116)

**Interfaces:**
- None (docs/config-only task, no code interfaces).

- [ ] **Step 1: Update `config.yaml`**

Current content (relevant lines, already confirmed via `rtk read`):

```yaml
llm:
  # deepseek | openai | anthropic | openrouter | openai-compatible | ollama | vllm | fake
  provider: anthropic
  # base_url: https://api.deepseek.com
  base_url: https://api-cc.freemodel.dev
  api_key_env: MY_KEY
  timeout: 600
  max_retries: 4
```

Change to:

```yaml
llm:
  # deepseek | openai | anthropic | openrouter | openai-compatible | ollama | vllm | fake
  provider: anthropic
  # anthropic provider 现已改用本机已登录的 Claude Code CLI（claude -p ...），
  # 不再需要 base_url / api_key_env，下面两行留作历史记录，已不生效：
  # base_url: https://api-cc.freemodel.dev
  # api_key_env: MY_KEY
  timeout: 600
  max_retries: 4
```

- [ ] **Step 2: Verify `config.yaml` still parses correctly**

Run:

```bash
cd "D:/Software/wenyi-me/wenyi" && .venv/Scripts/python.exe -c "from trans_novel.config import Config; c = Config.load('config.yaml'); print(c.llm.provider, c.llm.base_url, c.llm.api_key_env)"
```

Expected output: `anthropic None None` (both fields now `None` since they're commented out).

- [ ] **Step 3: Update `docs/configuration.md`**

Read the current section first (already confirmed via `rtk read`, lines ~90-116):

```markdown
### Anthropic（Claude 原生接口）

`anthropic` provider 直接调用 Anthropic 官方 Messages API（而非 OpenAI 兼容
接口），可完整使用 adaptive thinking 与 `effort` 档位调节：

```yaml
llm:
  provider: anthropic
  api_key_env: ANTHROPIC_API_KEY # 默认值，可省略
  tiers:
    strong:
      model: claude-opus-4-8
      options:
        thinking: true
        reasoning_effort: high # low | medium | high | xhigh | max（xhigh/max 仅 Opus 档支持）
    cheap:
      model: claude-haiku-4-5
      options:
        thinking: false # Haiku 4.5 不支持 effort，关闭思考即可
    fast:
      model: claude-haiku-4-5
      options:
        thinking: false
```

默认读取 `ANTHROPIC_API_KEY`；`base_url` 一般无需配置（留空即用官方地址，
Claude Platform on AWS 等兼容部署可覆盖）。`thinking: true` 时启用
`thinking: {type: "adaptive"}` 并附带 `effort`；关闭思考的档位不发送
`effort`（Haiku 4.5 等非 Opus 模型不支持该参数）。
```

Replace with:

```markdown
### Anthropic（本机 Claude Code CLI）

`anthropic` provider 通过本机已登录的 Claude Code CLI（`claude -p ...`
非交互模式）调用 Claude 模型，而不是 Anthropic SDK/API key。鉴权完全依赖
本机 `claude` 的登录态（OAuth/订阅），无需配置 API key：

```yaml
llm:
  provider: anthropic
  timeout: 600
  max_retries: 4
  tiers:
    strong:
      model: claude-opus-4-8
      options:
        thinking: true
        reasoning_effort: high # low | medium | high | xhigh | max（xhigh/max 仅 Opus 档支持）
    cheap:
      model: claude-haiku-4-5
      options:
        thinking: false # Haiku 4.5 不支持 effort，关闭思考即可
    fast:
      model: claude-haiku-4-5
      options:
        thinking: false
```

`base_url`、`api_key_env`、`tiers.<tier>.options.extra_body` 在此 provider
下不再生效（配置了会被忽略并打印一条提示）。`thinking: true` 时映射为 CLI
的 `--effort <reasoning_effort>`；关闭思考的档位不发送 `--effort`（Haiku
4.5 等非 Opus 模型不支持该参数）。

默认通过 `shutil.which("claude")` 定位可执行文件；如果本机 PATH 上的
`claude` 命令有问题（比如损坏的全局 shim），可用 `llm.cli_path` 显式指定：

```yaml
llm:
  provider: anthropic
  cli_path: C:\Program Files\nodejs\claude.cmd
```
```

- [ ] **Step 4: Commit**

```bash
git -C "D:/Software/wenyi-me/wenyi" add config.yaml docs/configuration.md
git -C "D:/Software/wenyi-me/wenyi" commit -m "docs: document anthropic provider running via local claude CLI"
```

---

### Task 5: End-to-end manual verification against the real local `claude` CLI

**Files:**
- None modified — this task is a manual verification pass using the real environment (not mocked).

**Interfaces:**
- None (verification-only task).

- [ ] **Step 1: Confirm `claude` CLI is reachable and logged in**

Run:

```bash
"C:/Program Files/nodejs/claude.cmd" --version
```

Expected: prints a version string like `2.1.208 (Claude Code)` with exit code 0. If this fails, STOP — fix the local `claude` CLI installation/login before continuing (this task cannot be verified without it).

- [ ] **Step 2: Run a real (non-mocked) `AnthropicClient.complete()` call**

Run:

```bash
cd "D:/Software/wenyi-me/wenyi" && .venv/Scripts/python.exe -c "
from trans_novel.config import LLMConfig
from trans_novel.llm.providers.anthropic import AnthropicClient

cfg = LLMConfig(
    provider='anthropic',
    cli_path=r'C:\Program Files\nodejs\claude.cmd',
    tiers={'strong': {'model': 'claude-haiku-4-5-20251001', 'options': {'thinking': False}}},
)
client = AnthropicClient(cfg)
text = client.complete(
    [
        {'role': 'system', 'content': 'You are a translator. Reply with only the translation.'},
        {'role': 'user', 'content': 'Translate to Chinese: The library was quiet in the evening.'},
    ],
    tier='strong',
)
print('RESULT:', text)
print('USAGE:', client.usage_summary())
"
```

Expected: prints a Chinese translation of the sentence (readable, not garbled/mojibake — this is the concrete check that the `encoding="utf-8"` fix from Task 3 actually works end-to-end against the real CLI, not just the mocked test), followed by a usage summary dict with non-zero `prompt_tokens`/`completion_tokens` under `totals`.

- [ ] **Step 3: Run a real `complete_json` call to confirm JSON-mode path works too**

Run:

```bash
cd "D:/Software/wenyi-me/wenyi" && .venv/Scripts/python.exe -c "
from trans_novel.config import LLMConfig
from trans_novel.llm.providers.anthropic import AnthropicClient

cfg = LLMConfig(
    provider='anthropic',
    cli_path=r'C:\Program Files\nodejs\claude.cmd',
    tiers={'strong': {'model': 'claude-haiku-4-5-20251001', 'options': {'thinking': False}}},
)
client = AnthropicClient(cfg)
data = client.complete_json(
    [
        {'role': 'system', 'content': 'Reply with a JSON object: {\"greeting\": <a short Chinese greeting>}'},
        {'role': 'user', 'content': 'go'},
    ],
    tier='strong',
)
print('PARSED:', data)
"
```

Expected: prints a parsed Python dict (e.g. `PARSED: {'greeting': '你好'}`), confirming `complete_json` (in `trans_novel/llm/base.py`, unchanged) correctly layers `parse_json_loose` over the new CLI-backed `complete()`.

- [ ] **Step 4: Confirm the full test suite still passes after all changes**

Run: `.venv/Scripts/python.exe -m unittest discover tests -v 2>&1 | tail -40`
Expected: `OK`, no failures/errors.

- [ ] **Step 5: No commit needed for this task** — it's a verification-only pass with no file changes. If Steps 2 or 3 reveal a bug, go back to Task 3 and fix it there (with an accompanying test update), then re-run this task's verification steps.

---

## Plan Self-Review Notes

- **Spec coverage:** Task 1 covers `llm.cli_path` config field (Spec §2). Task 2 covers CLI argv/stdin building, `--effort` mapping, json_mode instruction, `extra_body` no-op (Spec §2, §3). Task 3 covers fixed CLI flags, subprocess timeout/retry/error handling, UTF-8 encoding, usage parsing, executable resolution (Spec §3). Task 4 covers config.yaml cleanup and docs (Spec §4). Task 5 covers real end-to-end verification against the actual local CLI (not in spec explicitly but required by `superpowers:verification-before-completion` before claiming this done). `pyproject.toml` dependency retention (Spec §4) required no task — it's a "leave as-is" decision, nothing to change.
- **Placeholder scan:** No TBD/TODO markers; every step has literal code or literal shell commands with expected output.
- **Type consistency:** `build_cli_invocation` returns `tuple[list[str], str, str]` consistently referenced in Task 2 and Task 3. `AnthropicClient.complete()` signature matches `LLMClient.complete` abstract method exactly (same param names/types) across all tasks. `LLMConfig.cli_path: str | None` introduced in Task 1 is consumed identically in Task 3's `_ensure_cli_path`.
