"""通过本机已登录的 claude CLI（Claude Code）调用 Claude 模型（非 SDK/API）。

与其它 provider 不同：不走 HTTP API，而是 spawn 本机 `claude -p ...
--output-format json` 子进程；请求/响应形状、system 处理和用量字段沿用
Anthropic 原生格式，因此不复用 OpenAICompatibleBaseClient，只借用其中与
协议无关的档位解析帮助函数。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...config import LLMConfig
from ..base import LLMClient, Messages
from ..tiers import resolve_tier
from ..usage import UsageSample, read_usage_int
from ._openai_compatible import ResolvedTier, resolve_provider_tiers

_JSON_MODE_INSTRUCTION = "Output must be valid json."


class AnthropicTierOptions(BaseModel):
    """Anthropic 档位的专属请求选项。"""

    model_config = ConfigDict(extra="forbid")

    thinking: bool = True
    # low | medium | high | xhigh | max；xhigh/max 仅 Opus 档支持，
    # Haiku 4.5 不支持 effort（连同 thinking 一起发送会被拒绝），故只在
    # thinking=True 时随请求发出。字段名沿用其它 provider 的 reasoning_effort，
    # CLI 模式下映射为 `--effort <value>`。
    reasoning_effort: str = "high"
    # SDK 时代遗留字段：CLI 模式下不再消费（那是 API 请求体覆盖，CLI 无此概念）。
    # 保留字段以兼容旧配置文件，配置了会被忽略（AnthropicClient 会打一条提示）。
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

        system_prompt_file: str | None = None
        if system_prompt:
            # 用临时文件而非 `--system-prompt <text>` 传递：Windows 上解析到的
            # claude 可执行文件是 npm 生成的 .cmd 包装脚本，即使 shell=False，
            # Windows 也会隐式经由 cmd.exe 执行它；cmd.exe 自身的命令行解析会把
            # 参数里未转义的 < / > 当作重定向符处理（哪怕被"引号"包裹），而
            # system prompt 里常见的 "<占位符>" 写法就会触发"系统找不到指定的
            # 文件"。文件传递完全绕开 argv/cmd.exe 解析，从根上避免这个问题。
            fd, system_prompt_file = tempfile.mkstemp(suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system_prompt)
            argv += ["--system-prompt-file", system_prompt_file]
        argv += ["-p"]

        try:

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
        finally:
            if system_prompt_file is not None:
                os.remove(system_prompt_file)
