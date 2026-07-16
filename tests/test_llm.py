"""LLM 抽象层与 JSON 解析的测试（离线）。"""

from __future__ import annotations

import unittest

from trans_novel.llm.json_parser import parse_json_loose
from trans_novel.llm.providers.fake import FakeClient


class TestParseJsonLoose(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json_loose('{"a":1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(parse_json_loose("```json\n[1,2,3]\n```"), [1, 2, 3])

    def test_surrounded_by_prose(self):
        text = '思考结束。结果如下：["译文1","译文2"] 完毕。'
        self.assertEqual(parse_json_loose(text), ["译文1", "译文2"])

    def test_failure(self):
        with self.assertRaises(ValueError):
            parse_json_loose("没有任何 JSON 内容")


class TestResolveTier(unittest.TestCase):
    def test_fallback_chain(self):
        from trans_novel.config import TierConfig
        from trans_novel.llm.tiers import resolve_tier

        strong = TierConfig(model="pro")
        cheap = TierConfig(model="flash")
        fast = TierConfig(model="flash", options={"thinking": False})

        # 三档全有 → 各归各
        tiers = {"strong": strong, "cheap": cheap, "fast": fast}
        self.assertIs(resolve_tier(tiers, "fast"), fast)
        self.assertIs(resolve_tier(tiers, "cheap"), cheap)
        self.assertIs(resolve_tier(tiers, "strong"), strong)
        # 无 fast → 落 cheap（不升到更贵的 strong）
        tiers2 = {"strong": strong, "cheap": cheap}
        self.assertIs(resolve_tier(tiers2, "fast"), cheap)
        # 只有 strong → 都落 strong
        tiers3 = {"strong": strong}
        self.assertIs(resolve_tier(tiers3, "fast"), strong)
        self.assertIs(resolve_tier(tiers3, "cheap"), strong)
        # 未知档 → 落 strong
        self.assertIs(resolve_tier(tiers, "unknown"), strong)


class TestFakeClient(unittest.TestCase):
    def test_default(self):
        c = FakeClient()
        self.assertEqual(c.complete([{"role": "user", "content": "x"}]), "")
        self.assertEqual(c.complete_json([{"role": "user", "content": "x"}]), [])

    def test_handler(self):
        def handler(messages, tier, json_mode):
            return '["A","B"]' if json_mode else "hello"

        c = FakeClient(handler=handler)
        self.assertEqual(c.complete([{"role": "user", "content": "x"}]), "hello")
        self.assertEqual(c.complete_json([{"role": "user", "content": "x"}]), ["A", "B"])
        self.assertEqual(len(c.calls), 2)


class TestParseJsonLooseRepairs(unittest.TestCase):
    def test_inner_ascii_quotes_repaired(self):
        # 真实案例：claude-opus-4.6 经 OpenRouter 输出的译文含未转义英文引号
        raw = '{"translations":["磨到那份锱铢必较里暗含的"小气"二字无声地烫上面颊。"]}'
        got = parse_json_loose(raw)
        self.assertEqual(got["translations"][0], '磨到那份锱铢必较里暗含的"小气"二字无声地烫上面颊。')

    def test_trailing_extra_brace(self):
        # 真实案例：gemini-3.1-pro 输出末尾多一个 }
        self.assertEqual(parse_json_loose('{"a": 1}\n}'), {"a": 1})

    def test_unescaped_quotes_with_trailing_extra_brace_keeps_object(self):
        raw = '{"translations":["他说"好"。"]}\n}'
        self.assertEqual(
            parse_json_loose(raw),
            {"translations": ['他说"好"。']},
        )

    def test_valid_json_untouched(self):
        self.assertEqual(parse_json_loose('{"a": "b, c: d"}'), {"a": "b, c: d"})

    def test_escaped_quotes_still_work(self):
        self.assertEqual(parse_json_loose('{"a": "he said \\"hi\\""}'), {"a": 'he said "hi"'})


class TestProviderRequestKwargs(unittest.TestCase):
    messages = [{"role": "user", "content": "x"}]

    def test_json_mode_adds_lowercase_keyword_without_mutating_messages(self):
        from trans_novel.llm.providers._openai_compatible import (
            base_request_kwargs,
        )

        messages = [
            {"role": "system", "content": "仅输出指定对象。"},
            {"role": "user", "content": "x"},
        ]
        kwargs = base_request_kwargs("m", messages, json_mode=True)

        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertIn("json", kwargs["messages"][0]["content"])
        self.assertEqual(messages[0]["content"], "仅输出指定对象。")

    def test_deepseek_dialect_and_recursive_extra_body(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.deepseek import (
            DeepSeekTierOptions,
            build_request_kwargs,
        )

        tier = ResolvedTier(
            model="m",
            options=DeepSeekTierOptions(
                extra_body={"thinking": {"budget": 8192}},
            ),
        )
        kwargs = build_request_kwargs(tier, self.messages)

        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(
            kwargs["extra_body"],
            {"thinking": {"type": "enabled", "budget": 8192}},
        )

        disabled = ResolvedTier(
            model="m",
            options=DeepSeekTierOptions(thinking=False),
        )
        disabled_kwargs = build_request_kwargs(disabled, self.messages)
        self.assertNotIn("reasoning_effort", disabled_kwargs)
        self.assertEqual(
            disabled_kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_openrouter_dialect_and_explicit_disable(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.openrouter import (
            OpenRouterTierOptions,
            build_request_kwargs,
        )

        enabled = ResolvedTier(
            model="m",
            options=OpenRouterTierOptions(reasoning_effort="high"),
        )
        disabled = ResolvedTier(
            model="m",
            options=OpenRouterTierOptions(thinking=False),
        )

        self.assertEqual(
            build_request_kwargs(enabled, self.messages)["extra_body"],
            {"reasoning": {"effort": "high"}},
        )
        self.assertEqual(
            build_request_kwargs(disabled, self.messages)["extra_body"],
            {"reasoning": {"enabled": False}},
        )

    def test_openai_dialect(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.openai import (
            OpenAITierOptions,
            build_request_kwargs,
        )

        tier = ResolvedTier(
            model="m",
            options=OpenAITierOptions(reasoning_effort="low"),
        )
        kwargs = build_request_kwargs(tier, self.messages)

        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertNotIn("extra_body", kwargs)

        disabled = ResolvedTier(
            model="m",
            options=OpenAITierOptions(thinking=False),
        )
        disabled_kwargs = build_request_kwargs(disabled, self.messages)
        self.assertEqual(disabled_kwargs["reasoning_effort"], "none")

    def test_openai_uses_max_completion_tokens(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.openai import (
            OpenAITierOptions,
            build_request_kwargs,
        )

        enabled = ResolvedTier(model="m", options=OpenAITierOptions())
        disabled = ResolvedTier(
            model="m",
            options=OpenAITierOptions(thinking=False),
        )

        enabled_kwargs = build_request_kwargs(
            enabled,
            self.messages,
            max_tokens=100,
        )
        disabled_kwargs = build_request_kwargs(
            disabled,
            self.messages,
            max_tokens=100,
        )
        self.assertNotIn("max_tokens", enabled_kwargs)
        self.assertEqual(enabled_kwargs["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", disabled_kwargs)
        self.assertEqual(disabled_kwargs["max_completion_tokens"], 100)

    def test_generic_compatible_endpoint_maps_reasoning_dialects(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.openai_compatible import (
            OpenAICompatibleTierOptions,
            build_request_kwargs,
        )

        tier = ResolvedTier(
            model="m",
            options=OpenAICompatibleTierOptions(
                thinking=True,
                reasoning_effort="medium",
                request_overrides={"thinking": {"budget": 8192}},
            ),
        )
        deepseek = build_request_kwargs(
            tier,
            self.messages,
            max_tokens=100,
            reasoning_style="deepseek",
        )
        openai = build_request_kwargs(
            tier,
            self.messages,
            reasoning_style="openai",
        )
        openrouter = build_request_kwargs(
            tier,
            self.messages,
            reasoning_style="openrouter",
        )

        self.assertEqual(deepseek["reasoning_effort"], "medium")
        self.assertEqual(
            deepseek["extra_body"],
            {"thinking": {"type": "enabled", "budget": 8192}},
        )
        self.assertEqual(deepseek["max_tokens"], 4096)
        self.assertEqual(openai["reasoning_effort"], "medium")
        self.assertEqual(
            openai["extra_body"],
            {"thinking": {"budget": 8192}},
        )
        self.assertEqual(
            openrouter["extra_body"],
            {
                "reasoning": {"effort": "medium"},
                "thinking": {"budget": 8192},
            },
        )

    def test_generic_compatible_endpoint_explicitly_disables_reasoning(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.openai_compatible import (
            OpenAICompatibleTierOptions,
            build_request_kwargs,
        )

        tier = ResolvedTier(
            model="m",
            options=OpenAICompatibleTierOptions(thinking=False),
        )

        self.assertEqual(
            build_request_kwargs(
                tier,
                self.messages,
                reasoning_style="deepseek",
            )["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            build_request_kwargs(
                tier,
                self.messages,
                reasoning_style="openai",
            )["reasoning_effort"],
            "none",
        )
        self.assertEqual(
            build_request_kwargs(
                tier,
                self.messages,
                reasoning_style="openrouter",
            )["extra_body"],
            {"reasoning": {"enabled": False}},
        )

    def test_generic_compatible_endpoint_can_only_use_raw_overrides(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.openai_compatible import (
            OpenAICompatibleTierOptions,
            build_request_kwargs,
        )

        tier = ResolvedTier(
            model="m",
            options=OpenAICompatibleTierOptions(
                thinking=True,
                request_overrides={"enable_thinking": True},
            ),
        )
        kwargs = build_request_kwargs(tier, self.messages, max_tokens=100)

        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["extra_body"], {"enable_thinking": True})
        self.assertEqual(kwargs["max_tokens"], 4096)


class TestAnthropicProvider(unittest.TestCase):
    messages = [
        {"role": "system", "content": "你是翻译。"},
        {"role": "user", "content": "x"},
    ]

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

    def test_usage_normalization_treats_cache_write_as_miss(self):
        from trans_novel.llm.providers.anthropic import normalize_anthropic_usage

        usage = {
            "input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 20,
            "output_tokens": 7,
        }
        sample = normalize_anthropic_usage(usage)

        self.assertEqual(sample.cache_miss_tokens, 15)
        self.assertEqual(sample.cache_hit_tokens, 20)
        self.assertEqual(sample.prompt_tokens, 35)
        self.assertEqual(sample.completion_tokens, 7)
        self.assertEqual(sample.total_tokens, 42)

    def test_usage_normalization_handles_missing_usage(self):
        from trans_novel.llm.providers.anthropic import normalize_anthropic_usage

        self.assertIsNone(normalize_anthropic_usage(None))

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
            prompt_file = argv[argv.index("--system-prompt-file") + 1]
            with open(prompt_file, "r", encoding="utf-8") as f:
                captured["system_prompt_file_content"] = f.read()

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
        self.assertIn("--system-prompt-file", captured["argv"])
        self.assertNotIn("--system-prompt", captured["argv"])
        self.assertEqual(captured["system_prompt_file_content"], "你是翻译。")
        self.assertIn("--model", captured["argv"])
        self.assertIn("claude-opus-4-8", captured["argv"])

        summary = client.usage_summary()
        self.assertEqual(summary["totals"]["prompt_tokens"], 10)
        self.assertEqual(summary["totals"]["completion_tokens"], 5)

    def test_complete_cleans_up_system_prompt_temp_file(self):
        import json
        import os
        from unittest.mock import patch

        from trans_novel.config import LLMConfig
        from trans_novel.llm.providers.anthropic import AnthropicClient

        cfg = LLMConfig(tiers={"strong": {"model": "m"}})
        client = AnthropicClient(cfg)

        captured = {}

        def fake_run(argv, *, input, capture_output, text, timeout, encoding):
            prompt_file = argv[argv.index("--system-prompt-file") + 1]
            captured["prompt_file"] = prompt_file
            self.assertTrue(os.path.exists(prompt_file))

            class Result:
                returncode = 0
                stdout = json.dumps({"is_error": False, "result": "ok", "usage": None})
                stderr = ""

            return Result()

        with patch("shutil.which", return_value="/usr/bin/claude"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            client.complete(
                [
                    {"role": "system", "content": "sys with <angle> brackets"},
                    {"role": "user", "content": "x"},
                ]
            )

        self.assertFalse(os.path.exists(captured["prompt_file"]))

    def test_complete_omits_system_prompt_file_flag_when_no_system_message(self):
        import json
        from unittest.mock import patch

        from trans_novel.config import LLMConfig
        from trans_novel.llm.providers.anthropic import AnthropicClient

        cfg = LLMConfig(tiers={"strong": {"model": "m"}})
        client = AnthropicClient(cfg)

        captured = {}

        def fake_run(argv, *, input, capture_output, text, timeout, encoding):
            captured["argv"] = argv

            class Result:
                returncode = 0
                stdout = json.dumps({"is_error": False, "result": "ok", "usage": None})
                stderr = ""

            return Result()

        with patch("shutil.which", return_value="/usr/bin/claude"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            client.complete([{"role": "user", "content": "x"}])

        self.assertNotIn("--system-prompt-file", captured["argv"])

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


class TestCodexProvider(unittest.TestCase):
    def test_prompt_preserves_roles_and_json_instruction(self):
        from trans_novel.llm.providers.codex import build_codex_prompt

        prompt = build_codex_prompt(
            [
                {"role": "system", "content": "你是译者。"},
                {"role": "user", "content": "翻译这句话。"},
            ],
            json_mode=True,
        )

        self.assertIn("[SYSTEM]", prompt)
        self.assertIn("[USER]", prompt)
        self.assertIn("valid JSON", prompt)

    def test_parse_jsonl_result_and_usage(self):
        import json

        from trans_novel.llm.providers.codex import parse_codex_events

        output = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "t"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "你好"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 4,
                            "output_tokens": 3,
                        },
                    }
                ),
            ]
        )

        text, usage = parse_codex_events(output)

        self.assertEqual(text, "你好")
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.cache_hit_tokens, 4)
        self.assertEqual(usage.cache_miss_tokens, 6)
        self.assertEqual(usage.total_tokens, 13)

    def test_complete_invokes_codex_exec(self):
        import json
        from unittest.mock import patch

        from trans_novel.config import LLMConfig
        from trans_novel.llm.providers.codex import CodexClient

        client = CodexClient(LLMConfig(tiers={"strong": {"model": "m"}}))
        captured = {}

        def fake_run(argv, *, input, capture_output, text, timeout, encoding):
            captured["argv"] = argv
            captured["input"] = input

            class Result:
                returncode = 0
                stdout = "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {"type": "agent_message", "text": "ok"},
                            }
                        ),
                        json.dumps({"type": "turn.completed", "usage": None}),
                    ]
                )
                stderr = ""

            return Result()

        with patch("shutil.which", return_value="/usr/bin/codex"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            result = client.complete([{"role": "user", "content": "x"}])

        self.assertEqual(result, "ok")
        self.assertEqual(captured["argv"][:2], ["/usr/bin/codex", "exec"])
        self.assertIn("--json", captured["argv"])
        self.assertIn("--sandbox", captured["argv"])
        self.assertIn("mcp_servers={}", captured["argv"])
        self.assertEqual(captured["input"], "[USER]\nx")


class TestProviderFactory(unittest.TestCase):
    def _config(
        self,
        provider: str,
        *,
        base_url: str | None = None,
        reasoning_style: str | None = None,
    ):
        from trans_novel.config import Config

        llm = {
            "provider": provider,
            "tiers": {"strong": {"model": "m"}},
        }
        if base_url is not None:
            llm["base_url"] = base_url
        if reasoning_style is not None:
            llm["reasoning_style"] = reasoning_style
        return Config.from_dict({"llm": llm})

    def test_builds_each_provider_from_its_own_module(self):
        from trans_novel.llm.factory import build_client
        from trans_novel.llm.providers.anthropic import AnthropicClient
        from trans_novel.llm.providers.codex import CodexClient
        from trans_novel.llm.providers.ollama import OllamaClient
        from trans_novel.llm.providers.openai import OpenAIClient
        from trans_novel.llm.providers.openai_compatible import (
            OpenAICompatibleClient,
        )
        from trans_novel.llm.providers.openrouter import OpenRouterClient
        from trans_novel.llm.providers.vllm import VLLMClient

        cases = (
            ("openai", OpenAIClient, None),
            ("anthropic", AnthropicClient, None),
            ("codex", CodexClient, None),
            ("openrouter", OpenRouterClient, None),
            ("openai-compatible", OpenAICompatibleClient, "https://example.test/v1"),
            ("ollama", OllamaClient, None),
            ("vllm", VLLMClient, None),
        )
        for provider, expected_type, base_url in cases:
            with self.subTest(provider=provider):
                self.assertIsInstance(
                    build_client(self._config(provider, base_url=base_url)),
                    expected_type,
                )


    def test_local_provider_defaults(self):
        from trans_novel.llm.factory import build_client
        from trans_novel.llm.providers.ollama import OllamaClient
        from trans_novel.llm.providers.vllm import VLLMClient

        ollama = build_client(self._config("ollama"))
        vllm = build_client(self._config("vllm"))
        assert isinstance(ollama, OllamaClient)
        assert isinstance(vllm, VLLMClient)

        self.assertEqual(ollama.base_url, "http://localhost:11434/v1")
        self.assertEqual(vllm.base_url, "http://localhost:8000/v1")
        self.assertFalse(ollama.requires_api_key)
        self.assertFalse(vllm.requires_api_key)

    def test_generic_provider_requires_base_url(self):
        from trans_novel.llm.factory import build_client

        with self.assertRaisesRegex(ValueError, "base_url"):
            build_client(self._config("openai-compatible"))

    def test_compatible_clients_use_configured_reasoning_style(self):
        from trans_novel.llm.factory import build_client
        from trans_novel.llm.providers.ollama import OllamaClient
        from trans_novel.llm.providers.openai_compatible import (
            OpenAICompatibleClient,
        )
        from trans_novel.llm.providers.vllm import VLLMClient

        compatible = build_client(
            self._config(
                "openai-compatible",
                base_url="https://example.test/v1",
                reasoning_style="deepseek",
            )
        )
        ollama = build_client(self._config("ollama", reasoning_style="openai"))
        vllm = build_client(self._config("vllm", reasoning_style="openrouter"))
        assert isinstance(compatible, OpenAICompatibleClient)
        assert isinstance(ollama, OllamaClient)
        assert isinstance(vllm, VLLMClient)

        self.assertEqual(compatible.reasoning_style, "deepseek")
        self.assertEqual(ollama.reasoning_style, "openai")
        self.assertEqual(vllm.reasoning_style, "openrouter")


if __name__ == "__main__":
    unittest.main()
