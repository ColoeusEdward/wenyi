# Anthropic Provider 改用本机 Claude Code CLI 设计

**日期**：2026-07-14
**状态**：已批准，待实现

## 背景

`config.yaml` 中 `llm.provider: anthropic` 目前通过 `anthropic` Python SDK 直接调用
Messages API（`trans_novel/llm/providers/anthropic.py` 的 `AnthropicClient`）。用户希望
选择 anthropic provider 时改为调用本机已登录的 `claude` CLI（Claude Code），而不是
Anthropic SDK ——利用本机 OAuth/订阅登录态，不再需要配置 API key / base_url。

## 范围

只改动 `trans_novel/llm/providers/anthropic.py` 及其配套测试；`LLMClient` 抽象接口、
`factory.py`、各 agent（`agents/*.py`）、`pipeline/orchestrator.py` 均不受影响——它们
只依赖 `LLMClient.complete()` / `complete_json()`，对内部是 SDK 还是 CLI 无感知。

## 整体方案

`AnthropicClient` 内部实现从 `anthropic` SDK 换成 `subprocess` 调用本机 `claude` CLI
的非交互模式（`claude -p ... --output-format json`），解析其 JSON 输出得到回复文本和
usage 统计。对外行为（方法签名、返回值、异常语义）保持不变。

### 实测验证过的关键事实

1. `claude --safe-mode --no-session-persistence --output-format json --tools none
   --model <model> -p` （user 内容走 stdin）返回单个 JSON 对象，`result` 字段是回复
   文本，`usage` 字段的形状（`input_tokens` / `cache_creation_input_tokens` /
   `cache_read_input_tokens` / `output_tokens`）与 Anthropic 原生 Messages API 完全一致，
   可以直接喂给现有的 `normalize_anthropic_usage()`。
2. 必须用 `--safe-mode` 而非 `--bare`：`--bare` 强制只认 `ANTHROPIC_API_KEY` /
   `apiKeyHelper`，会绕开 OAuth/订阅登录态；`--safe-mode` 保留正常鉴权、模型选择、权限
   逻辑，只关闭 CLAUDE.md / skills / hooks / 插件等项目定制，避免用户全局配置污染翻译
   prompt，且不影响鉴权来源。
3. Windows 上用 `subprocess.run` 调用 CLI 时必须显式传 `encoding="utf-8"`；默认走
   系统代码页（GBK）解码，中文输出会触发 `UnicodeDecodeError` 或产生乱码（已实测复现）。
4. 可执行文件解析不能死认 `shutil.which("claude")` 的返回值一定可用：本机环境里
   `shutil.which` 命中的 pnpm 全局 shim 因为 NODE_PATH 记录陈旧而无法运行，需要留一个
   显式覆盖入口。

## 配置字段语义调整

`config.yaml` 的 `llm` 段字段处理方式：

| 字段 | 处理方式 |
|---|---|
| `provider: anthropic` | 不变，触发 `AnthropicClient` |
| `base_url` | **不再使用**。CLI 走本机登录态没有 base_url 概念；若配置了则忽略并输出提示日志（不报错） |
| `api_key_env` | **不再使用**，同样忽略并提示 |
| `timeout` | 复用，作为 `subprocess.run(..., timeout=...)` 的超时秒数 |
| `max_retries` | 复用，CLI 调用失败/超时/`is_error: true` 时按原有 tenacity 指数退避重试 |
| `tiers.<tier>.model` | 复用，传给 `--model` |
| `tiers.<tier>.options.thinking` | 复用；`false` 时不传 `--effort`；`true` 时传 `--effort <reasoning_effort>` |
| `tiers.<tier>.options.reasoning_effort` | 复用，作为 `--effort` 的值（`low\|medium\|high\|xhigh\|max`） |
| `tiers.<tier>.options.extra_body` | **不再适用**（那是 API 请求体字段），配置了则忽略并提示 |
| `llm.cli_path`（新增，可选） | 显式指定 `claude` 可执行文件路径；不填则用 `shutil.which("claude")` |

`AnthropicTierOptions` 模型（pydantic，`extra="forbid"`）保持字段名不变，只是
`extra_body` 在这个 provider 下不再被消费（仍允许配置，静默忽略并记录一条提示）。

### 项目 config.yaml 处理

项目根 `config.yaml` 里 anthropic 段现有的 `base_url` / `api_key_env: MY_KEY` 两行
注释掉，并加一句说明：这两项在 CLI 模式下不再需要。`config.py` 内嵌的
`_DEFAULT_CONFIG_YAML`（默认 provider 是 `deepseek`）不受影响，无需改动。

### 依赖处理

`pyproject.toml` 的 `anthropic>=0.80` 依赖予以保留（不再被 `AnthropicClient` 直接
import，但作为公开可选能力的历史依赖不删除，避免影响可能仍在其他地方使用它的场景）。

## `AnthropicClient.complete()` 实现细节

### 消息 → CLI 参数转换

- 复用现有 `_split_system(messages)` 把 system 角色内容提取出来。
- `system_text` 非空时通过 `--system-prompt <text>` 传递。
- `chat_messages`（去掉 system 后的剩余消息）拼接成一段纯文本，通过 **stdin** 传给
  `-p`（不留空触发交互式 stdin 等待警告）。当前代码库所有调用点
  （`agents/base.py`、`pipeline/orchestrator.py`）都只有单条 system + 单条 user，
  多条消息拼接只是兜底，不是当前实际路径。
- json_mode 时复用现有逻辑：在 `system_text` 末尾追加 `_JSON_MODE_INSTRUCTION`
  （"Output must be valid json."）。不使用 CLI 的 `--json-schema`（需要预先定义
  schema 且走工具调用强制路径，和 `parse_json_loose` 的"宽松尽力解析"风格不一致）。

### 固定 CLI 参数（每次调用都带）

```
claude --safe-mode --no-session-persistence --output-format json --tools none
       --model <tier_model> [--effort <reasoning_effort>] -p
```

`--tools none` 禁用工具调用，确保只做纯文本补全。

### 超时与重试

复用现有 `@retry(stop_after_attempt(cfg.max_retries + 1), wait_exponential(...),
retry_if_exception_type(Exception), reraise=True)` 装饰器包装调用函数。
`subprocess.run(..., timeout=cfg.timeout, encoding="utf-8")`。以下情况视为失败并触发
重试：

- `subprocess.TimeoutExpired`
- 进程返回码非 0
- stdout 不是合法 JSON
- 解析出的 JSON 里 `is_error: true`

### 返回值与用量

- 返回值取 JSON 的 `result` 字段（字符串）。
- 用量：JSON 的 `usage` 字段结构与 Anthropic 原生 usage 一致，直接传给现有的
  `normalize_anthropic_usage()` 复用。

### 可执行文件解析与并发

- `_ensure_client()` 语义调整为"解析并校验一次可执行文件路径"（不再有 SDK client 对象）。
- 解析顺序：`cfg.cli_path`（若配置）→ `shutil.which("claude")` → 都没有则
  `RuntimeError`，提示信息包含"可在 config.yaml 的 llm.cli_path 显式指定"。
- 沿用 `threading.Lock()` 保护路径探测过程，避免并发下重复报错的竞态。

## 测试改动

`tests/test_llm.py` 的 `TestAnthropicProvider`：

- `normalize_anthropic_usage()` 相关测试**不变**（输入 usage dict 结构未变）。
- 原 `build_request_kwargs()`（构造 SDK 请求体）相关测试改写为新函数（构造 CLI
  argv + stdin 文本），覆盖：system 提取正确、`--effort` 按 `thinking` 开关有无出现、
  json_mode 追加 instruction 且不修改入参、多档位参数覆盖生效。
- 新增：mock `subprocess.run` 验证 `complete()` 正确拼参数、解析 `result`/`usage`、
  失败重试次数、CLI 不存在时报错信息包含 `cli_path` 关键字。
- `TestProviderFactory` 中 `anthropic` 分支不变（`build_client` 仍返回
  `AnthropicClient` 实例，构造时不再要求 `base_url`）。

## 不做的事（YAGNI）

- 不支持 CLI 的多轮会话/`--resume`（每次调用都是独立无状态子进程，符合
  `--no-session-persistence` 的设计）。
- 不实现流式输出（`--output-format stream-json`）——现有 `LLMClient.complete()`
  接口就是同步返回完整文本，没有消费流式结果的调用方。
- 不改动 `OpenAICompatibleBaseClient` 或其他 provider。
