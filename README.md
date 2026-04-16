# BenchLocal Python Benchmark Suite - 使用说明

## 目录结构

```
benchmarks/
├── .env                    # 模型配置（需自行创建）
├── run_benchmarks.py       # 统一入口：跑全部/选定基准，生成报告
├── toolcall15.py           # ToolCall-15 基准
├── reasonmath15.py         # ReasonMath-15 基准
├── instructfollow15.py     # InstructFollow-15 基准
├── dataextract15.py        # DataExtract-15 基准
├── bugfind15.py            # BugFind-15 基准
├── structoutput15.py       # StructOutput-15 基准
├── hermesagent20.py        # HermesAgent-20 基准
└── BENCHMARK_USAGE.md      # 本文档
```

> **注意：** `.env` 文件需放在运行命令的目录中。建议 `cd benchmarks/` 后再执行脚本。

## 概览

本套件包含 7 个独立的 LLM 评测基准，均通过 OpenAI 兼容 API 调用模型并在本地评分。`run_benchmarks.py` 是统一入口，可一次性运行全部或部分基准并生成汇总报告。

| 基准 | 脚本 | 场景数 | 测试能力 |
|------|------|--------|----------|
| ToolCall-15 | `toolcall15.py` | 15 | 工具调用：选择、参数、链式调用、错误处理 |
| ReasonMath-15 | `reasonmath15.py` | 15 | 数学推理：算术、逻辑、陷阱题、应用题 |
| InstructFollow-15 | `instructfollow15.py` | 15 | 指令遵循：格式约束、排序、对抗性指令 |
| DataExtract-15 | `dataextract15.py` | 15 | 数据提取：从非结构化文本提取 JSON |
| BugFind-15 | `bugfind15.py` | 15 | 代码调试：Python/JS/Rust/Go 多语言 Bug 定位与修复 |
| StructOutput-15 | `structoutput15.py` | 15 | 结构化输出：JSON/CSV/YAML/XML/SQL 等格式生成 |
| HermesAgent-20 | `hermesagent20.py` | 20 | Agent 能力：记忆、编排、技能、调度、恢复 |

---

## 环境要求

- Python 3.10+
- `requests` 库

```bash
pip install requests
```

StructOutput-15 的 YAML/TOML 验证可选安装：

```bash
pip install pyyaml    # YAML 场景验证（SO-03）
# TOML 使用 Python 3.11+ 内置 tomllib，无需额外安装
```

---

## 模型配置

所有模型配置通过 **`.env` 文件**或**环境变量**完成。在项目根目录创建 `.env` 文件：

### .env 文件格式

```bash
# ─── 模型列表 ───
# 格式: provider:model_name，多个用逗号分隔
LLM_MODELS=openrouter:openai/gpt-4.1,ollama:qwen3:8b

# 可选第二组模型（与 LLM_MODELS 合并运行）
LLM_MODELS_2=openai_compatible:my-model

# ─── API Keys ───
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx

# ─── 本地/自部署服务地址 ───
OLLAMA_HOST=http://localhost:11434
LLAMACPP_HOST=http://localhost:8080
MLX_HOST=http://localhost:8080
LMSTUDIO_HOST=http://localhost:1234
OPENAI_COMPATIBLE_HOST=https://your-server.com:8443

# ─── 可选：全局请求超时（秒）───
MODEL_REQUEST_TIMEOUT_SECONDS=60
```

### 支持的 Provider

| Provider | 模型格式 | 需要的环境变量 | 说明 |
|----------|----------|---------------|------|
| `openrouter` | `openrouter:openai/gpt-4.1` | `OPENROUTER_API_KEY` | 云端路由，支持数百种模型 |
| `ollama` | `ollama:qwen3:8b` | `OLLAMA_HOST` | 本地 Ollama 服务 |
| `llamacpp` | `llamacpp:my-model` | `LLAMACPP_HOST` | llama.cpp server |
| `mlx` | `mlx:mlx-model` | `MLX_HOST` | Apple MLX 推理服务 |
| `lmstudio` | `lmstudio:model-name` | `LMSTUDIO_HOST` | LM Studio 本地服务 |
| `openai_compatible` | `openai_compatible:model-id` | `OPENAI_COMPATIBLE_HOST` | 任意 OpenAI 兼容 API |

### 配置示例

**使用 OpenRouter 测试云端模型：**

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_MODELS=openrouter:openai/gpt-4.1,openrouter:anthropic/claude-sonnet-4
```

**使用本地 Ollama：**

```bash
OLLAMA_HOST=http://localhost:11434
LLM_MODELS=ollama:qwen3:8b,ollama:llama3.1:8b
```

**混合云端 + 本地对比：**

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OLLAMA_HOST=http://localhost:11434
LLM_MODELS=openrouter:openai/gpt-4.1,ollama:qwen3:8b
```

**自部署 vLLM / SGLang 等 OpenAI 兼容服务：**

```bash
OPENAI_COMPATIBLE_HOST=https://your-gpu-server.com:8443
LLM_MODELS=openai_compatible:Qwen/Qwen3-30B-A3B
```

> **注意：** Host 地址会自动补全 `/v1` 路径。例如 `http://localhost:11434` 会变成 `http://localhost:11434/v1`。如果你的服务已经包含 `/v1`，不会重复添加。

---

## 使用方式

```bash
cd benchmarks/
```

### 运行全部基准

```bash
python run_benchmarks.py
```

### 运行指定基准

```bash
# 单个
python run_benchmarks.py --bench toolcall15

# 多个
python run_benchmarks.py --bench reasonmath15 --bench bugfind15 --bench instructfollow15
```

### 指定模型（覆盖 .env 中的 LLM_MODELS）

```bash
python run_benchmarks.py --model openrouter:openai/gpt-4.1

# 多个模型对比
python run_benchmarks.py --model openrouter:openai/gpt-4.1 --model ollama:qwen3:8b
```

### 运行特定场景

```bash
# 只跑 RM-01 和 RM-10（会传递给对应 benchmark 脚本）
python run_benchmarks.py --bench reasonmath15 --scenario RM-01 --scenario RM-10
```

### 调整生成参数

```bash
python run_benchmarks.py --temperature 0.0 --top-p 0.9 --timeout 120
```

### 输出格式

```bash
# 终端文本报告（默认）
python run_benchmarks.py

# JSON 报告
python run_benchmarks.py --json

# Markdown 报告
python run_benchmarks.py --markdown

# 带详细 trace 日志
python run_benchmarks.py --show-raw
```

### 保存结果

```bash
# 保存到目录（每个 benchmark 一个 JSON + 汇总 report.json + report.md）
python run_benchmarks.py --save-to results/2026-04-16

# 目录结构：
# results/2026-04-16/
# ├── toolcall15.json        # ToolCall-15 原始结果
# ├── reasonmath15.json      # ReasonMath-15 原始结果
# ├── instructfollow15.json
# ├── dataextract15.json
# ├── bugfind15.json
# ├── structoutput15.json
# ├── hermesagent20.json
# ├── report.json            # 汇总 JSON 报告
# └── report.md              # 汇总 Markdown 报告
```

### 从已保存结果重新生成报告

```bash
# 不重新跑测试，直接从保存的 JSON 生成报告
python run_benchmarks.py --report-only results/2026-04-16

# 生成 Markdown
python run_benchmarks.py --report-only results/2026-04-16 --markdown

# 生成 JSON
python run_benchmarks.py --report-only results/2026-04-16 --json
```

---

## 单独运行某个基准

每个 `.py` 文件都可以独立运行，CLI 参数一致：

```bash
python toolcall15.py --json
python reasonmath15.py --scenario RM-01 --show-raw
python bugfind15.py --model openrouter:openai/gpt-4.1 --temperature 0
python dataextract15.py --json --model ollama:qwen3:8b
```

---

## 报告示例

### 终端输出

```
======================================================================
  BenchLocal - Full Benchmark Report
  Generated: 2026-04-16 10:30:00 UTC
======================================================================

  Model: openrouter:openai/gpt-4.1
----------------------------------------------------------------------

  Benchmark              Score    Rating
  ---------------------- -------  --------------------
  ToolCall-15               87/100  ★★★★ Good
  ReasonMath-15             92/100  ★★★★★ Excellent
  InstructFollow-15         78/100  ★★★★ Good
  DataExtract-15            85/100  ★★★★ Good
  BugFind-15                73/100  ★★★ Adequate
  StructOutput-15           80/100  ★★★★ Good
  HermesAgent-20            65/100  ★★★ Adequate
  ---------------------- -------  --------------------
  OVERALL                   80/100  ★★★★ Good

  Category Breakdown:

    ToolCall-15:
      Tool Selection               ████████████████░░░░  83%
      Parameter Precision          ██████████████████░░  90%
      ...
```

### Markdown 报告

生成的 `report.md` 包含：
- 每个模型的分数汇总表格
- 每个 benchmark 的分类维度细分
- 运行耗时统计

---

## 评分体系

| 基准 | 评分方式 | 分数范围 |
|------|----------|----------|
| ToolCall-15 | 每题 0/1/2 分，5 类加权平均 | 0-100 |
| ReasonMath-15 | 答案轴(70%) + 推理轴(30%) | 0-100 |
| InstructFollow-15 | 约束通过率 | 0-100 |
| DataExtract-15 | JSON 字段正确率 | 0-100 |
| BugFind-15 | 识别(35%) + 修复(40%) + 纪律(25%) | 0-100 |
| StructOutput-15 | 可解析(40%) + 正确性(35%) + 纪律(25%) | 0-100 |
| HermesAgent-20 | 关键词匹配评分 | 0-100 |

**星级评定：**

| 分数 | 评级 |
|------|------|
| 90+ | ★★★★★ Excellent |
| 75-89 | ★★★★ Good |
| 60-74 | ★★★ Adequate |
| 40-59 | ★★ Weak |
| 0-39 | ★ Poor |

---

## 常见参数速查

| 参数 | 说明 | 示例 |
|------|------|------|
| `--bench NAME` | 指定基准（可重复） | `--bench toolcall15 --bench bugfind15` |
| `--model ID` | 指定模型（可重复） | `--model openrouter:openai/gpt-4.1` |
| `--models IDS` | 逗号分隔模型列表 | `--models ollama:a,ollama:b` |
| `--scenario ID` | 指定场景（可重复） | `--scenario RM-01` |
| `--temperature F` | 采样温度 | `--temperature 0.0` |
| `--timeout N` | 单次请求超时（秒） | `--timeout 120` |
| `--bench-timeout N` | 单个基准超时（分钟） | `--bench-timeout 60` |
| `--json` | JSON 格式输出 | |
| `--markdown` | Markdown 格式输出 | |
| `--save-to DIR` | 保存结果到目录 | `--save-to results/run1` |
| `--report-only DIR` | 从已保存结果生成报告 | `--report-only results/run1` |
| `--show-raw` | 显示详细 trace 日志 | |
