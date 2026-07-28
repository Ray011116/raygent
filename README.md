# RayGent

基于 ReAct (Reasoning + Acting) 模式的智能 Agent 框架，支持工具调用和多轮推理。

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                    CLI (cli.py)                  │
│         命令行交互 / 命令处理 / 错误恢复           │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│               ReActAgent (agent.py)              │
│     Thought → Action → Observation 循环          │
│     System Prompt / 消息历史 / 迭代控制            │
└──────────┬──────────────────────┬───────────────┘
           │                      │
┌──────────▼──────────┐  ┌───────▼───────────────┐
│   LLMClient (llm.py) │  │  ToolRegistry (tools.py) │
│  OpenAI / Anthropic  │  │  Calculator / Search     │
│  统一接口 + 格式转换   │  │  ReadFile / WriteFile    │
└──────────────────────┘  └─────────────────────────┘
```

### ReAct 循环

```
用户输入 → LLM 思考 → 需要工具？
                         ├─ 是 → 执行工具 → 观察结果 → 回到思考
                         └─ 否 → 输出最终答案
                         
单次 run() 最多 10 轮迭代，防止死循环。
```

## 快速开始

```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 配置 .env（参考 .env 文件）
# 必需: LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

# 3. 启动 CLI
python -m src.cli
```

## CLI 命令

| 命令 | 作用 |
|------|------|
| `/help` | 显示帮助 |
| `/tools` | 列出可用工具 |
| `/history` | 显示当前对话消息数 |
| `/clear` | 清空对话历史 |
| `/exit` / `/quit` | 退出 |

## 可用工具

| 工具 | 功能 | 安全机制 |
|------|------|----------|
| `calculator` | 安全数学计算（+ - * / // % **） | AST 白名单，禁止函数调用 |
| `search` | 搜索（Mock 实现） | 返回预设数据 |
| `read_file` | 读取文件 | 沙箱：只能访问项目目录 |
| `write_file` | 写入文件 | 沙箱 + 拒绝覆盖已存在文件 |

## 项目结构

```
raygent/
├── src/
│   ├── agent.py          # ReAct Agent 核心逻辑
│   ├── cli.py            # 命令行交互入口
│   ├── llm.py            # LLM 统一调用接口（OpenAI / Anthropic）
│   └── tools.py          # 工具集（Calculator, Search, ReadFile, WriteFile）
├── tests/
│   ├── test_tools.py     # 工具单元测试（33 个）
│   ├── test_agent.py     # Agent 行为测试（9 个，Mock LLM）
│   ├── test_edge_cases.py # 边界和异常测试（19 个）
│   └── test_eval.py      # 评测基线（10 个，真实 LLM）
├── .env                  # 环境配置
├── pytest.ini            # Pytest 配置
└── README.md
```

## 测试

### 快速测试（不含真实 LLM 调用）

```bash
pytest tests/ -v -m "not slow"
```

### 评测基线（需要真实 LLM）

```bash
pytest tests/test_eval.py -v -m slow
```

### 跳过评测

```bash
SKIP_EVAL=1 pytest tests/ -v
```

### 测试分层

| 文件 | 数量 | 类型 | LLM | 速度 |
|------|------|------|-----|------|
| `test_tools.py` | 33 | 单元测试 | Mock | <0.1s |
| `test_agent.py` | 9 | 行为测试 | Mock | <0.1s |
| `test_edge_cases.py` | 19 | 边界测试 | Mock | <1s |
| `test_eval.py` | 10 | 评测基线 | 真实 | ~30s |

## 设计决策

### 依赖注入

`LLMClient` 和 `ToolRegistry` 在 `main()` 中创建，注入到 `ReActAgent`：

- Agent 不关心用的是 OpenAI 还是 Anthropic
- 测试时可以注入 Mock 对象
- 符合"依赖倒置原则"

### 统一 LLM 接口

`LLMClient.chat()` 返回统一格式，无论底层是 OpenAI 还是 Anthropic：

```python
# 文本响应
{"type": "text", "content": "...", "tool_calls": None}

# 工具调用
{"type": "tool_call", "content": None, "tool_calls": [{"name": "...", "arguments": {...}}]}
```

### 安全设计

- **Calculator**: AST 白名单验证，禁止函数调用、变量访问、字符串
- **ReadFile/WriteFile**: 路径沙箱（`os.path.abspath` + `startswith`），防止路径穿越攻击
- **WriteFile**: 拒绝覆盖已存在文件

### 三态命令处理

`handle_command()` 返回三态值：

- `True` → 退出程序
- `False` → 命令已处理，跳过 Agent
- `None` → 不是命令，交给 Agent

## 配置

通过 `.env` 文件配置：

```env
# LLM 提供商: openai / anthropic
LLM_PROVIDER=openai

# OpenAI 兼容接口
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-api-endpoint/v1
OPENAI_MODEL=deepseek-v4-flash

# Agent 配置
MAX_ITERATIONS=10
TEMPERATURE=0.0
```

## 学习路线上下文

此项目是 6 个月 Agent Engineer 学习路线中 Week 1-2 的产出：

- ✅ 核心 ReAct Agent 实现
- ✅ CLI 命令行入口
- ✅ 测试套件（单元 + 行为 + 边界 + 评测）
- ✅ 项目文档

后续计划：Week 3-4 记忆系统、Week 5-6 多 Agent 协作、Week 7-8 RAG 等。