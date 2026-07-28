"""
RayGent CLI - 命令行交互入口

启动方式: python -m src.cli
"""

import sys
import os
import logging
from src.llm import LLMClient
from src.tools import (
    CalculatorTool,
    SearchTool,
    ReadFileTool,
    WriteFileTool,
    ToolRegistry,
)
from src.agent import ReActAgent

# 配置日志级别：INFO 会显示每次迭代的工具调用，DEBUG 更详细
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 配置常量
# ══════════════════════════════════════════════════════════════════════════════

WELCOME_TEXT = """
╔══════════════════════════════════════════╗
║         🧠 RayGent Agent                ║
║    基于 ReAct 的智能助手                 ║
╚══════════════════════════════════════════╝

可用工具: 计算器 | 搜索 | 文件读写
输入 /help 查看帮助，/exit 退出
"""

HELP_TEXT = """
┌──────────────────────────────────────────┐
│  命令列表                                │
├──────────────────────────────────────────┤
│  /exit, /quit    退出程序                │
│  /help           显示此帮助              │
│  /tools          列出可用工具            │
│  /history        显示当前对话轮次        │
│  /clear          清空对话历史            │
│                                          │
│  直接输入问题即可与 Agent 对话           │
└──────────────────────────────────────────┘
"""

# ══════════════════════════════════════════════════════════════════════════════
# 工具初始化
# ══════════════════════════════════════════════════════════════════════════════


def create_tool_registry(base_dir: str | None = None) -> ToolRegistry:
    """
    创建并注册所有工具。

    Args:
        base_dir: 文件读写工具的安全根目录，默认为当前工作目录。

    Returns:
        已注册所有工具的 ToolRegistry 实例。

    学习点:
      - 为什么用工厂函数而不是在 main() 里直接写？
        → 可测试性：测试时可以传入临时目录，不影响真实文件。
      - base_dir 参数的作用是什么？
        → 沙箱安全：ReadFileTool/WriteFileTool 只能访问该目录下的文件，
          防止 Agent 读取系统敏感文件（如 /etc/passwd）。
    """
    registry = ToolRegistry()

    # 注册所有工具
    registry.register(CalculatorTool())
    registry.register(SearchTool())

    # 文件读写工具需要 base_dir 做沙箱隔离
    # 如果没传 base_dir，工具内部会用 os.getcwd() 作为默认值
    registry.register(ReadFileTool(base_dir=base_dir))
    registry.register(WriteFileTool(base_dir=base_dir))

    return registry


# ══════════════════════════════════════════════════════════════════════════════
# 命令处理
# ══════════════════════════════════════════════════════════════════════════════


def handle_command(user_input: str, agent: ReActAgent) -> bool | None:
    """
    处理元命令（以 / 开头）。

    Args:
        user_input: 用户输入的原始字符串。
        agent: ReActAgent 实例，某些命令需要访问 agent 状态。

    Returns:
        True  → 需要退出程序
        False → 是元命令，已被处理，不需要让 Agent 回答
        None  → 不是元命令，需要交给 Agent 处理

    学习点:
      - 为什么返回值用 True/False/None 三态？
        → True: 退出信号
        → False: 命令已处理，跳过 Agent 调用
        → None: 不是命令，继续走 Agent 流程
      - 这是典型的"命令模式"简化版。
    """
    cmd = user_input.strip().lower()

    # /exit 和 /quit —— 退出
    if cmd in ("/exit", "/quit"):
        print("👋 再见！")
        return True

    # /help —— 帮助
    elif cmd == "/help":
        print(HELP_TEXT)
        return False

    # /tools —— 列出可用工具
    elif cmd == "/tools":
        print("\n📦 可用工具：")
        for schema in agent.tools.get_schemas():
            func = schema["function"]
            print(f"  • {func['name']}: {func['description']}")
        return False

    # /history —— 显示对话统计
    elif cmd == "/history":
        # 只统计非 system 消息（user + assistant + tool）
        conversation_msgs = [m for m in agent.messages if m["role"] != "system"]
        print(f"\n📊 当前对话: {len(conversation_msgs)} 条消息")
        return False

    # /clear —— 清空对话
    elif cmd == "/clear":
        agent.messages = [{"role": "system", "content": agent._build_system_prompt()}]
        print("🧹 对话历史已清空。")
        return False

    # 空输入
    elif not cmd:
        return False

    # 不是命令，交给 Agent
    else:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════════════


def main():
    """
    程序入口。

    TODO 学习点 - 整体流程:
      1. 初始化 → 2. 打印欢迎信息 → 3. 交互循环 → 4. 退出

      为什么 LLMClient 和 ToolRegistry 在 main() 里创建，而不是在 Agent 内部？
        → 依赖注入（Dependency Injection）:
          Agent 不关心用的是 OpenAI 还是 Anthropic，也不关心有哪些工具。
          这让你在测试时可以注入 mock 对象，是写可测试代码的核心习惯。
    """
    # ---- 初始化阶段 ----

    print("⏳ 正在初始化...")

    # 创建 LLM 客户端（自动从 .env 读取配置）
    llm_client = LLMClient()

    # 创建工具注册中心（当前目录作为文件操作的安全根目录）
    tool_registry = create_tool_registry()

    # 创建 Agent（注入 LLM 和工具，最多 10 轮迭代防止死循环）
    agent = ReActAgent(llm_client=llm_client, tool_registry=tool_registry, max_iterations=10)

    # 打印欢迎信息
    print(WELCOME_TEXT)
    print(f"🤖 模型: {llm_client.model}")

    # ---- 交互循环 ----

    while True:
        try:
            # 读取用户输入
            user_input = input("\n🧑 You > ")

            # 处理元命令
            result = handle_command(user_input, agent)
            if result is True:
                break          # /exit: 退出循环
            elif result is False:
                continue       # 元命令已处理: 跳过 Agent，开始下一轮

            # 交给 Agent 处理
            print("\n🤖 Agent 思考中...")
            response = agent.run(user_input)
            print(f"\n🤖 Agent > {response}")

        except KeyboardInterrupt:
            # 用户按 Ctrl+C
            print("\n\n👋 再见！")
            break

        except Exception as e:
            # LLM 调用异常（网络错误、API Key 错误等）
            # 分层处理：logger 记录完整错误栈供调试，print 给用户看简洁信息
            logger.error(f"运行出错: {e}", exc_info=True)
            print(f"\n❌ 出错了: {e}")
            print("💡 提示: 检查网络连接和 API Key 配置，或输入 /exit 退出。")


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()