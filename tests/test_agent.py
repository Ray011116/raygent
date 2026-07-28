"""
Agent 行为测试。

测试策略:
  - 使用 MockLLMClient 替代真实 LLM，获得确定性行为
  - 测试 ReAct 循环逻辑，而非 LLM 能力
  - 验证: 消息构建 / 工具调用流程 / 迭代限制 / 错误处理

学习点:
  - 为什么 Mock 是 Agent 测试的核心？
    → 真实 LLM 对同一问题可能返回不同答案，测试无法做确定性断言
    → Mock 让你精确控制每一步的 LLM 响应，从而测试 Agent 的"决策逻辑"
  - 测试的是什么？
    → 不是"LLM 有没有调用工具"（那是 LLM 的能力）
    → 而是"当 LLM 说调用工具时，Agent 是否正确执行了工具并传递了结果"
"""

from src.agent import ReActAgent
from src.tools import CalculatorTool, SearchTool, ToolRegistry


# ══════════════════════════════════════════════════════════════════════════════
# Mock LLM Client
# ══════════════════════════════════════════════════════════════════════════════

class MockLLMClient:
    """模拟 LLM 客户端，按顺序返回预设响应。

    学习点:
      - responses 是一个列表，每次 chat() 按顺序消费一个
      - call_history 记录每次调用的参数，用于事后验证
      - 如果消费完所有预设响应，返回默认文本（避免 None 错误）
      - 这种模式叫 "Stub"（测试替身的一种），用于替代外部依赖
    """

    def __init__(self, responses: list[dict] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.call_history: list[dict] = []  # 记录每次 chat() 的参数

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        # 记录调用历史，供测试验证
        self.call_history.append({
            "messages": messages,
            "tools": tools,
        })

        # 按顺序消费预设响应
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        # 兜底：预设响应用完了，返回默认文本
        return {"type": "text", "content": "Done", "tool_calls": None}


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def make_text_response(content: str) -> dict:
    """快捷构造：文本响应"""
    return {"type": "text", "content": content, "tool_calls": None}


def make_tool_call_response(name: str, arguments: dict) -> dict:
    """快捷构造：工具调用响应

    学习点:
      - 为什么单独抽一个辅助函数？
        → 减少测试代码中的重复，提高可读性
        → 如果响应格式变化，只需改一处
    """
    return {
        "type": "tool_call",
        "content": None,
        "tool_calls": [{"name": name, "arguments": arguments}],
    }


def make_registry() -> ToolRegistry:
    """创建包含 Calculator 和 Search 的工具注册中心"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentBasicFlow:
    """测试 Agent 的基本 ReAct 循环。

    学习点:
      - 每个测试方法模拟一种 LLM 行为模式
      - 测试的是 Agent 的"决策执行"逻辑，不是 LLM 的"决策"本身
    """

    def test_direct_text_response(self):
        """场景：LLM 直接回答，不需要调用工具

        期望流程:
          1. Agent 收到用户输入 "你好"
          2. LLM 返回文本 "你好！有什么可以帮助你的？"
          3. Agent 直接返回该文本，不调用任何工具
          4. 只调用一次 LLM
        """
        llm = MockLLMClient(responses=[
            make_text_response("你好！有什么可以帮助你的？"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("你好")

        assert result == "你好！有什么可以帮助你的？"
        assert llm.call_count == 1  # 只调用了一次 LLM

    def test_single_tool_call(self):
        """场景：LLM 调用一次工具后给出最终答案

        期望流程:
          1. LLM 返回 tool_call: calculator("1+2")
          2. Agent 执行 calculator，得到结果 "计算结果: 3"
          3. Agent 将工具结果追加到消息历史
          4. LLM 第二次调用返回文本 "答案是3"
          5. Agent 返回 "答案是3"
        """
        llm = MockLLMClient(responses=[
            make_tool_call_response("calculator", {"expression": "1+2"}),
            make_text_response("1+2等于3"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("1+2等于多少？")

        assert result == "1+2等于3"
        assert llm.call_count == 2  # 两次 LLM 调用

    def test_multi_step_tool_calls(self):
        """场景：LLM 连续调用两次工具后才给出最终答案

        期望流程:
          1. LLM 调用 calculator: "2+3"
          2. Agent 执行，得到 "计算结果: 5"
          3. LLM 调用 calculator: "5*4"
          4. Agent 执行，得到 "计算结果: 20"
          5. LLM 返回文本 "最终结果是20"
          6. Agent 返回 "最终结果是20"

        学习点:
          - 这是 ReAct 循环的核心：多步推理
          - 每一步 LLM 都能看到前一步的工具结果（Observation）
          - 这就是为什么消息历史中要包含 tool role 的消息
        """
        llm = MockLLMClient(responses=[
            make_tool_call_response("calculator", {"expression": "2+3"}),
            make_tool_call_response("calculator", {"expression": "5*4"}),
            make_text_response("最终结果是20"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("计算 (2+3)*4")

        assert result == "最终结果是20"
        assert llm.call_count == 3


class TestAgentMessagesStructure:
    """测试 Agent 构建的消息历史结构是否正确。

    学习点:
      - 验证消息结构比验证最终结果更细粒度
      - 如果消息结构不对，LLM 可能收到错误的上下文
      - tool_call_id 必须匹配，否则 LLM 无法关联工具调用和结果
    """

    def test_messages_contain_system_prompt(self):
        """测试：消息历史第一條是 system prompt"""
        llm = MockLLMClient(responses=[
            make_text_response("好的"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())
        agent.run("测试")

        # 第一次 LLM 调用收到的消息
        first_call_messages = llm.call_history[0]["messages"]
        assert first_call_messages[0]["role"] == "system"
        assert "智能助手" in first_call_messages[0]["content"]
        assert "思考" in first_call_messages[0]["content"]
        assert first_call_messages[1]["role"] == "user"
        assert first_call_messages[1]["content"] == "测试"

    def test_tool_call_message_format(self):
        """测试：工具调用后，消息历史包含正确的 assistant 和 tool 消息

        学习点:
          - assistant 消息中 tool_calls 的格式必须符合 OpenAI 规范
          - tool 消息中 tool_call_id 必须与 assistant 消息中的 id 匹配
          - 这是 OpenAI Function Calling 协议的要求，不匹配会报错
        """
        llm = MockLLMClient(responses=[
            make_tool_call_response("calculator", {"expression": "1+1"}),
            make_text_response("结果是2"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())
        agent.run("1+1")

        # 第二次 LLM 调用收到的消息（包含 tool 结果）
        second_call_messages = llm.call_history[1]["messages"]

        # 找到 assistant 消息（包含 tool_calls）
        assistant_msgs = [m for m in second_call_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assistant_msg = assistant_msgs[0]
        assert assistant_msg["tool_calls"] is not None
        assert len(assistant_msg["tool_calls"]) == 1
        call_id = assistant_msg["tool_calls"][0]["id"]

        # 找到对应的 tool 消息
        tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        tool_msg = tool_msgs[0]
        assert tool_msg["tool_call_id"] == call_id  # ID 必须匹配
        assert "2" in tool_msg["content"]  # 计算结果

    def test_tools_passed_to_llm(self):
        """测试：工具 schemas 被正确传递给 LLM"""
        llm = MockLLMClient(responses=[
            make_text_response("好的"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())
        agent.run("测试")

        # 验证 LLM 收到了工具定义
        tools_sent = llm.call_history[0]["tools"]
        assert tools_sent is not None
        tool_names = [t["function"]["name"] for t in tools_sent]
        assert "calculator" in tool_names
        assert "search" in tool_names


class TestAgentLimits:
    """测试 Agent 的边界和限制。"""

    def test_max_iterations_reached(self):
        """场景：LLM 一直调用工具，达到 max_iterations 上限

        期望流程:
          1. LLM 调用工具（第1次）
          2. Agent 执行工具
          3. LLM 调用工具（第2次）
          4. Agent 执行工具
          ...（重复到 max_iterations=3）
          6. Agent 返回错误信息，不再继续

        学习点:
          - max_iterations 是防止无限循环的安全阀
          - 如果 LLM 陷入"调用工具→不满意→再调用工具"的死循环，
            max_iterations 确保 Agent 最终会退出
          - 这是一个重要的可靠性设计
        """
        # 所有响应都是 tool_call，LLM 永远不会给出最终答案
        infinite_tool_calls = [
            make_tool_call_response("calculator", {"expression": "1+1"})
            for _ in range(10)
        ]
        llm = MockLLMClient(responses=infinite_tool_calls)
        agent = ReActAgent(
            llm_client=llm,
            tool_registry=make_registry(),
            max_iterations=3,
        )

        result = agent.run("一直算")

        assert "未能完成" in result
        assert "3" in result  # 提示最大尝试次数
        assert llm.call_count == 3  # 恰好 3 次，没有多调用

    def test_no_responses_provided(self):
        """场景：MockLLMClient 没有预设响应（兜底行为）

        期望：返回默认文本 "Done"，不崩溃
        """
        llm = MockLLMClient(responses=[])  # 空列表
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("随便问")

        assert result == "Done"  # 兜底响应


class TestAgentMessagesReset:
    """测试 Agent 的消息历史管理。"""

    def test_messages_reset_on_new_run(self):
        """测试：每次 run() 都会重置消息历史

        学习点:
          - 每次 run() 是独立会话，不保留上一轮对话
          - 如果要支持多轮对话，需要修改 agent.run() 的逻辑
          - 当前设计是"一问一答"模式
        """
        llm = MockLLMClient(responses=[
            make_text_response("第一次回答"),
            make_text_response("第二次回答"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        agent.run("第一个问题")
        agent.run("第二个问题")

        # 第二次 run() 时，消息历史应该只有 2 条（system + 新问题）
        second_call_messages = llm.call_history[1]["messages"]
        assert len(second_call_messages) == 2  # system + user
        assert second_call_messages[1]["content"] == "第二个问题"
        # 不应该包含 "第一个问题" 的历史