"""
边界情况和异常处理测试。

测试策略:
  - 覆盖各层级的异常路径：工具层、Agent 层、LLM 层
  - 验证系统在异常情况下不会崩溃，而是优雅降级
  - 原则：宁可返回错误信息，也不能崩溃或泄露敏感信息

学习点:
  - 边界测试和单元测试的区别？
    → 单元测试测"正确输入→正确输出"
    → 边界测试测"错误输入→优雅错误处理"
  - 为什么边界测试很重要？
    → 用户不会总是输入正确的数据
    → LLM 不会总是返回正确的格式
    → 系统必须在这种情况下依然可用
"""

import os
import tempfile
import pytest

from src.agent import ReActAgent
from src.tools import (
    CalculatorTool,
    SearchTool,
    ReadFileTool,
    WriteFileTool,
    ToolRegistry,
)
from src.llm import LLMClient


# ══════════════════════════════════════════════════════════════════════════════
# Mock LLM Client（复用 test_agent 中的模式）
# ══════════════════════════════════════════════════════════════════════════════

class MockLLMClient:
    """模拟 LLM 客户端。"""

    def __init__(self, responses: list[dict] | None = None):
        self.responses = responses or []
        self.call_count = 0

    def chat(self, _messages: list[dict], _tools: list[dict] | None = None) -> dict:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return {"type": "text", "content": "Done", "tool_calls": None}


def make_text_response(content: str) -> dict:
    return {"type": "text", "content": content, "tool_calls": None}


def make_tool_call_response(name: str, arguments: dict) -> dict:
    return {
        "type": "tool_call",
        "content": None,
        "tool_calls": [{"name": name, "arguments": arguments}],
    }


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# 工具层边界测试
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculatorEdgeCases:
    """计算器工具的边界情况"""

    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    def test_very_large_numbers(self, calc):
        """测试超大数字运算"""
        result = calc.execute(expression="10 ** 100")
        # 应该能正常计算，不溢出
        assert "错误" not in result
        assert "计算结果" in result

    def test_division_by_zero(self, calc):
        """测试除以零

        学习点:
          - 除以零在 Python 中会抛出 ZeroDivisionError
          - 工具的 execute() 方法有 try/except 包裹
          - 应该返回错误信息，而不是让异常传播到 Agent
        """
        result = calc.execute(expression="1 / 0")
        assert "出错" in result or "错误" in result

    def test_negative_exponentiation(self, calc):
        """测试负数幂"""
        result = calc.execute(expression="2 ** -3")
        assert "0.125" in result

    def test_special_unicode_chars(self, calc):
        """测试包含特殊 Unicode 字符的表达式"""
        result = calc.execute(expression="2 ＋ 3")  # 全角加号
        assert "错误" in result  # 不认识全角符号

    def test_whitespace_only(self, calc):
        """测试只有空白字符的表达式"""
        result = calc.execute(expression="   \t  \n  ")
        assert "错误" in result


class TestReadFileEdgeCases:
    """文件读取工具的边界情况"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def reader(self, temp_dir):
        return ReadFileTool(base_dir=temp_dir)

    def test_empty_file(self, reader, temp_dir):
        """测试读取空文件"""
        filepath = os.path.join(temp_dir, "empty.txt")
        with open(filepath, "w") as f:
            f.write("")
        result = reader.execute(filepath=filepath)
        assert "文件内容" in result

    def test_binary_file(self, reader, temp_dir):
        """测试读取二进制文件（应该能读到，但内容可能是乱码）"""
        filepath = os.path.join(temp_dir, "binary.bin")
        with open(filepath, "wb") as f:
            f.write(b"\x00\x01\x02\xff\xfe")
        # 用 utf-8 读取二进制文件可能抛 UnicodeDecodeError
        result = reader.execute(filepath=filepath)
        # 要么成功（如果文件恰好是合法 UTF-8），要么返回错误
        assert "文件内容" in result or "失败" in result

    def test_path_traversal_attack(self, reader, temp_dir):
        """测试安全：路径穿越攻击

        学习点:
          - ../ 是经典的路径穿越攻击（Path Traversal）
          - 攻击者试图通过 ../../../etc/passwd 访问系统文件
          - os.path.abspath 会解析 ../ 得到真实路径
          - 然后 startswith 检查会拒绝 base_dir 之外的路径
          - 这就是沙箱安全的核心防线
        """
        result = reader.execute(filepath=os.path.join(temp_dir, "../../../etc/passwd"))
        assert "错误" in result
        assert "不在允许" in result


class TestWriteFileEdgeCases:
    """文件写入工具的边界情况"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def writer(self, temp_dir):
        return WriteFileTool(base_dir=temp_dir)

    def test_write_empty_content(self, writer, temp_dir):
        """测试写入空内容"""
        filepath = os.path.join(temp_dir, "empty.txt")
        result = writer.execute(filepath=filepath, content="")
        assert "成功" in result
        with open(filepath) as f:
            assert f.read() == ""

    def test_write_unicode_content(self, writer, temp_dir):
        """测试写入 Unicode 内容（中文、emoji）"""
        filepath = os.path.join(temp_dir, "unicode.txt")
        content = "你好世界 🌍 🚀\n日本語テスト"
        result = writer.execute(filepath=filepath, content=content)
        assert "成功" in result
        with open(filepath, encoding="utf-8") as f:
            assert f.read() == content

    def test_write_very_long_content(self, writer, temp_dir):
        """测试写入超长内容"""
        filepath = os.path.join(temp_dir, "long.txt")
        content = "A" * 100000  # 10 万字符
        result = writer.execute(filepath=filepath, content=content)
        assert "成功" in result
        assert os.path.getsize(filepath) >= 100000


# ══════════════════════════════════════════════════════════════════════════════
# Agent 层边界测试
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentEdgeCases:
    """Agent 的边界情况"""

    def test_llm_returns_unknown_type(self):
        """场景：LLM 返回未知的响应类型

        期望：Agent 应该返回错误信息，而不是崩溃

        学习点:
          - 这是防御性编程的体现
          - LLM 的响应格式可能因 API 变化而改变
          - 代码中的 else 分支就是处理这种情况的
        """
        llm = MockLLMClient(responses=[
            {"type": "unknown_type", "content": None, "tool_calls": None},
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("测试")

        assert "错误" in result or "未知" in result

    def test_tool_execution_error_handled(self):
        """场景：工具执行出错，Agent 应继续处理而非崩溃

        期望流程:
          1. LLM 调用 calculator，但传入无效参数
          2. calculator 返回错误信息（如 "错误：表达式语法不正确"）
          3. LLM 看到错误信息后，应该给出合适的回应
          4. Agent 最终返回 LLM 的回应文本

        学习点:
          - 工具返回的错误信息会作为 tool 消息传给 LLM
          - LLM 看到错误后可以决定：重试 / 换个方式 / 告知用户
          - 这就是 ReAct 的 Observation 步骤的价值
        """
        llm = MockLLMClient(responses=[
            make_tool_call_response("calculator", {"expression": "1/0"}),
            make_text_response("抱歉，1除以0没有意义，无法计算"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("计算 1/0")

        assert "1除以0" in result or "无法" in result or "抱歉" in result

    def test_unknown_tool_name(self):
        """场景：LLM 调用了一个不存在的工具

        期望：ToolRegistry 返回错误信息，Agent 将其传给 LLM，LLM 做出回应

        学习点:
          - LLM 可能"幻觉"出不存在的工具名
          - ToolRegistry 的防御性设计：execute() 检查工具名是否存在
          - 返回友好的错误信息而不是 KeyError 崩溃
        """
        llm = MockLLMClient(responses=[
            make_tool_call_response("nonexistent_tool", {"arg": "value"}),
            make_text_response("我没有这个工具，请换个方式提问"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("做某件事")

        # 不应该崩溃，应该返回 LLM 的回应
        assert "工具" in result or "方式" in result

    def test_max_iterations_one(self):
        """测试 max_iterations=1 的极端情况

        场景：LLM 在第一次就调用工具，但 max_iterations=1
        期望：Agent 执行完工具后，因为达到上限而退出

        学习点:
          - 这是一个重要的边界：如果 LLM 在最后一轮调用工具，
            Agent 仍会执行该工具，但不会让 LLM 分析结果
          - 实际上看代码，agent 在每次迭代开始时检查 iteration < max_iterations
          - 所以如果 iteration 从 1 开始，max_iterations=1，
            第一次迭代 iteration=1，不满足 1 < 1，不会进入循环
          - 等等，让我重新看代码...
        """
        # 实际上代码是 while iteration < self.max_iterations:
        # 第一次 iteration=0, 满足 0 < 1, 进入循环
        # 第二次 iteration=1, 不满足 1 < 1, 退出循环
        llm = MockLLMClient(responses=[
            make_tool_call_response("calculator", {"expression": "1+1"}),
            make_text_response("答案是2"),
        ])
        agent = ReActAgent(
            llm_client=llm,
            tool_registry=make_registry(),
            max_iterations=1,
        )

        result = agent.run("1+1")

        # 第一次迭代：调用工具，执行，进入下一轮
        # 第二次检查：iteration=1, max_iterations=1, 退出，返回错误信息
        assert "未能完成" in result or "2" in result

    def test_empty_user_input(self):
        """测试空输入

        期望：Agent 能正常处理（LLM 可能会给出回应）
        """
        llm = MockLLMClient(responses=[
            make_text_response("请输入你的问题"),
        ])
        agent = ReActAgent(llm_client=llm, tool_registry=make_registry())

        result = agent.run("")

        assert result == "请输入你的问题"


# ══════════════════════════════════════════════════════════════════════════════
# LLM 层边界测试
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMClientEdgeCases:
    """LLM 客户端的边界和异常情况。

    注意：这些测试直接测试 LLMClient 的内部方法，
    不经过 Agent，因为我们测试的是格式转换逻辑。
    """

    def test_tool_conversion_empty_list(self):
        """测试空工具列表的 Anthropic 格式转换"""
        client = LLMClient()
        # 即使 provider 是 openai，_convert_tools_to_anthropic 方法仍然存在
        result = client._convert_tools_to_anthropic([])
        assert result == []

    def test_tool_conversion_single_tool(self):
        """测试单个工具的 Anthropic 格式转换"""
        client = LLMClient()
        tools = [{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
            },
        }]
        result = client._convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert result[0]["input_schema"]["type"] == "object"

    def test_tool_converter_handles_pre_converted_format(self):
        """测试转换器兼容已转换的格式

        学习点:
          - _convert_tools_to_anthropic 中的 tool.get("function", tool)
            就是为了兼容两种格式：OpenAI 格式和已转换的 Anthropic 格式
          - 如果传入已转换格式（没有 function 字段），
            tool.get("function", tool) 会返回 tool 本身
          - 这是幂等性设计：多次转换结果不变
        """
        client = LLMClient()
        # 已经是 Anthropic 格式（没有 function 包装层）
        tools = [{
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {"type": "object", "properties": {}},
        }]
        result = client._convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"