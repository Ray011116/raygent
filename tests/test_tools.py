"""
工具单元测试。

测试策略:
  - 每个工具独立测试，不依赖 LLM 或其他模块
  - 覆盖: 正常路径 / 错误路径 / 安全边界
  - 文件工具使用 tempfile 创建临时目录，不污染真实文件系统

学习点:
  - 为什么用 tempfile.TemporaryDirectory？
    → 测试结束后自动清理，不会留下垃圾文件
    → 隔离：不同测试之间不互相影响
  - 为什么 Calculator 的安全测试很重要？
    → 如果 AST 检查有漏洞，恶意表达式可能执行任意代码
"""

import os
import tempfile
import pytest

from src.tools import (
    CalculatorTool,
    SearchTool,
    ReadFileTool,
    WriteFileTool,
    ToolRegistry,
)


# ══════════════════════════════════════════════════════════════════════════════
# CalculatorTool 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculatorTool:
    """计算器工具测试。

    学习点:
      - @pytest.fixture 是 pytest 的依赖注入机制
      - 每个测试方法运行前，pytest 自动调用 fixture 创建新实例
      - 这保证了测试之间互不干扰（隔离性）
    """

    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    # ---- 正常路径 ----

    def test_basic_addition(self, calc):
        """测试基本加法"""
        result = calc.execute(expression="2 + 3")
        assert "5" in result

    def test_multiplication_precedence(self, calc):
        """测试运算符优先级：乘法优先于加法"""
        result = calc.execute(expression="2 + 3 * 4")
        assert "14" in result  # 不是 20

    def test_parentheses(self, calc):
        """测试括号改变优先级"""
        result = calc.execute(expression="(2 + 3) * 4")
        assert "20" in result

    def test_unary_minus(self, calc):
        """测试一元负号"""
        result = calc.execute(expression="-5 + 3")
        assert "-2" in result

    def test_exponentiation(self, calc):
        """测试幂运算"""
        result = calc.execute(expression="2 ** 10")
        assert "1024" in result

    def test_floor_division(self, calc):
        """测试整除"""
        result = calc.execute(expression="10 // 3")
        assert "3" in result

    def test_modulo(self, calc):
        """测试取模"""
        result = calc.execute(expression="10 % 3")
        assert "1" in result

    def test_float_result(self, calc):
        """测试浮点结果"""
        result = calc.execute(expression="10 / 3")
        assert "3.3333333333" in result  # round to 10 decimal places

    def test_complex_expression(self, calc):
        """测试复合表达式"""
        result = calc.execute(expression="(1 + 2) * (3 + 4) - 5 ** 2 // 3")
        assert "13" in result  # 3*7 - 25//3 = 21 - 8 = 13

    # ---- 错误路径 ----

    def test_invalid_syntax(self, calc):
        """测试非法语法"""
        result = calc.execute(expression="2 + + 3")
        assert "错误" in result

    def test_empty_expression(self, calc):
        """测试空表达式"""
        result = calc.execute(expression="")
        assert "错误" in result

    # ---- 安全测试 ----

    def test_no_function_calls(self, calc):
        """测试安全：不能调用函数

        学习点:
          - 为什么这是安全测试？
            → 如果 eval() 没有 AST 白名单检查，__import__('os').system('rm -rf /')
              这样的表达式就能执行任意代码
          - AST 白名单 = 只允许特定节点类型（Constant, BinOp, UnaryOp）
          - 函数调用是 ast.Call 节点，不在白名单中，所以被拒绝
        """
        result = calc.execute(expression="__import__('os').system('ls')")
        assert "错误" in result

    def test_no_variable_access(self, calc):
        """测试安全：不能访问变量"""
        result = calc.execute(expression="x + 1")
        assert "错误" in result

    def test_no_string_literals(self, calc):
        """测试安全：不能使用字符串"""
        result = calc.execute(expression="'hello' + 'world'")
        assert "错误" in result


# ══════════════════════════════════════════════════════════════════════════════
# SearchTool 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchTool:
    """搜索工具测试。

    学习点:
      - SearchTool 是 Mock 实现，返回预设数据
      - 测试的是"匹配逻辑"是否正确，而非真实搜索
      - 真正的搜索工具（如 SerpAPI）应该在集成测试中验证
    """

    @pytest.fixture
    def search(self):
        return SearchTool()

    def test_search_beijing_weather(self, search):
        """测试搜索：北京天气（精确匹配）"""
        result = search.execute(query="北京天气")
        assert "25°C" in result

    def test_search_python(self, search):
        """测试搜索：Python"""
        result = search.execute(query="Python")
        assert "编程语言" in result

    def test_search_agent(self, search):
        """测试搜索：Agent"""
        result = search.execute(query="Agent")
        assert "智能体" in result

    def test_search_partial_match(self, search):
        """测试搜索：部分匹配（query 包含关键词）"""
        result = search.execute(query="今天北京天气怎么样")
        assert "25°C" in result  # "北京天气" 是 "今天北京天气怎么样" 的子串

    def test_search_no_result(self, search):
        """测试搜索：无匹配结果"""
        result = search.execute(query="不存在的内容xyz")
        assert "未找到" in result


# ══════════════════════════════════════════════════════════════════════════════
# ReadFileTool 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestReadFileTool:
    """文件读取工具测试。

    学习点:
      - 使用 tempfile.TemporaryDirectory 创建临时目录
      - 测试隔离：每个测试方法用独立的临时目录
      - 安全测试：验证沙箱机制（不能读取 base_dir 之外的文件）
    """

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录作为沙箱。

        学习点:
          - yield 之前是 setup（创建临时目录）
          - yield 之后是 teardown（自动清理临时目录）
          - 这就是 fixture 的强大之处：测试代码只需关注业务逻辑
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def reader(self, temp_dir):
        return ReadFileTool(base_dir=temp_dir)

    def test_read_existing_file(self, reader, temp_dir):
        """测试读取已存在的文件"""
        filepath = os.path.join(temp_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("Hello, World!")
        result = reader.execute(filepath=filepath)
        assert "Hello, World!" in result

    def test_read_non_existent_file(self, reader, temp_dir):
        """测试读取不存在的文件"""
        result = reader.execute(filepath=os.path.join(temp_dir, "nope.txt"))
        assert "错误" in result
        assert "不存在" in result

    def test_read_directory(self, reader, temp_dir):
        """测试读取目录（而非文件）"""
        result = reader.execute(filepath=temp_dir)
        assert "错误" in result
        assert "目录" in result

    def test_read_outside_base_dir(self, reader):
        """测试安全：读取 base_dir 之外的文件应被拒绝

        学习点:
          - 这是沙箱安全的核心测试
          - 即使文件存在（如 /etc/hosts），也不能读取
          - 防止 Agent 被利用来窃取系统文件
        """
        result = reader.execute(filepath="/etc/hosts")
        assert "错误" in result
        assert "不在允许" in result

    def test_read_large_file_truncated(self, reader, temp_dir):
        """测试读取超大文件时截断"""
        filepath = os.path.join(temp_dir, "big.txt")
        with open(filepath, "w") as f:
            f.write("A" * 6000)  # 超过 5000 字符限制
        result = reader.execute(filepath=filepath)
        assert "截断" in result
        assert len(result) < 6000  # 被截断了


# ══════════════════════════════════════════════════════════════════════════════
# WriteFileTool 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteFileTool:
    """文件写入工具测试。"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def writer(self, temp_dir):
        return WriteFileTool(base_dir=temp_dir)

    def test_write_new_file(self, writer, temp_dir):
        """测试写入新文件"""
        filepath = os.path.join(temp_dir, "output.txt")
        result = writer.execute(filepath=filepath, content="Hello")
        assert "成功" in result
        # 验证文件确实被创建且内容正确
        with open(filepath) as f:
            assert f.read() == "Hello"

    def test_write_overwrite_existing(self, writer, temp_dir):
        """测试拒绝覆盖已存在的文件

        学习点:
          - 这是安全策略：防止 Agent 意外覆盖重要文件
          - 如果要允许覆盖，需要修改工具实现
        """
        filepath = os.path.join(temp_dir, "existing.txt")
        with open(filepath, "w") as f:
            f.write("original")
        result = writer.execute(filepath=filepath, content="new content")
        assert "拒绝覆盖" in result
        # 验证原文件内容未被修改
        with open(filepath) as f:
            assert f.read() == "original"

    def test_write_outside_base_dir(self, writer):
        """测试安全：不能写入 base_dir 之外"""
        result = writer.execute(filepath="/etc/malicious.txt", content="bad")
        assert "错误" in result
        assert "不在允许" in result

    def test_write_creates_parent_dirs(self, writer, temp_dir):
        """测试自动创建父目录"""
        filepath = os.path.join(temp_dir, "subdir", "nested", "file.txt")
        result = writer.execute(filepath=filepath, content="nested")
        assert "成功" in result
        assert os.path.exists(filepath)


# ══════════════════════════════════════════════════════════════════════════════
# ToolRegistry 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """工具注册中心测试。

    学习点:
      - ToolRegistry 是"注册表模式"（Registry Pattern）
      - 它不关心工具的具体实现，只负责存储和查找
      - 这种解耦让你可以随时添加新工具而不改 Registry 代码
    """

    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(SearchTool())
        return reg

    def test_register_and_get_schemas(self, registry):
        """测试注册后能获取正确的 schema 列表"""
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = [s["function"]["name"] for s in schemas]
        assert "calculator" in names
        assert "search" in names

    def test_execute_valid_tool(self, registry):
        """测试执行已注册的工具"""
        result = registry.execute("calculator", {"expression": "1 + 1"})
        assert "2" in result

    def test_execute_unknown_tool(self, registry):
        """测试执行未注册的工具"""
        result = registry.execute("nonexistent", {})
        assert "错误" in result
        assert "未找到" in result

    def test_duplicate_registration(self):
        """测试重复注册同一工具应报错

        学习点:
          - 这是防御性编程：防止意外覆盖已注册的工具
          - 如果不报错，第二个注册会静默覆盖第一个，导致难以调试的 bug
        """
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="已注册"):
            registry.register(CalculatorTool())

    def test_schema_format(self, registry):
        """测试 schema 格式符合 OpenAI function calling 规范"""
        schemas = registry.get_schemas()
        for schema in schemas:
            assert schema["type"] == "function"
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"