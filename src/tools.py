import os
import ast
import operator
import json
from typing import Dict, Any, List

class BaseTool:
    """工具基类"""
    @property
    def schema(self) -> dict:
        raise NotImplementedError

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    """安全计算器，支持 + - * / // % ** 和括号"""
    
    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "执行安全的数学计算，支持 + - * / // % ** 和括号，不支持函数和变量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要计算的数学表达式，例如 '2 + 3 * (4 - 1)'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

    def execute(self, expression: str) -> str:
        # 1. 允许的操作符和节点类型
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,   # 一元负号
        }
        # 只列出真正的 AST 节点类型，operator（Add/Sub 等）通过 allowed_operators 检查
        allowed_nodes = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp)

        try:
            tree = ast.parse(expression.strip(), mode='eval')
        except SyntaxError:
            return "错误：表达式语法不正确"

        # 2. 安全遍历：只允许特定节点
        def _check(node):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"不支持的语法元素: {type(node).__name__}")
            if isinstance(node, ast.BinOp):
                _check(node.left)
                _check(node.right)
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            elif isinstance(node, ast.UnaryOp):
                _check(node.operand)
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            elif isinstance(node, ast.Constant):
                # 只允许数值常量
                if not isinstance(node.value, (int, float)):
                    raise ValueError("只允许数字常量")
            elif isinstance(node, ast.Expression):
                _check(node.body)

        try:
            _check(tree)
        except ValueError as e:
            return f"错误：{e}"

        # 3. 执行计算（用访问者模式或直接eval，但我们已经确保安全）
        # 简单起见，使用eval但限制全局/局部命名空间为空，但eval本身可能不安全，最好自己实现求值。
        # 这里使用一个简单的递归求值函数
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_func = allowed_operators[type(node.op)]
                return op_func(left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_func = allowed_operators[type(node.op)]
                return op_func(operand)
            else:
                raise ValueError(f"未知节点类型: {type(node).__name__}")

        try:
            result = _eval(tree.body)
            # 避免浮点精度过长
            if isinstance(result, float):
                result = round(result, 10)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算时出错: {e}"


class SearchTool(BaseTool):
    """模拟搜索工具（mock），实际可替换为真实API"""
    
    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "模拟搜索引擎，返回预设的测试结果。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, query: str) -> str:
        # Mock数据，可替换为真实搜索API
        mock_results = {
            "北京天气": "北京今天晴朗，25°C，南风3级。",
            "Python": "Python是一种高级编程语言，广泛应用于数据分析、Web开发等。",
            "Agent": "Agent是一种能够感知环境并采取行动以实现目标的智能体。",
        }
        # 简单匹配
        for key, value in mock_results.items():
            if key in query:
                return f"搜索结果: {value}"
        return f"未找到关于 '{query}' 的模拟结果，请尝试其他关键词。"


class ReadFileTool(BaseTool):
    """安全读取文件（限制项目目录）"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.getcwd())  # 默认项目根目录
        else:
            self.base_dir = os.path.abspath(base_dir)

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取指定文件的内容，只能访问项目目录内的文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "文件路径（相对或绝对路径，但必须位于项目目录内）"
                        }
                    },
                    "required": ["filepath"]
                }
            }
        }

    def execute(self, filepath: str) -> str:
        try:
            # 解析绝对路径
            abs_path = os.path.abspath(filepath)
            # 安全检查：确保路径在 base_dir 下
            if not abs_path.startswith(self.base_dir):
                return f"错误：文件路径 '{filepath}' 不在允许的项目目录内。"
            if not os.path.exists(abs_path):
                return f"错误：文件 '{filepath}' 不存在。"
            if os.path.isdir(abs_path):
                return f"错误：'{filepath}' 是一个目录，请指定文件。"
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 限制返回长度，防止过大
            max_len = 5000
            if len(content) > max_len:
                content = content[:max_len] + "\n...(截断)"
            return f"文件内容:\n{content}"
        except Exception as e:
            return f"读取文件失败: {e}"


class WriteFileTool(BaseTool):
    """安全写入文件（限制项目目录）"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.getcwd())
        else:
            self.base_dir = os.path.abspath(base_dir)

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "将内容写入文件，只能写入项目目录内的文件（不会覆盖已存在文件）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "目标文件路径"
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入的内容"
                        }
                    },
                    "required": ["filepath", "content"]
                }
            }
        }

    def execute(self, filepath: str, content: str) -> str:
        try:
            abs_path = os.path.abspath(filepath)
            if not abs_path.startswith(self.base_dir):
                return f"错误：文件路径 '{filepath}' 不在允许的项目目录内。"
            # 检查是否已存在，如果存在且不是目录，则拒绝覆盖（安全策略）
            if os.path.exists(abs_path):
                return f"错误：文件 '{filepath}' 已存在，拒绝覆盖。"
            # 确保目录存在
            dirname = os.path.dirname(abs_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"成功写入文件: {filepath}"
        except Exception as e:
            return f"写入文件失败: {e}"


class ToolRegistry:
    """工具注册中心"""
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """注册工具，使用schema中的name作为键"""
        name = tool.schema["function"]["name"]
        if name in self._tools:
            raise ValueError(f"工具 '{name}' 已注册")
        self._tools[name] = tool

    def get_schemas(self) -> List[dict]:
        """返回所有工具的OpenAI格式schema列表"""
        return [tool.schema for tool in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """执行工具，返回字符串结果"""
        if name not in self._tools:
            return f"错误：未找到工具 '{name}'"
        tool = self._tools[name]
        try:
            # 调用execute，传入**arguments
            result = tool.execute(**arguments)
            # 确保返回字符串
            if not isinstance(result, str):
                result = str(result)
            return result
        except Exception as e:
            return f"执行工具 '{name}' 时出错: {e}"