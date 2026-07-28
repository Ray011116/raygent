"""
评测基线测试。

与 test_agent.py 的区别:
  - test_agent 用 Mock LLM，测试代码逻辑（确定性）
  - test_eval 用真实 LLM，测试模型能力（非确定性）

测试策略:
  - 每个测试用例包含：问题 + 预期关键词列表
  - 使用模糊匹配：Agent 的回答中包含所有预期关键词即算通过
  - 全部标记为 @pytest.mark.slow，默认不运行

运行方式:
  pytest tests/test_eval.py -v -m slow          # 只跑 eval
  pytest tests/test_eval.py -v -m "not slow"    # 跳过 eval（CI 默认行为）

学习点:
  - 为什么用模糊匹配而非精确匹配？
    → LLM 的回答是自由文本，不会每次都一模一样
    → "1+1等于2" 和 "答案是2" 都是正确答案
    → 关键词匹配比精确匹配更适合 LLM 评测
  - 为什么标记为 slow？
    → 真实 LLM 调用需要网络请求，延迟高（秒级）
    → CI 中每次提交都跑 eval 太慢、太贵
    → 应该在 PR 合并前或定时任务中跑
"""

import os
import pytest
from src.agent import ReActAgent
from src.llm import LLMClient
from src.tools import CalculatorTool, SearchTool, ToolRegistry


# ══════════════════════════════════════════════════════════════════════════════
# 评测配置
# ══════════════════════════════════════════════════════════════════════════════

# 跳过条件：如果环境变量 SKIP_EVAL 被设置，跳过所有 eval 测试
# 用法: SKIP_EVAL=1 pytest tests/test_eval.py -v
SKIP_EVAL = bool(os.getenv("SKIP_EVAL"))

# 标记所有 eval 测试为 slow，且可通过 SKIP_EVAL 跳过
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(SKIP_EVAL, reason="SKIP_EVAL 环境变量已设置"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 评测用例定义
# ══════════════════════════════════════════════════════════════════════════════

# 每个用例: (问题, 预期关键词列表)
# 预期关键词：Agent 的回答中必须包含所有这些词才算通过

MATH_CASES = [
    (
        "计算 1 + 2 * 3",
        ["7"],  # 1 + 2*3 = 7，不是 9
    ),
    (
        "25 乘以 4 等于多少",
        ["100"],
    ),
    (
        "100 除以 3 等于多少",
        ["33.3333333333"],  # 精确到 10 位小数
    ),
    (
        "2 的 10 次方是多少",
        ["1024"],
    ),
    (
        "计算 (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) 除以 5",
        ["11"],  # (55) / 5 = 11
    ),
]

SEARCH_CASES = [
    (
        "北京今天天气怎么样",
        ["25°C", "晴朗"],  # 来自 mock 数据
    ),
    (
        "Python 是什么",
        ["编程语言"],
    ),
    (
        "什么是 Agent",
        ["智能体"],
    ),
]

SIMPLE_QA_CASES = [
    (
        "你好",
        [],  # 不检查具体内容，只要不崩溃就算通过
    ),
    (
        "请用中文回答：太阳从哪边升起",
        ["东"],
    ),
]

# 合并所有用例
ALL_CASES = MATH_CASES + SEARCH_CASES + SIMPLE_QA_CASES


# ══════════════════════════════════════════════════════════════════════════════
# Fixture
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def agent():
    """创建 Agent 实例（module 级别，所有测试共享）。

    学习点:
      - scope="module" 表示整个测试模块只创建一次 Agent
      - 避免每个测试用例都重新初始化 LLM 客户端
      - 但注意：Agent 的 run() 方法每次都会重置消息历史
    """
    llm = LLMClient()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    return ReActAgent(llm_client=llm, tool_registry=registry, max_iterations=10)


# ══════════════════════════════════════════════════════════════════════════════
# 评测执行
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_case(agent, question: str, expected_keywords: list[str]) -> tuple[bool, str]:
    """执行单个评测用例，返回 (通过/失败, 实际回答)。

    学习点:
      - 为什么单独抽一个函数？
        → 方便复用，也方便统计 pass/fail
        → 与 pytest 参数化结合，每个用例成为一个独立测试
    """
    try:
        answer = agent.run(question)
    except Exception as e:
        return False, f"异常: {e}"

    # 模糊匹配：所有预期关键词都必须出现在回答中
    for keyword in expected_keywords:
        if keyword not in answer:
            return False, answer

    return True, answer


# ══════════════════════════════════════════════════════════════════════════════
# 参数化测试
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("question,expected_keywords", ALL_CASES)
def test_eval_case(agent, question, expected_keywords):
    """评测单个用例。

    学习点:
      - @pytest.mark.parametrize 自动为每个用例生成一个独立测试
      - 这样每个用例有独立的 PASS/FAIL 状态，方便定位问题
      - 而不是所有用例放在一个测试里，一个失败就全失败
    """
    passed, answer = evaluate_case(agent, question, expected_keywords)

    if not passed:
        # 构建清晰的失败信息
        missing = [kw for kw in expected_keywords if kw not in answer]
        pytest.fail(
            f"\n问题: {question}"
            f"\n预期关键词: {expected_keywords}"
            f"\n缺失关键词: {missing}"
            f"\n实际回答: {answer}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 分组统计测试（可选：按类别跑）
# ══════════════════════════════════════════════════════════════════════════════

class TestEvalByCategory:
    """按类别组织的评测，方便单独跑某一类。

    运行方式:
      pytest tests/test_eval.py::TestEvalByCategory::test_math -v -m slow
      pytest tests/test_eval.py::TestEvalByCategory::test_search -v -m slow
      pytest tests/test_eval.py::TestEvalByCategory::test_simple_qa -v -m slow
    """

    @pytest.fixture(scope="class")
    def agent(self):
        llm = LLMClient()
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool())
        return ReActAgent(llm_client=llm, tool_registry=registry, max_iterations=10)

    @pytest.mark.parametrize("question,expected_keywords", MATH_CASES)
    def test_math(self, agent, question, expected_keywords):
        """数学计算评测"""
        passed, answer = evaluate_case(agent, question, expected_keywords)
        if not passed:
            missing = [kw for kw in expected_keywords if kw not in answer]
            pytest.fail(
                f"\n问题: {question}"
                f"\n缺失关键词: {missing}"
                f"\n实际回答: {answer}"
            )

    @pytest.mark.parametrize("question,expected_keywords", SEARCH_CASES)
    def test_search(self, agent, question, expected_keywords):
        """搜索查询评测"""
        passed, answer = evaluate_case(agent, question, expected_keywords)
        if not passed:
            missing = [kw for kw in expected_keywords if kw not in answer]
            pytest.fail(
                f"\n问题: {question}"
                f"\n缺失关键词: {missing}"
                f"\n实际回答: {answer}"
            )

    @pytest.mark.parametrize("question,expected_keywords", SIMPLE_QA_CASES)
    def test_simple_qa(self, agent, question, expected_keywords):
        """简单问答评测"""
        passed, answer = evaluate_case(agent, question, expected_keywords)
        if not passed:
            missing = [kw for kw in expected_keywords if kw not in answer]
            pytest.fail(
                f"\n问题: {question}"
                f"\n缺失关键词: {missing}"
                f"\n实际回答: {answer}"
            )