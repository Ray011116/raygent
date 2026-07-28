import uuid
import logging
from typing import Dict, Any
import json

# 配置日志（便于调试）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActAgent:
    def __init__(self, llm_client, tool_registry, max_iterations=10):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.messages: list[dict] = []

    def _build_system_prompt(self) -> str:
        """构造 System Prompt，引导 LLM 按照 ReAct 模式思考，并限制每次只调用一个工具"""
        return (
            "你是一个智能助手，可以使用工具来完成任务。\n\n"
            "对于每个任务，请按以下格式思考和行动：\n"
            "1. 思考（Thought）：分析当前状态，决定下一步\n"
            "2. 行动（Action）：如果需要信息，调用合适的工具（一次只调用一个）\n"
            "3. 观察（Observation）：分析工具返回的结果（该步骤由系统自动提供）\n"
            "4. 重复上述步骤，直到可以给出最终答案\n\n"
            "重要规则：\n"
            "- 涉及数学计算时，必须使用 calculator 工具，绝对不要自己心算。\n"
            "- 需要搜索信息时，必须使用 search 工具，不要凭空编造。\n"
            "- 如果可以直接回答（且不涉及计算或搜索），不要调用工具，直接输出最终答案。\n"
            "- 每次只能调用一个工具，且工具调用后必须分析结果再决定下一步。\n"
            "- 如果工具调用失败，尝试其他方法或直接告知无法完成。\n"
            "- 最终答案必须基于工具返回的事实，且清晰、简洁。\n"
            "- 输出最终答案时，不需要额外解释过程，直接给出答案即可。"
        )

    def run(self, user_input: str) -> str:
        """
        执行 ReAct 循环，返回最终答案。
        """
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input}
        ]

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"=== 迭代 {iteration}/{self.max_iterations} ===")

            response = self.llm.chat(self.messages, self.tools.get_schemas())

            if response["type"] == "text":
                final_answer = response["content"]
                logger.info(f"最终结果: {final_answer}")
                return final_answer

            elif response["type"] == "tool_call":
                tool_calls = response["tool_calls"]

                # 1. 为每个工具调用生成唯一 ID
                tool_call_infos = []
                for tc in tool_calls:
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_call_infos.append({
                        "id": call_id,
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    })

                # 2. 构建 assistant 消息（包含所有 tool_calls）
                assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": info["id"],
                            "type": "function",
                            "function": {
                                "name": info["name"],
                                "arguments": json.dumps(info["arguments"])
                            }
                        }
                        for info in tool_call_infos
                    ]
                }
                self.messages.append(assistant_msg)

                # 3. 执行每个工具，并用对应的 ID 追加结果
                for info in tool_call_infos:
                    logger.info(f"调用工具: {info['name']}({info['arguments']})")
                    result = self.tools.execute(info["name"], info["arguments"])
                    logger.info(f"工具结果: {result[:100]}..." if len(result) > 100 else f"工具结果: {result}")

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": info["id"],  # 关键：每个结果对应自己的 ID
                        "content": result
                    }
                    self.messages.append(tool_msg)

                # 继续循环，让 LLM 基于工具结果做下一步决策
                continue

            else:
                error_msg = f"未知响应类型: {response.get('type')}"
                logger.error(error_msg)
                return f"错误: {error_msg}"

        logger.warning(f"达到最大迭代次数 {self.max_iterations}，强制结束")
        return f"未能完成任务，已达到最大尝试次数 {self.max_iterations}。"

