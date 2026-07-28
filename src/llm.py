import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()
logger = logging.getLogger(__name__)

class LLMClient:
    """
    统一的 LLM 调用接口。
    支持 OpenAI 和 Anthropic，自动根据 .env 中的 LLM_PROVIDER 选择。
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai")

        if self.provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),  # 自定义端点
            )
            self.model = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
        elif self.provider == "anthropic":
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url=os.getenv("ANTHROPIC_BASE_URL"),  # 自定义端点
            )
            self.model = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-flash")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """
        发送消息给 LLM，返回统一格式的响应。

        统一返回格式：
        {
            "type": "text" | "tool_call",
            "content": "..." | None,
            "tool_calls": [{"name": "...", "arguments": {...}}] | None
        }
        """
        if self.provider == "openai":
            return self._chat_openai(messages, tools)
        else:
            return self._chat_anthropic(messages, tools)

    def _chat_openai(self, messages, tools):
        """
        OpenAI 风格调用（兼容 DeepSeek、GPT 等）。
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tools:  # 只在有工具时才传 tools 参数，避免 None 导致的意外行为
                kwargs["tools"] = tools

            # DEBUG: 打印实际发送的消息和工具
            logger.debug(f"发送消息: {[m['role'] for m in messages]}")
            logger.debug(f"用户输入: {messages[-1]['content'] if messages else 'N/A'}")
            logger.debug(f"工具数量: {len(tools) if tools else 0}")

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # 检查是否有工具调用
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        # LLM 偶尔返回不合法的 JSON，兜底用空 dict
                        args = {}
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_calls": tool_calls,
                }
            else:
                return {
                    "type": "text",
                    "content": message.content,
                    "tool_calls": None,
                }
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def _convert_tools_to_anthropic(self, tools: list[dict]) -> list[dict]:
        """
        将 OpenAI 格式的工具定义转换为 Anthropic 格式。

        OpenAI 格式:
          {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Anthropic 格式:
          {"name": ..., "description": ..., "input_schema": ...}
        """
        converted = []
        for tool in tools:
            func = tool.get("function", tool)  # 兼容已转换好的格式
            converted.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted

    def _chat_anthropic(self, messages, tools):
        """
        Anthropic 风格调用（Claude / DeepSeek via Anthropic API）。
        """
        try:
            # 1. 提取系统提示（Anthropic 的 system 是独立参数）
            system_prompt = None
            filtered_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    filtered_messages.append(msg)

            # 2. 转换消息格式（Anthropic 只认 user/assistant，没有 system）
            converted = []
            for msg in filtered_messages:
                role = msg["role"]
                if role == "system":
                    continue  # 已提取，跳过
                content = msg["content"]
                converted.append({"role": role, "content": content})

            # 3. 构建请求参数，tools 为 None 时不传
            kwargs = {
                "model": self.model,
                "messages": converted,
                "system": system_prompt,
                "temperature": self.temperature,
            }
            if tools:
                kwargs["tools"] = self._convert_tools_to_anthropic(tools)

            # 4. 调用 API
            response = self.client.messages.create(**kwargs)

            # 5. 解析响应内容（content 是一个列表，可能包含 text 和 tool_use 块）
            text_parts = []
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "name": block.name,
                        "arguments": block.input,  # 已经是 dict
                    })

            # 6. 统一返回格式
            if tool_calls:
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_calls": tool_calls
                }
            else:
                return {
                    "type": "text",
                    "content": "\n".join(text_parts),
                    "tool_calls": None
                }
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")