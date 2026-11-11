"""Tool-calling loop on top of any OpenAI-compatible API (I use Groq + Llama 3.3 70B)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, FinOpsTools

SYSTEM_PROMPT = """You are CloudBuddy, a FinOps assistant for an AWS EC2 fleet.
Rules:
- Every number you mention must come from a tool result. Don't make up instance ids, prices or savings.
- Use recommend_actions for fleet-wide questions and forecast_usage for a single instance.
- You can't change infrastructure. If the user wants a change, call request_action and tell them
  it is waiting for approval (not done).
- Keep it short: one summary line, then a compact list."""


class GroqChat:
    def __init__(self, model=None, base_url=None, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                             api_key=api_key or os.getenv("LLM_API_KEY"))
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def chat(self, messages, tools):
        resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                   tool_choice="auto", temperature=0.1)
        return resp.choices[0].message.model_dump(exclude_none=True)


@dataclass
class AgentReply:
    answer: str
    steps: list = field(default_factory=list)


class FinOpsAgent:
    def __init__(self, llm, tools: FinOpsTools, max_steps=8):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, question: str) -> AgentReply:
        self.messages.append({"role": "user", "content": question})
        steps = []
        for _ in range(self.max_steps):
            msg = self.llm.chat(self.messages, TOOL_SCHEMAS)
            calls = msg.get("tool_calls") or []
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            self.messages.append(assistant)
            if not calls:
                return AgentReply(assistant["content"], steps)
            for c in calls:
                name = c["function"]["name"]
                args = c["function"].get("arguments") or "{}"
                result = self.tools.call(name, args)
                steps.append({"tool": name, "args": args, "result": result})
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        return AgentReply("Too many tool calls, try a more specific question.", steps)
