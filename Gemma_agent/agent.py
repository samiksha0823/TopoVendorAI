"""
The ReAct-style multi-step agent loop: Gemma reasons -> picks a tool ->
observes result -> repeats -> finalizes with a structured JSON answer.
This is the 'agents and multi-step reasoning' + 'function calling' +
'structured outputs' integration point.
"""
from .client import OllamaClient
from .tools import call_tool
from .prompts import SOC_SYSTEM_PROMPT, build_event_prompt, FINAL_ANSWER_SCHEMA

MAX_STEPS = 3


class SOCAgent:
    def __init__(self, model: str = "gemma3:4b"):
        self.client = OllamaClient(model=model)

    def analyze_event(self, row: dict) -> dict:
        # Fast path: skip Gemma entirely for Low risk to save compute
        if str(row.get("risk_tier", "")).lower() == "low":
            return {
                "summary": "No significant anomalies detected; within normal vendor baseline.",
                "recommended_actions": ["Continue standard monitoring"],
                "priority": "Routine",
            }

        messages = [
            {"role": "system", "content": SOC_SYSTEM_PROMPT},
            {"role": "user", "content": build_event_prompt(row)},
        ]

        for step in range(MAX_STEPS):
            is_last_step = step == MAX_STEPS - 1
            raw = self.client.chat(
                messages,
                json_schema=FINAL_ANSWER_SCHEMA if is_last_step else None,
            )
            parsed = self.client.try_parse_json(raw)

            if not parsed:
                # Model didn't follow the format — force a final structured retry
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                  "content": "Respond with valid JSON only, per the schema."})
                continue

            if parsed.get("action") == "tool_call":
                tool_name = parsed.get("tool")
                args = parsed.get("args", {})
                observation = call_tool(tool_name, args)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                  "content": f"Tool result for {tool_name}: {observation}\n"
                                             f"Continue reasoning or finalize now."})
                continue

            if parsed.get("action") == "final_answer":
                return {
                    "summary": parsed.get("summary", ""),
                    "recommended_actions": parsed.get("recommended_actions", []),
                    "priority": parsed.get("priority", "Elevated"),
                }

        # Fallback if agent never finalized cleanly
        return {
            "summary": "Agent could not converge on a final answer within step limit.",
            "recommended_actions": ["Escalate to human SOC analyst for manual review"],
            "priority": "Elevated",
        }