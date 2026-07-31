import json
from .tools import TOOL_SCHEMAS

SOC_SYSTEM_PROMPT = """You are an expert Cybersecurity Operations Center (SOC) analyst \
specializing in third-party vendor risk and supply-chain attack detection.

You have access to tools to pull additional context before making a final call. \
On each turn, respond with ONLY a JSON object in one of these two forms:

1. To call a tool:
{"action": "tool_call", "tool": "<tool_name>", "args": {...}}

2. To give your final answer (only after you have enough context):
{"action": "final_answer", "summary": "...", "recommended_actions": ["...", "..."], "priority": "Routine|Elevated|Critical"}

Available tools:
""" + json.dumps(TOOL_SCHEMAS, indent=2) + """

Rules:
- Do NOT invent facts. Base your analysis only on the provided event data and tool observations.
- Call at most 2-3 tools before finalizing — you are not required to call every tool.
- Low risk tier events should generally get "Routine" priority with minimal action.
- High risk tier + credential/session anomalies should lean "Critical".
- Output ONLY valid JSON. No markdown, no prose outside the JSON object.
"""

def build_event_prompt(row: dict) -> str:
    return f"""Analyze the following vendor security event:
* Vendor ID: {row.get('vendor_id')}
* Time Window: {row.get('window_start')}
* Risk Tier: {row.get('risk_tier')}
* Dynamic Risk Score: {row.get('dynamic_risk_score')}
* Specific Anomalies Detected: {row.get('risk_reasons')}
* New Resources Accessed: {row.get('new_resources_list') or 'None'}
* Shared Resource Count: {row.get('shared_resource_count')}

Decide if you need more context via a tool call, or finalize your assessment now."""

# JSON schema for the FINAL structured output (used with Ollama's format param
# once the agent decides to finalize, to guarantee clean parseable output)
FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["final_answer"]},
        "summary": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "string", "enum": ["Routine", "Elevated", "Critical"]},
    },
    "required": ["action", "summary", "recommended_actions", "priority"],
}