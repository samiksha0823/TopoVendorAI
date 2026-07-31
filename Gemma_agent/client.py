"""
Thin wrapper around a local Ollama instance running Gemma 3.
Handles plain generation, chat, and JSON-schema-constrained structured output.
"""
import requests
import json

OLLAMA_URL = "http://localhost:11434"


class OllamaClient:
    def __init__(self, model: str = "gemma3:4b", timeout: int = 120):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system: str = None, json_schema: dict = None,
                 temperature: float = 0.2) -> str:
        """
        Single-shot generation. If json_schema is provided, Ollama will
        constrain output to match it (structured output requirement).
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if json_schema:
            payload["format"] = json_schema

        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "")

    def chat(self, messages: list, json_schema: dict = None, temperature: float = 0.2) -> str:
        """
        Multi-turn chat — used by the agent loop to keep tool call/observation history.
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema:
            payload["format"] = json_schema

        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def try_parse_json(self, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Gemma sometimes wraps JSON in ```json fences despite instructions
            cleaned = text.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None