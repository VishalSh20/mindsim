"""LLM Client — OpenAI / Ollama abstraction.

Uses OpenAI API for cloud models, local Ollama as fallback.
All calls return raw strings. Caller is responsible for JSON parsing.
"""

from __future__ import annotations

import json
import os
import re
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default models
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OPENAI_MODEL = "gpt-4.1"


def _detect_model() -> str:
    """Detect the best available model.

    Priority:
    1. MINDSIM_MODEL env var (explicit override)
    2. OpenAI API (if key available)
    3. Local Ollama (if running)
    """
    env_model = os.getenv("MINDSIM_MODEL", "auto")

    if env_model and env_model != "auto":
        return env_model

    if os.getenv("OPENAI_API_KEY"):
        return DEFAULT_OPENAI_MODEL

    if _ollama_available():
        return f"ollama/{DEFAULT_OLLAMA_MODEL}"

    return DEFAULT_OPENAI_MODEL


def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


class LLMClient:
    """Unified LLM client for all mindsim LLM calls."""

    def __init__(self, model: str | None = None):
        self.model = model or _detect_model()
        self._call_count = 0
        self._is_ollama = self.model.startswith("ollama/")
        logger.info(f"LLM Client initialized with model: {self.model}")

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.15,
        max_retries: int = 1,
    ) -> str:
        """Make a completion call and return raw string."""
        for attempt in range(1 + max_retries):
            try:
                if self._is_ollama:
                    content = self._call_ollama(system_prompt, user_message, temperature)
                else:
                    content = self._call_openai(system_prompt, user_message, temperature)
                self._call_count += 1
                return content.strip() if content else ""
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                    continue
                raise

    def _call_openai(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
    ) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def _call_ollama(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
    ) -> str:
        """Call local Ollama."""
        import urllib.request
        import json as _json

        model_name = self.model.removeprefix("ollama/")
        payload = _json.dumps({
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 4096},
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
            return data.get("message", {}).get("content", "")

    def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.15,
    ) -> dict:
        """Make a completion call and parse response as JSON.

        Handles markdown code fences and retries on parse failure.
        """
        raw = self.complete(system_prompt, user_message, temperature)

        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed

        logger.warning("JSON parse failed, retrying with correction prompt")
        fix_prompt = (
            f"Your previous response was not valid JSON. "
            f"Here is what you returned:\n\n{raw}\n\n"
            f"Please return ONLY valid JSON, no markdown fences, no explanation."
        )
        raw2 = self.complete(system_prompt, fix_prompt, temperature)
        parsed2 = _extract_json(raw2)
        if parsed2 is not None:
            return parsed2

        raise ValueError(f"Failed to parse LLM response as JSON after retry:\n{raw2}")

    @property
    def call_count(self) -> int:
        return self._call_count


def _extract_json(text: str) -> dict | None:
    """Extract JSON from text that may contain markdown code fences."""
    text = text.strip()

    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        text = matches[0].strip()

    think_pattern = r"<think>.*?</think>"
    text = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None
