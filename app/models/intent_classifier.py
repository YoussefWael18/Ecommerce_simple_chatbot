"""
Few-shot intent classifier powered by OpenRouter LLM API.
Classifies user queries into e-commerce customer support intent categories.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path for relative/absolute prompt imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from prompts.intent_prompt import INTENT_SYSTEM_PROMPT
except ImportError:
    try:
        from ..prompts.intent_prompt import INTENT_SYSTEM_PROMPT
    except ImportError:
        # Re-raise or fallback if prompts is structured differently
        raise

from openai import OpenAI

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Zero-shot and few-shot intent classifier using an LLM via OpenRouter.
    Identifies customer inquiry categories (e.g., greeting, order_status, complaint).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/llama-3.3-70b-instruct:free",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize the IntentClassifier.

        Args:
            api_key: OpenRouter API key. If not provided, loaded from config or environment.
            model_name: Model identifier on OpenRouter.
            base_url: OpenRouter API base endpoint URL.
        """
        if not api_key or not model_name:
            try:
                from app.config import OPENROUTER_API_KEY, LLM_MODEL_NAME
                if not api_key:
                    api_key = OPENROUTER_API_KEY
                if not model_name or model_name == "meta-llama/llama-3.3-70b-instruct:free":
                    model_name = LLM_MODEL_NAME
            except Exception:
                if not api_key:
                    api_key = os.getenv("openrouter") or os.getenv("OPENROUTER_API_KEY", "")
                if not model_name:
                    model_name = "minimax/minimax-m3:free"

        self.api_key = api_key or ""
        self.model_name = model_name or "minimax/minimax-m3:free"
        self.base_url = base_url
        self.system_prompt = INTENT_SYSTEM_PROMPT

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "EMPTY_API_KEY",
        )

    def predict(self, text: str) -> Dict[str, str]:
        """
        Classify customer message into an intent group and confidence level.

        Args:
            text: Customer inquiry message.

        Returns:
            dict: {'intent_group': str, 'confidence': str}
        """
        fallback_result = {"intent_group": "out_of_scope", "sentiment": "neutral", "confidence": "low"}

        if not text or not str(text).strip():
            return fallback_result

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": str(text).strip()},
                ],
                temperature=0.1,
                max_tokens=100,
            )

            if not response or not response.choices:
                return fallback_result

            content = response.choices[0].message.content
            if not content:
                return fallback_result

            parsed = self._parse_json_response(content)
            intent_group = parsed.get("intent_group")
            sentiment = parsed.get("sentiment", "neutral")
            confidence = parsed.get("confidence", "low")

            if not intent_group:
                return fallback_result

            clean_sentiment = str(sentiment).strip().lower()
            if clean_sentiment not in ("negative", "neutral", "positive"):
                clean_sentiment = "neutral"

            return {
                "intent_group": str(intent_group).strip(),
                "sentiment": clean_sentiment,
                "confidence": str(confidence).strip(),
            }

        except Exception as exc:
            logger.warning("Intent classification failed with error: %s", exc)
            return fallback_result

    @staticmethod
    def _parse_json_response(content: str) -> Dict[str, Any]:
        """
        Safely extract and parse JSON from the LLM text output.

        Handles:
          - Pure JSON strings: {"intent_group": "...", "confidence": "..."}
          - Markdown fenced blocks: ```json ... ```
          - Surrounding text with embedded JSON { ... }
        """
        cleaned = content.strip()

        # 1. Direct JSON parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Markdown fenced block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. First outer brace pair
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse valid JSON from LLM response: {content}")

