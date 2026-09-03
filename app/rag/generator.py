"""
Generator module for generating grounded LLM responses via OpenRouter.
"""

from __future__ import annotations

import logging
from typing import Any
from openai import OpenAI

from prompts.rag_prompt import format_rag_prompt

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I apologize, but I'm having trouble processing your request right now. "
    "Please try again or contact our support team directly."
)


class Generator:
    """
    Generates customer support responses using an OpenRouter LLM grounded on retrieved context.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        """
        Initialize the Generator.

        Args:
            api_key: OpenRouter API key.
            model_name: Model identifier on OpenRouter.
            base_url: OpenRouter API base endpoint URL.
        """
        if not api_key or not model_name:
            try:
                from app.config import OPENROUTER_API_KEY, LLM_MODEL_NAME
                if not api_key:
                    api_key = OPENROUTER_API_KEY
                if not model_name:
                    model_name = LLM_MODEL_NAME
            except Exception:
                import os
                if not api_key:
                    api_key = os.getenv("openrouter") or os.getenv("OPENROUTER_API_KEY", "")
                if not model_name:
                    model_name = "minimax/minimax-m3:free"

        self.api_key = api_key or ""
        self.model_name = model_name or "minimax/minimax-m3:free"
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key or "EMPTY_KEY",
        )

    def generate(
        self,
        user_message: str,
        detected_sentiment: str,
        retrieved_chunks: list[dict[str, Any]],
        detected_language: str = "en",
    ) -> str:
        """
        Generate a grounded customer support response.

        Args:
            user_message: The customer's message.
            detected_sentiment: Sentiment classification ('negative', 'neutral', 'positive').
            retrieved_chunks: Context chunks retrieved from vector store.
            detected_language: Detected language code.

        Returns:
            The generated response string, or a fallback message upon failure.
        """
        try:
            prompt_data = format_rag_prompt(
                user_message=user_message,
                detected_sentiment=detected_sentiment,
                retrieved_chunks=retrieved_chunks,
            )

            system_content = prompt_data.get("system", "")
            user_content = prompt_data.get("user", user_message)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            if response and response.choices:
                content = response.choices[0].message.content
                if content and str(content).strip():
                    return str(content).strip()

            return FALLBACK_MESSAGE

        except Exception as e:
            logger.error("Error generating response with LLM: %s", e, exc_info=True)
            return FALLBACK_MESSAGE
