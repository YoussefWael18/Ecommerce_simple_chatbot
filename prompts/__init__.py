"""
Prompts package for e-commerce customer support chatbot.
Exposes intent classification prompts, RAG generation prompts, and canned greeting responses.
"""

from .intent_prompt import INTENT_SYSTEM_PROMPT, INTENT_CATEGORIES, INTENT_CATEGORY_DESCRIPTIONS
from .rag_prompt import RAG_SYSTEM_PROMPT, EMPATHY_PREFIX, format_rag_prompt
from .greeting_responses import GREETING_RESPONSES, get_greeting_response

__all__ = [
    "INTENT_SYSTEM_PROMPT",
    "INTENT_CATEGORIES",
    "INTENT_CATEGORY_DESCRIPTIONS",
    "RAG_SYSTEM_PROMPT",
    "EMPATHY_PREFIX",
    "format_rag_prompt",
    "GREETING_RESPONSES",
    "get_greeting_response",
]
