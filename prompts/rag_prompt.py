"""
RAG generation prompt templates and formatting utilities.
Constructs system and user prompts for context-grounded response generation.
"""

from typing import Any, Dict, List

EMPATHY_PREFIX: str = (
    "I'm truly sorry to hear about your experience. "
    "I understand your frustration, and I want to help resolve this as quickly as possible. "
)

_RAW_RAG_SYSTEM_PROMPT = """You are a helpful, professional customer support assistant for an online retailer.
Answer the customer's question using ONLY the information in the retrieved support responses below.

Emotional Tone & Tone Adjustment:
- Customer detected sentiment: {detected_sentiment}
- If the customer sounds frustrated, angry, or disappointed (sentiment: negative), acknowledge their frustration with genuine empathy and apologize before answering.
- If the customer has a neutral tone (sentiment: neutral), be clear, direct, polite, and helpful.
- If the customer sounds happy or satisfied (sentiment: positive), match their warm, positive tone.
- If the retrieved context does not cover the question, say so honestly and offer to escalate to a human agent rather than guessing.
Keep your response concise, friendly, grounded, and professional.

Retrieved Support Responses:
{retrieved_context}"""


RAG_SYSTEM_PROMPT: str = _RAW_RAG_SYSTEM_PROMPT


def _extract_chunk_text(chunk: Any) -> str:
    """Extract readable text from a chunk item (dict or string).

    Handles multiple formats:
      - Retriever output: {'document': ..., 'response': ..., 'metadata': {'response': ...}}
      - Raw dict with 'response', 'text', etc.
      - Plain string
    """
    if isinstance(chunk, dict):
        # First check for metadata.response (retriever output format)
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            meta_response = metadata.get("response", "")
            if meta_response and str(meta_response).strip():
                return str(meta_response).strip()

        # Then check top-level keys
        for key in ("response", "text", "content", "answer", "document", "chunk"):
            val = chunk.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()

        # Handle instruction/response pair if both present
        instruction = chunk.get("instruction", "")
        response = chunk.get("response", "")
        if instruction or response:
            parts = [p for p in (str(instruction).strip(), str(response).strip()) if p]
            return ": ".join(parts)

        # Fallback to values
        vals = [str(v).strip() for v in chunk.values() if v is not None and str(v).strip()]
        if vals:
            return " ".join(vals)

    return str(chunk).strip()


def format_rag_prompt(
    user_message: str,
    detected_sentiment: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Format the RAG prompt into system and user messages ready for an LLM API call.

    Parameters
    ----------
    user_message : str
        The customer's input query or message.
    detected_sentiment : str
        The detected sentiment of the customer (e.g. 'negative', 'neutral', 'positive').
    retrieved_chunks : List[Dict[str, Any]]
        List of retrieved knowledge base / support response chunk dictionaries.
        Each chunk is formatted into a numbered list item.

    Returns
    -------
    Dict[str, str]
        Dictionary with 'system' and 'user' keys ready for the chat API call.
    """
    formatted_items: List[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        chunk_text = _extract_chunk_text(chunk)
        formatted_items.append(f"{idx}. {chunk_text}")

    if formatted_items:
        retrieved_context = "\n".join(formatted_items)
    else:
        retrieved_context = "No relevant support responses found."

    system_content = RAG_SYSTEM_PROMPT.format(
        detected_sentiment=detected_sentiment,
        retrieved_context=retrieved_context,
    )

    return {
        "system": system_content,
        "user": user_message,
    }
