"""
Canned responses for greeting, goodbye, and gratitude interactions.
Provides predetermined, professional responses for conversational chit-chat
to avoid unnecessary LLM calls.
"""

import random
from typing import Dict, List

# Canned responses mapped by sub-type
GREETING_RESPONSES: Dict[str, List[str]] = {
    "greeting": [
        "Hello! How can I assist you with your order or account today?",
        "Hi there! Welcome to customer support. What can I help you with?",
        "Hello! Thanks for reaching out. How may I assist you today?",
        "Hi! I'm here to help. What questions or concerns do you have today?",
    ],
    "goodbye": [
        "Thank you for contacting us. Have a wonderful day!",
        "Goodbye! Please feel free to reach out again if you need any further assistance.",
        "Have a great day ahead! Don't hesitate to reach out if you need anything else.",
        "Thank you for chatting with us today. Take care and goodbye!",
    ],
    "gratitude": [
        "You're very welcome! I'm glad I could help.",
        "Happy to help! Please let me know if there is anything else you need.",
        "It's my pleasure! Don't hesitate to reach out if you need further support.",
        "Anytime! Have a fantastic day ahead!",
    ],
}


def get_greeting_response(sub_type: str) -> str:
    """
    Randomly select and return a canned response for the given greeting sub-type.

    Parameters
    ----------
    sub_type : str
        The sub-type of greeting ('greeting', 'goodbye', 'gratitude', etc.).
        If the sub-type is not recognized or found, defaults to 'greeting'.

    Returns
    -------
    str
        A randomly chosen canned response string.
    """
    if not isinstance(sub_type, str):
        normalized_type = "greeting"
    else:
        normalized_type = sub_type.lower().strip()

    # Map common variations to canonical categories
    synonym_map = {
        "thank you": "gratitude",
        "thank_you": "gratitude",
        "thanks": "gratitude",
        "bye": "goodbye",
        "farewell": "goodbye",
        "hello": "greeting",
        "hi": "greeting",
        "hey": "greeting",
    }
    canonical_type = synonym_map.get(normalized_type, normalized_type)

    responses = GREETING_RESPONSES.get(canonical_type, GREETING_RESPONSES["greeting"])
    return random.choice(responses)
