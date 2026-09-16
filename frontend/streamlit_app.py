"""Customer-facing Streamlit chat for the existing FastAPI support pipeline."""

import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")

st.set_page_config(
    page_title="Nova | Customer Support",
    page_icon="N",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f8fafc; color: #162238; }
      .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 7rem; }
      [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5eaf0; }
      [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
      [data-testid="stChatMessage"] {
        background: #ffffff; border: 1px solid #e5eaf0; border-radius: 16px;
        padding: 1.1rem 1.2rem; margin-bottom: 0.85rem;
      }
      [data-testid="stChatInput"] { border-radius: 14px; }
      .support-brand { font-size: 1.35rem; font-weight: 750; letter-spacing: -.03em; }
      .support-mark {
        display: inline-flex; width: 34px; height: 34px; align-items: center;
        justify-content: center; border-radius: 10px; background: #1644a8;
        color: #fff; font-size: 1rem; margin-right: .45rem;
      }
      .support-eyebrow { color: #466890; font-weight: 700; font-size: .82rem;
        letter-spacing: .11em; text-transform: uppercase; margin-bottom: .3rem; }
      .support-title { font-size: 2.2rem; font-weight: 760; letter-spacing: -.04em;
        line-height: 1.12; margin-bottom: .35rem; }
      .support-subtitle { color: #637187; font-size: 1rem; margin-bottom: 1.7rem; }
      .support-note { color: #7c899a; font-size: .79rem; line-height: 1.45; }
      #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def ask_backend(message: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/chat", json={"message": message}, timeout=90
    )
    response.raise_for_status()
    return response.json()


def new_conversation() -> list[dict]:
    return [{
        "role": "assistant",
        "content": "Hello! I can help with orders, delivery, returns, payments, and account questions. What would you like to know?",
        "time": datetime.now().strftime("%I:%M %p"),
        "sources": [],
        "diagnostics": {},
    }]


if "messages" not in st.session_state:
    st.session_state.messages = new_conversation()

with st.sidebar:
    st.markdown(
        '<div class="support-brand"><span class="support-mark">N</span> Nova Support</div>',
        unsafe_allow_html=True,
    )
    st.caption("Customer care")
    st.divider()
    if st.button("New conversation", width="stretch", type="primary"):
        st.session_state.messages = new_conversation()
        st.rerun()
    st.write("")
    show_diagnostics = st.toggle("Show diagnostics", value=False)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="support-note">Answers in this project use sample policy content. '
        'Check actual retailer terms before acting.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="support-eyebrow">Customer care</div>', unsafe_allow_html=True)
st.markdown('<div class="support-title">How can we help?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="support-subtitle">Ask a question and we’ll look up the relevant support information.</div>',
    unsafe_allow_html=True,
)

if not API_BASE_URL:
    st.error("Support is not configured. Set API_BASE_URL in the environment and restart the app.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + " · ".join(message["sources"]))
        if message.get("escalate"):
            st.info("Human review is recommended. Please contact a support agent; no ticket was submitted here.")
        if show_diagnostics and message.get("diagnostics"):
            details = message["diagnostics"]
            with st.expander("Response details"):
                st.write(
                    f"Language: {details.get('language', 'unknown')} · "
                    f"Intent: {details.get('intent', 'unknown')} · "
                    f"Tone: {details.get('sentiment', 'unknown')}"
                )
                for chunk in details.get("retrieved_chunks", []):
                    st.caption(chunk.get("metadata", {}).get("chunk_id", "Retrieved passage"))
                    st.write(chunk.get("document", ""))

question = st.chat_input("Ask about an order, delivery, return, or payment…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Looking up support information…"):
            try:
                result = ask_backend(question)
            except (requests.RequestException, ValueError):
                st.error("Support is temporarily unavailable. Please try again shortly.")
                st.stop()
        answer = result.get("response", "I couldn't find an answer right now.")
        sources = result.get("sources", [])
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + " · ".join(sources))
        if result.get("escalate"):
            st.info("Human review is recommended. Please contact a support agent; no ticket was submitted here.")
        diagnostics = {key: result.get(key) for key in
                       ("language", "intent", "sentiment", "retrieved_chunks")}
        if show_diagnostics:
            with st.expander("Response details"):
                st.write(
                    f"Language: {diagnostics['language']} · Intent: {diagnostics['intent']} · "
                    f"Tone: {diagnostics['sentiment']}"
                )
                for chunk in diagnostics.get("retrieved_chunks") or []:
                    st.caption(chunk.get("metadata", {}).get("chunk_id", "Retrieved passage"))
                    st.write(chunk.get("document", ""))
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources,
        "escalate": result.get("escalate", False), "diagnostics": diagnostics,
    })
