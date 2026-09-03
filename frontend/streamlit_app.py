"""
Streamlit Web Application — Production-Grade Customer Support Portal
Features:
  - Clean, consumer-facing retail branding (Nova Retail 24/7 Support Concierge)
  - Toggle between "Customer View" (clean retail portal) and "Evaluation / NLP Inspector Mode"
  - Realistic Priority Support Ticket generation upon escalation
  - Quick Suggestion Chips for standard e-commerce queries
  - Feedback rating (Helpful / Not Helpful)
  - Export chat transcript utility
"""

import sys
import time
import random
from datetime import datetime
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.pipeline import ChatPipeline

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nova Retail | Customer Support Concierge",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for User-Ready Service Styling ──────────────────────────────────
st.markdown(
    """
    <style>
    /* Main container background and font */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card */
    .retail-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .retail-brand {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .retail-tagline {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .status-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Escalation Ticket Card */
    .ticket-card {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #e11d48;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 14px 0;
    }
    .ticket-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        color: #9f1239;
        font-size: 15px;
    }
    .ticket-body {
        color: #4c0519;
        font-size: 13.5px;
        margin-top: 8px;
        line-height: 1.5;
    }

    /* Suggestion Chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 20px 0;
    }

    /* Footer note */
    .retail-footer {
        text-align: center;
        padding: 20px 0;
        color: #64748b;
        font-size: 12.5px;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load and Cache Pipeline ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Nova Retail Support AI Engine...")
def load_pipeline():
    """Load the full 4-stage pipeline once and cache it in memory."""
    return ChatPipeline()


pipeline = load_pipeline()


# ── Session State Initialization ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! Welcome to **Nova Retail Support**. I'm your dedicated virtual concierge. How can I assist you with your orders, refunds, shipping, or account today?",
            "time": datetime.now().strftime("%I:%M %p"),
            "metadata": {
                "language": "en",
                "sentiment": "positive",
                "intent": "greeting",
                "escalate": False,
                "ticket_id": None,
                "retrieved_chunks": [],
            },
        }
    ]

if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None


# ── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1556742049-0a67e5572293?w=600&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### **Nova Retail Care**")
    st.caption("Official 24/7 AI-Powered Customer Service Portal")

    st.markdown("---")

    # Developer / Evaluator Mode Toggle
    st.markdown("#### ⚙️ Display Settings")
    dev_mode = st.toggle(
        "🔬 **Evaluator / Diagnostics Mode**",
        value=True,
        help="Turn on to inspect the underlying 4-stage NLP pipeline (Language, Sentiment, Intent, and Retrieved RAG chunks). Turn off for clean customer experience.",
    )

    st.markdown("---")
    st.markdown("#### ⚡ Quick Actions")
    st.caption("Common inquiries you can test with one click:")

    quick_actions = [
        ("📦 Track My Order", "Where is my order? Can you help me track my package status?"),
        ("💳 Request a Refund", "How do I request a refund for an item I returned?"),
        ("😠 File a Complaint", "I am furious! My order arrived completely shattered and nobody answered my emails!"),
        ("🚚 Shipping & Delivery Rates", "What shipping speeds and delivery options do you offer?"),
        ("🔑 Reset Password", "How do I reset my account password?"),
        ("🇪🇸 Consulta en Español", "¿Cuáles son las opciones de entrega y el tiempo estimado?"),
        ("👋 Customer Gratitude", "Thank you so much, your help resolved my issue!"),
    ]

    for label, prompt_text in quick_actions:
        if st.button(label, use_container_width=True):
            st.session_state.preset_prompt = prompt_text

    st.markdown("---")
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Chat history refreshed. What can I help you with today?",
                    "time": datetime.now().strftime("%I:%M %p"),
                    "metadata": {
                        "language": "en",
                        "sentiment": "positive",
                        "intent": "greeting",
                        "escalate": False,
                        "ticket_id": None,
                        "retrieved_chunks": [],
                    },
                }
            ]
            st.rerun()

    with col_export:
        # Build plain-text transcript
        transcript_lines = []
        for m in st.session_state.messages:
            sender = "Customer" if m["role"] == "user" else "Nova Support"
            t = m.get("time", "")
            transcript_lines.append(f"[{t}] {sender}:\n{m['content']}\n")
        transcript_txt = "\n".join(transcript_lines)

        st.download_button(
            label="📥 Export Chat",
            data=transcript_txt,
            file_name=f"nova_support_transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Telemetry metrics in sidebar if in dev mode
    if dev_mode:
        st.markdown("---")
        st.markdown("#### 📊 System Telemetry")
        st.metric("Vector Index Size", f"{pipeline.vector_store.count():,} Passages")
        st.caption("• Dense Embeddings: `all-MiniLM-L6-v2` (384-dim)")
        st.caption("• Intent & Tone Engine: `OpenRouter` (`minimax/minimax-m3:free`)")
        st.caption("• Language Detector: `MultinomialNB (TF-IDF)`")


# ── Retail Customer Support Header ─────────────────────────────────────────────
st.markdown(
    """
    <div class="retail-header">
        <div>
            <div class="retail-brand">🛍️ Nova Retail Concierge</div>
            <div class="retail-tagline">Intelligent, 24/7 customer service for orders, billing, and account inquiries.</div>
        </div>
        <div>
            <div class="status-badge">
                <span class="status-dot"></span> Online & Ready
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Render Chat Messages ───────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🛍️"):
        # Header with timestamp
        t = msg.get("time", "")
        if t:
            st.caption(f"_{t}_")

        # Response content
        st.markdown(msg["content"])

        # If escalated, render a realistic support ticket card
        meta = msg.get("metadata", {})
        if msg["role"] == "assistant" and meta.get("escalate", False):
            ticket_id = meta.get("ticket_id") or f"TKT-{random.randint(10000, 99999)}"
            st.markdown(
                f"""
                <div class="ticket-card">
                    <div class="ticket-header">
                        <span>🚨 Priority Support Ticket Created: #{ticket_id}</span>
                        <span style="font-size: 12px; background: #ffe4e6; padding: 3px 8px; border-radius: 6px;">HIGH PRIORITY</span>
                    </div>
                    <div class="ticket-body">
                        <strong>Assigned To:</strong> Retail Customer Resolutions Specialist<br>
                        <strong>Status:</strong> Under Urgent Review<br>
                        <strong>Estimated Callback:</strong> Within 15 minutes to your registered contact.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Developer / Evaluator Diagnostics Panel
        if dev_mode and msg["role"] == "assistant" and meta:
            lang = meta.get("language", "en").upper()
            sentiment = meta.get("sentiment", "neutral").capitalize()
            intent = meta.get("intent", "general")
            escalate = meta.get("escalate", False)
            chunks = meta.get("retrieved_chunks", [])

            sent_icon = "😊" if sentiment.lower() == "positive" else ("😠" if sentiment.lower() == "negative" else "😐")
            mode_badge = "🚨 Escalated to Tier-2 Human" if escalate else "🤖 Grounded RAG Automated Response"

            with st.expander(f"🔬 Pipeline Diagnostics — [Lang: {lang} | Tone: {sent_icon} {sentiment} | Intent: `{intent}`]"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Language Detected", lang)
                c2.metric("Customer Tone", f"{sent_icon} {sentiment}")
                c3.metric("Classified Intent", intent)
                c4.metric("Escalation Flag", "TRUE 🚨" if escalate else "FALSE ✅")

                if chunks:
                    st.markdown("---")
                    st.markdown(f"**📚 Knowledge Base Retrieved Context ({len(chunks)} Chunks):**")
                    for c_idx, c in enumerate(chunks, start=1):
                        doc = c.get("document", "")
                        resp = c.get("response", "")
                        cat = c.get("category", "RETAIL")
                        dist = c.get("distance")
                        score_text = f" • Cosine Dist: `{dist:.4f}`" if dist is not None else ""
                        with st.container():
                            st.markdown(f"**Chunk {c_idx}** [{cat}]{score_text}")
                            st.caption(f"**Indexed Question:** {doc}")
                            st.info(f"**Gold Response:** {resp}")
                elif intent == "greeting":
                    st.caption("⚡ Direct conversational response applied without RAG database query.")


# ── Chat Input and Processing ──────────────────────────────────────────────────
user_input = st.chat_input("Type your question here (e.g. 'Where is my order?', 'I want a refund')...")

# Check preset prompt or text input
if st.session_state.preset_prompt:
    prompt_to_process = st.session_state.preset_prompt
    st.session_state.preset_prompt = None
elif user_input:
    prompt_to_process = user_input
else:
    prompt_to_process = None

if prompt_to_process:
    current_time = datetime.now().strftime("%I:%M %p")

    # 1. Append & render user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_to_process,
        "time": current_time,
    })
    with st.chat_message("user", avatar="🧑‍💻"):
        st.caption(f"_{current_time}_")
        st.markdown(prompt_to_process)

    # 2. Generate response through the 4-stage pipeline
    with st.chat_message("assistant", avatar="🛍️"):
        asst_time = datetime.now().strftime("%I:%M %p")
        st.caption(f"_{asst_time}_")

        with st.spinner("Checking support knowledge base..."):
            result = pipeline.process(prompt_to_process)

        response_text = result["response"]
        ticket_id = f"TKT-{random.randint(10000, 99999)}" if result["escalate"] else None

        metadata = {
            "language": result["language"],
            "sentiment": result["sentiment"],
            "intent": result["intent"],
            "escalate": result["escalate"],
            "ticket_id": ticket_id,
            "retrieved_chunks": result.get("retrieved_chunks", []),
        }

        st.markdown(response_text)

        # If escalated, show the ticket
        if result["escalate"]:
            st.markdown(
                f"""
                <div class="ticket-card">
                    <div class="ticket-header">
                        <span>🚨 Priority Support Ticket Created: #{ticket_id}</span>
                        <span style="font-size: 12px; background: #ffe4e6; padding: 3px 8px; border-radius: 6px;">HIGH PRIORITY</span>
                    </div>
                    <div class="ticket-body">
                        <strong>Assigned To:</strong> Retail Customer Resolutions Specialist<br>
                        <strong>Status:</strong> Under Urgent Review<br>
                        <strong>Estimated Callback:</strong> Within 15 minutes to your registered contact.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # If in diagnostics mode, render inspector
        if dev_mode:
            lang = metadata["language"].upper()
            sentiment = metadata["sentiment"].capitalize()
            intent = metadata["intent"]
            escalate = metadata["escalate"]
            chunks = metadata["retrieved_chunks"]

            sent_icon = "😊" if sentiment.lower() == "positive" else ("😠" if sentiment.lower() == "negative" else "😐")

            with st.expander(f"🔬 Pipeline Diagnostics — [Lang: {lang} | Tone: {sent_icon} {sentiment} | Intent: `{intent}`]"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Language Detected", lang)
                c2.metric("Customer Tone", f"{sent_icon} {sentiment}")
                c3.metric("Classified Intent", intent)
                c4.metric("Escalation Flag", "TRUE 🚨" if escalate else "FALSE ✅")

                if chunks:
                    st.markdown("---")
                    st.markdown(f"**📚 Knowledge Base Retrieved Context ({len(chunks)} Chunks):**")
                    for c_idx, c in enumerate(chunks, start=1):
                        doc = c.get("document", "")
                        resp = c.get("response", "")
                        cat = c.get("category", "RETAIL")
                        dist = c.get("distance")
                        score_text = f" • Cosine Dist: `{dist:.4f}`" if dist is not None else ""
                        with st.container():
                            st.markdown(f"**Chunk {c_idx}** [{cat}]{score_text}")
                            st.caption(f"**Indexed Question:** {doc}")
                            st.info(f"**Gold Response:** {resp}")
                elif intent == "greeting":
                    st.caption("⚡ Direct conversational response applied without RAG database query.")

    # 3. Save to state
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "time": asst_time,
        "metadata": metadata,
    })


# ── Retail Footer ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="retail-footer">
        Nova Retail Customer Care • Call 1-800-555-0199 (24/7) • support@novaretail.com<br>
        Protected by End-to-End Encryption • Grounded in Official Support Policies
    </div>
    """,
    unsafe_allow_html=True,
)

