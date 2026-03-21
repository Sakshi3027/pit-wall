import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import google.generativeai as genai
from dotenv import load_dotenv
from app.data.ergast_client import (
    get_driver_standings, get_constructor_standings,
    get_season_schedule,
)

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def build_f1_context() -> str:
    try:
        standings    = get_driver_standings(2026)
        constructors = get_constructor_standings(2026)
        schedule     = get_season_schedule(2026)

        driver_str = "\n".join([
            f"  P{row['position']} {row['full_name']} ({row['constructor']}): "
            f"{int(row['points'])} pts, {int(row['wins'])} wins"
            for _, row in standings.iterrows()
        ])
        constructor_str = "\n".join([
            f"  P{row['position']} {row['constructor']}: {int(row['points'])} pts"
            for _, row in constructors.iterrows()
        ])
        schedule_str = "\n".join([
            f"  R{row['round']} {row['gp_name']} — {row['date']} ({row['country']})"
            for _, row in schedule.iterrows()
        ])

        return f"""You are an expert F1 Race Engineer and data analyst with deep knowledge of Formula 1.
You have access to LIVE 2026 F1 season data.

=== 2026 DRIVER STANDINGS (LIVE) ===
{driver_str}

=== 2026 CONSTRUCTOR STANDINGS (LIVE) ===
{constructor_str}

=== 2026 SEASON SCHEDULE ===
{schedule_str}

=== YOUR ROLE ===
- Analyze current championship standings and title fights
- Discuss race strategy: tire compounds, pit windows, undercuts, overcuts
- Compare driver and constructor performance using real data above
- Explain F1 regulations and technical concepts
- Predict race outcomes based on current form
- Discuss historical F1 data and comparisons to previous seasons

Always use real numbers from the standings above. Be confident, specific, and expert.
Talk like a real race engineer — direct, data-driven, insightful.
"""
    except Exception as e:
        return f"""You are an expert F1 Race Engineer. 
Live data temporarily unavailable ({e}). 
Use your F1 knowledge to answer questions confidently."""


def get_gemini_response(context: str, messages: list) -> str:
    """Send conversation to Gemini and get response."""
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your_key then restart."

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=context,
    )

    # Convert messages to Gemini format
    history = []
    for msg in messages[:-1]:  # all except last
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })

    chat = model.start_chat(history=history)

    # Send the latest user message
    last_msg = messages[-1]["content"]
    response = chat.send_message(last_msg)
    return response.text


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Race Engineer — Pit Wall",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#e10600;}
    .sub-header  {font-size:1rem; color:#888; margin-top:-10px;}
    [data-testid="stSidebar"] {background:#0f0f0f;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ Pit Wall")
    st.markdown("---")
    st.markdown("**AI Race Engineer**")
    st.markdown("Powered by Gemini · Live 2026 data")
    st.markdown("---")

    if st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Try asking:**")
    suggestions = [
        "Who will win the 2026 championship?",
        "Why is Verstappen struggling in 2026?",
        "Compare Russell vs Antonelli",
        "How does the undercut strategy work?",
        "Which teams improved most in 2026?",
        "Explain DRS and how it affects strategy",
        "Who has the best race pace right now?",
        "Hamilton at Ferrari — how is he performing?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s[:25]}"):
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"].append({"role": "user", "content": s})
            with st.chat_message("user"):
                st.markdown(s)
            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    try:
                        if "f1_context" not in st.session_state:
                            st.session_state["f1_context"] = build_f1_context()
                        reply = get_gemini_response(
                            st.session_state["f1_context"],
                            st.session_state["messages"],
                        )
                    except Exception as e:
                        if "quota" in str(e).lower() or "429" in str(e):
                            reply = "⚠️ Free quota hit for today. Try again tomorrow — Gemini resets daily."
                        else:
                            reply = f"⚠️ Error: {str(e)}"
                st.markdown(reply)
                st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">AI Race Engineer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Gemini 1.5 Flash · Live 2026 F1 data</p>',
            unsafe_allow_html=True)

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not set. In terminal run: export GEMINI_API_KEY=your_key — then restart streamlit.")
    st.stop()

# Load F1 context once per session
if "f1_context" not in st.session_state:
    with st.spinner("Loading live 2026 F1 data..."):
        st.session_state["f1_context"] = build_f1_context()
    st.success("Live 2026 season data loaded!")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Welcome message
if not st.session_state["messages"]:
    st.markdown("""
    <div style="background:#1a1a1a;border-left:3px solid #e10600;
                padding:1rem 1.5rem;border-radius:0 8px 8px 0;margin-bottom:1rem;">
    <b>Race engineer online.</b> Live 2026 standings loaded —
    Russell leads Antonelli by 4 pts after 2 races.
    Ask me anything about strategy, predictions, or F1.
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your race engineer..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            try:
                reply = get_gemini_response(
                    st.session_state["f1_context"],
                    st.session_state["messages"],
                )
            except Exception as e:
                if "quota" in str(e).lower():
                    reply = "⚠️ Free quota hit for today. Try again tomorrow — Gemini resets daily."
                elif "api" in str(e).lower():
                    reply = f"⚠️ API error: {str(e)}"
                else:
                    reply = f"⚠️ Error: {str(e)}"

        st.markdown(reply)
        st.session_state["messages"].append({
            "role": "assistant", "content": reply
        })