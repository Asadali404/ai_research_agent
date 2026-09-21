"""
Streamlit frontend for the AI Research Agent.

The Groq API key is read ONLY from Streamlit secrets — set it in your app's
Settings -> Secrets on Streamlit Community Cloud as:

    GROQ_API_KEY = "your_real_groq_api_key_here"
"""

import streamlit as st
from research_crew import build_research_crew

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("Single-agent CrewAI researcher — Groq (openai/gpt-oss-120b) + DuckDuckGo search")

with st.sidebar:
    st.header("⚙️ About")
    st.markdown(
        "Built with [CrewAI](https://docs.crewai.com), "
        "[Groq](https://groq.com) and DuckDuckGo search."
    )

# ---------------------------------------------------------------------------
# API key — comes only from Streamlit secrets (set on Streamlit Cloud under
# Settings -> Secrets). If it's missing, stop with a clear error instead of
# letting the app crash deeper down.
# ---------------------------------------------------------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is not set in this app's Streamlit secrets. "
        "Go to your app's Settings -> Secrets and add:\n\n"
        '`GROQ_API_KEY = "your_real_groq_api_key_here"`'
    )
    st.stop()

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. The current state of solid-state batteries",
)

run_clicked = st.button("Run Research", type="primary", disabled=not topic)

if run_clicked:
    with st.spinner("Researching... this usually takes 30-90 seconds ⏳"):
        try:
            crew = build_research_crew(topic, groq_api_key)
            result = crew.kickoff()
            # CrewAI returns a CrewOutput object; .raw holds the final text.
            report_text = result.raw if hasattr(result, "raw") else str(result)
        except Exception as exc:
            st.error(f"Something went wrong while researching: {exc}")
            st.stop()

    st.success("Done!")
    st.markdown(report_text)

    st.download_button(
        label="⬇️ Download report as Markdown",
        data=report_text,
        file_name=f"{topic.strip().replace(' ', '_').lower()}_report.md",
        mime="text/markdown",
    )
