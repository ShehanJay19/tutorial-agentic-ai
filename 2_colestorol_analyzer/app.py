from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


st.set_page_config(
    page_title="Cholesterol Analyzer",
    page_icon="🫀",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_REPORT_PATH = BASE_DIR / "blood_work.txt"

load_dotenv()


@st.cache_resource
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


def build_extraction_prompt(report: str) -> str:
    return f"""
You are a medical data extraction assistant.
From the blood report below, extract all test values and classify each one as HIGH, LOW, or NORMAL based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{report}
"""


def build_diet_prompt(extraction_value: str) -> str:
    return f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 4 to 5 lines explaining the patient's condition in simple language.
2. A short, practical Indian diet plan having only two sections: (1) Foods to avoid (2) Foods to eat more of.

Do not include any other sections in the diet plan.

Blood Work Analysis:
{extraction_value}
"""


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eef4f8 100%);
    }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #113a5d 0%, #1e6f5c 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(17, 58, 93, 0.18);
    }
    .subtle-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(17, 58, 93, 0.08);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1 style="margin:0;">Cholesterol Analyzer</h1><p style="margin:0.4rem 0 0 0; font-size:1.02rem;">Paste a lab report and get a clear summary of diet tips and health signals.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Enter Report")
    default_report = SAMPLE_REPORT_PATH.read_text(encoding="utf-8") if SAMPLE_REPORT_PATH.exists() else ""
    report_text = st.text_area(
        "Paste the blood report here",
        value=default_report,
        height=420,
        placeholder="Paste cholesterol, glucose, and other lab values here...",
        label_visibility="collapsed",
    )

    analyze = st.button("Analyze report", type="primary", use_container_width=True)

with right:
    st.subheader("Response")
    st.caption("This is an AI-generated lifestyle-oriented interpretation, not a medical diagnosis.")

    if analyze:
        if not report_text.strip():
            st.warning("Please paste a blood report first.")
        else:
            try:
                llm = get_llm()
                with st.spinner("Analyzing report with LLM..."):
                    extraction_response = llm.invoke(build_extraction_prompt(report_text))
                    extraction_value = extraction_response.text

                    diet_response = llm.invoke(build_diet_prompt(extraction_value))
                    diet_value = diet_response.text

                st.markdown('<div class="subtle-card">', unsafe_allow_html=True)
                st.markdown("### Extracted findings")
                st.write(extraction_value)
                st.markdown("### Health and diet response")
                st.write(diet_value)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as exc:
                st.error("The LLM analysis could not run.")
                st.exception(exc)
    else:
        st.info("Paste a report on the left, then click Analyze report to see the diet and health summary.")
