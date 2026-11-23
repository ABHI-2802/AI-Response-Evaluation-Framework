import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluator import evaluate_payload

# Load bundled samples if available
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_INPUT_PATH = BASE_DIR / "sample_input.json"
INPUT_JSON_PATH = BASE_DIR / "input.json"

DEFAULT_SAMPLE = {
    "prompt": "Explain why testing AI responses matters in a customer-support product.",
    "constraints": {
        "max_words": 90,
        "must_include": ["reliability"],
        "must_not_include": ["guaranteed"],
        "expected_format": "bullets"
    },
    "evidence": [
        "AI response evaluations can help identify inaccurate or incomplete answers before they reach customers.",
        "Testing different responses helps teams compare instruction following, relevance, and writing quality."
    ],
    "responses": [
        {
            "agent": "Example A",
            "response": "- Testing improves reliability by helping teams identify inaccurate or incomplete answers.\n- Teams can compare relevance and instruction following before customers see the response."
        },
        {
            "agent": "Example B",
            "response": "AI answers are always perfect. Testing may help."
        }
    ]
}


def load_preset_payload(preset_name: str) -> dict:
    if preset_name == "Sample 1: Customer Support (Constraints & Evidence)" and SAMPLE_INPUT_PATH.exists():
        try:
            with open(SAMPLE_INPUT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    elif preset_name == "Sample 2: Linear Regression (input.json)" and INPUT_JSON_PATH.exists():
        try:
            with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_SAMPLE


st.set_page_config(page_title="AI Response Evaluation Framework", page_icon="🧪", layout="wide")
st.title("AI Response Evaluation Framework")
st.caption("A lightweight, transparent baseline for comparing text responses with heuristic signals.")

with st.sidebar:
    st.header("Input Options")
    preset_choice = st.selectbox(
        "Select a dataset preset:",
        [
            "Sample 1: Customer Support (Constraints & Evidence)",
            "Sample 2: Linear Regression (input.json)",
            "Custom (Upload / Manual Edit)"
        ]
    )

    uploaded = st.file_uploader("Upload JSON file", type=["json"])
    st.divider()
    st.markdown("""
    **Evaluation Dimensions:**
    - 📏 **Instruction Adherence**: Word limits, required/prohibited keywords, format.
    - 🎯 **Prompt Relevance**: TF-IDF cosine similarity to prompt.
    - 📑 **Evidence Alignment**: Sentence-level similarity against evidence citations.
    - 🔗 **Sentence Continuity**: Inter-sentence coherence proxy.
    - ✨ **Surface Fluency**: Surface length, repetition, and punctuation indicators.
    """)
    st.divider()
    st.caption("Scores are heuristic comparison metrics to guide human review and audit, not ground truth.")

# Determine current payload text
if uploaded is not None:
    try:
        payload_text = uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("The uploaded file must be UTF-8 encoded JSON.")
        st.stop()
elif preset_choice != "Custom (Upload / Manual Edit)":
    payload_text = json.dumps(load_preset_payload(preset_choice), indent=2, ensure_ascii=False)
else:
    payload_text = json.dumps(DEFAULT_SAMPLE, indent=2, ensure_ascii=False)

with st.expander("📝 View / Edit Input JSON", expanded=True):
    edited_text = st.text_area(
        "Expected keys: 'prompt', optional 'constraints'/'evidence', and 'responses' or 'agent_responses'",
        value=payload_text,
        height=280,
        label_visibility="collapsed"
    )
    run = st.button("🚀 Evaluate Responses", type="primary", use_container_width=True)

if run or st.session_state.get("evaluated"):
    st.session_state["evaluated"] = True
    try:
        payload = json.loads(edited_text)
        if not isinstance(payload, dict):
            raise ValueError("Top-level JSON value must be an object.")

        responses = payload.get("responses") or payload.get("agent_responses")
        if not isinstance(responses, list) or not responses:
            raise ValueError("Add at least one response in the 'responses' or 'agent_responses' array.")
        if not isinstance(payload.get("prompt", ""), str):
            raise ValueError("'prompt' must be a string.")
        if not isinstance(payload.get("constraints") or {}, dict):
            raise ValueError("'constraints' must be a JSON object.")
        if not isinstance(payload.get("evidence") or [], list):
            raise ValueError("'evidence' must be a JSON array.")

        results = evaluate_payload(payload)
        if not results:
            raise ValueError("No valid responses were found. Each item should be a string or an object with a 'response' field.")
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        st.error(f"Could not evaluate input: {error}")
        st.stop()

    st.subheader("📊 Evaluation Overview")
    winner = results[0]
    avg_score = sum(r["overall_score"] for r in results) / len(results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Ranked Agent", winner["agent"], f"{winner['overall_score']:.1f} / 100")
    with col2:
        st.metric("Total Candidates Evaluated", len(results))
    with col3:
        st.metric("Average Composite Score", f"{avg_score:.1f} / 100")

    # Leaderboard Table
    st.subheader("🏆 Leaderboard")
    rows = []
    for rank, item in enumerate(results, start=1):
        row = {"Rank": rank, "Agent": item["agent"], "Composite Index": item["overall_score"]}
        row.update(item["dimensions"])
        rows.append(row)

    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Charts
    st.subheader("📈 Comparative Analysis")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Composite Index Comparison**")
        overall_df = pd.DataFrame({
            "Agent": [item["agent"] for item in results],
            "Score": [item["overall_score"] for item in results]
        }).set_index("Agent")
        st.bar_chart(overall_df)

    with chart_col2:
        st.markdown("**Dimension Breakdown Across Agents**")
        dim_matrix = {}
        for r in results:
            dim_matrix[r["agent"]] = {k: v if v is not None else 0.0 for k, v in r["dimensions"].items()}
        dim_df = pd.DataFrame(dim_matrix).T
        st.bar_chart(dim_df)

    # Detail Expanders
    st.subheader("🔍 Response Details & Diagnostics")
    for position, item in enumerate(results, start=1):
        with st.expander(f"#{position} {item['agent']} — Score: {item['overall_score']:.1f}/100"):
            st.markdown("**Response Text:**")
            st.text_area(
                f"response_text_{position}",
                value=item["response"] or "(empty response)",
                height=120,
                disabled=True,
                label_visibility="collapsed"
            )

            st.markdown("**Dimension Scores:**")
            dim_rows = [
                {"Dimension": name, "Score": f"{value:.1f}/100" if value is not None else "Not available"}
                for name, value in item["dimensions"].items()
            ]
            st.dataframe(pd.DataFrame(dim_rows), use_container_width=True, hide_index=True)

            st.markdown(f"**Instruction Checks:** `{item['instruction_summary']}`")
            if item["instruction_checks"]:
                checks_df = pd.DataFrame(item["instruction_checks"])
                checks_df["status"] = checks_df["passed"].map({True: "✅ PASS", False: "❌ FAIL"})
                st.dataframe(checks_df[["status", "rule", "detail"]], use_container_width=True, hide_index=True)

            if item.get("notes"):
                st.caption("ℹ️ " + " | ".join(item["notes"]))

    # Download Reports
    st.subheader("💾 Export Results")
    down_col1, down_col2 = st.columns(2)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": payload.get("prompt", ""),
        "total_evaluated": len(results),
        "results": results,
        "limitations": [
            "Heuristic indicators are not calibrated probabilities.",
            "Evidence alignment is textual similarity, not factual verification.",
            "Human review remains necessary for high-stakes or production decisions."
        ]
    }

    with down_col1:
        st.download_button(
            "📥 Download JSON Report",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="evaluation_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with down_col2:
        csv_data = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV Leaderboard",
            data=csv_data,
            file_name="evaluation_leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption("Suggested workflow: compare heuristic scores with human judgments, inspect failure cases, and tune constraints before deployment.")
