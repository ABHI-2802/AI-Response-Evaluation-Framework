import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from evaluator import evaluate_payload


SAMPLE = {
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


st.set_page_config(page_title="AI Response Evaluation Framework", page_icon="🧪", layout="wide")
st.title("AI Response Evaluation Framework")
st.caption("A lightweight, transparent baseline for comparing text responses with heuristic signals.")

with st.sidebar:
    st.header("Input")
    st.write("Upload a JSON file or edit the example payload. One payload can contain multiple candidate responses.")
    uploaded = st.file_uploader("Upload JSON", type=["json"])
    use_sample = st.checkbox("Load example input", value=uploaded is None)
    st.divider()
    st.caption("This app does not call an LLM or verify facts independently. Scores are indicators for review, not ground truth.")

if uploaded is not None:
    try:
        payload_text = uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("The uploaded file must be UTF-8 encoded JSON.")
        st.stop()
elif use_sample:
    payload_text = json.dumps(SAMPLE, indent=2, ensure_ascii=False)
else:
    payload_text = json.dumps(SAMPLE, indent=2, ensure_ascii=False)

with st.expander("Edit input JSON", expanded=True):
    edited_text = st.text_area(
        "Expected keys: prompt, optional constraints/evidence, and responses",
        value=payload_text,
        height=300,
        label_visibility="collapsed"
    )
    run = st.button("Evaluate responses", type="primary", use_container_width=True)

if run:
    try:
        payload = json.loads(edited_text)
        if not isinstance(payload, dict):
            raise ValueError("Top-level JSON value must be an object.")
        if not isinstance(payload.get("responses"), list) or not payload["responses"]:
            raise ValueError("Add at least one response in the 'responses' array.")
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

    st.subheader("Comparison")
    st.caption("The composite index uses only available dimensions and renormalizes their weights.")
    rows = []
    for item in results:
        row = {"Agent": item["agent"], "Composite index": item["overall_score"]}
        row.update(item["dimensions"])
        rows.append(row)

    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)

    winner = results[0]
    left, right = st.columns([2, 1])
    with left:
        st.metric("Highest composite index", winner["agent"], f'{winner["overall_score"]:.1f}/100')
    with right:
        st.metric("Responses evaluated", len(results))

    st.subheader("Response details")
    for position, item in enumerate(results, start=1):
        with st.expander(f'{position}. {item["agent"]} — {item["overall_score"]:.1f}/100'):
            st.markdown("**Response**")
            st.write(item["response"] or "(empty response)")
            st.markdown("**Dimension scores**")
            dim_rows = [
                {"Dimension": name, "Score": f"{value:.1f}/100" if value is not None else "Not available"}
                for name, value in item["dimensions"].items()
            ]
            st.dataframe(pd.DataFrame(dim_rows), use_container_width=True, hide_index=True)
            st.markdown(f'**Instruction checks:** {item["instruction_summary"]}')
            if item["instruction_checks"]:
                checks_df = pd.DataFrame(item["instruction_checks"])
                checks_df["status"] = checks_df["passed"].map({True: "PASS", False: "FAIL"})
                st.dataframe(checks_df[["status", "rule", "detail"]], use_container_width=True, hide_index=True)
            st.info(" ".join(item["notes"]))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": payload.get("prompt", ""),
        "results": results,
        "limitations": [
            "Heuristic indicators are not calibrated probabilities.",
            "Evidence alignment is textual similarity, not factual verification.",
            "Human review remains necessary for high-stakes or production decisions."
        ]
    }
    st.download_button(
        "Download evaluation report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="evaluation_report.json",
        mime="application/json",
    )

st.divider()
st.caption("Suggested workflow: compare the scores with human judgments, inspect failure cases, and tune rules before using metrics in a real evaluation pipeline.")
