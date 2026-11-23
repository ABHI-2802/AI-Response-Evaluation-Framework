"""Streamlit Web Dashboard for the Agentic Evaluation Framework.

Features:
- MiniLM Semantic Similarity
- BART NLI Logical Entailment & Contradiction
- Sequential Sentence Coherence with Visual Stacked / Grouped Bar Charts
- GPT-2 Perplexity & Lexical Diversity
- LanguageTool Public Grammar Verification
- Groq LLM Judge for Instruction Adherence
- Claim Extraction + SerpAPI Evidence + Groq Factuality Judge
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from evaluator import evaluate_payload, evaluate_response


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Evaluation Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Agentic Evaluation Framework")
st.markdown(
    "*A Scalable and Explainable System for Evaluating AI Agent Responses across Semantic Similarity, "
    "NLI Entailment, Coherence, Fluency, Grammar, Instruction Adherence, and Factuality.*"
)


# ---------------------------------------------------------------------------
# Sample Datasets Loader
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

def load_json_file(filename: str) -> Dict[str, Any] | None:
    path = ROOT_DIR / filename
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

SAMPLE_INPUT = load_json_file("sample_input.json")
LINEAR_REGRESSION_INPUT = load_json_file("input.json")


# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration & Credentials")

st.sidebar.subheader("🔑 API Keys (Optional)")
groq_key_env = os.getenv("GROQ_API_KEY", "")
serpapi_key_env = os.getenv("SERP_API_KEY", "")

groq_api_key = st.sidebar.text_input(
    "Groq API Key (for LLM Judge & Fact Check)",
    value=groq_key_env,
    type="password",
    help="Used to power the Groq LLM judge for instruction adherence and factual error checks.",
)

serpapi_api_key = st.sidebar.text_input(
    "SerpAPI Key (for Web Evidence Retrieval)",
    value=serpapi_key_env,
    type="password",
    help="Used to retrieve real-time Google Search evidence snippets for claim fact-checking.",
)

judge_model = st.sidebar.selectbox(
    "LLM Judge Model",
    options=["llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
    index=0,
)

# Badges
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if groq_api_key.strip():
        st.success("Groq Ready", icon="🟢")
    else:
        st.info("Groq Offline", icon="⚪")

with col_b2:
    if serpapi_api_key.strip():
        st.success("SerpAPI Ready", icon="🟢")
    else:
        st.info("SerpAPI Offline", icon="⚪")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Input Dataset")

preset_options = ["Linear Regression (input.json)", "Customer Support (sample_input.json)", "Upload JSON", "Manual Entry"]
selected_preset = st.sidebar.radio("Select Input Mode", preset_options)

active_payload: Dict[str, Any] | None = None

if selected_preset == "Linear Regression (input.json)":
    active_payload = LINEAR_REGRESSION_INPUT
elif selected_preset == "Customer Support (sample_input.json)":
    active_payload = SAMPLE_INPUT
elif selected_preset == "Upload JSON":
    uploaded_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])
    if uploaded_file is not None:
        try:
            active_payload = json.load(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to parse JSON: {e}")
elif selected_preset == "Manual Entry":
    manual_prompt = st.sidebar.text_area("User Prompt", value="What are the key benefits of automated AI testing?")
    manual_resp1 = st.sidebar.text_area("Candidate Response 1", value="Automated testing improves reliability, catches hallucinations, and benchmarks performance.")
    manual_resp2 = st.sidebar.text_area("Candidate Response 2", value="AI systems do not require testing because models are always accurate.")
    active_payload = {
        "prompt": manual_prompt,
        "responses": [
            {"agent": "Candidate Alpha", "response": manual_resp1},
            {"agent": "Candidate Beta", "response": manual_resp2},
        ],
    }

run_button = st.sidebar.button("🚀 Run Full Evaluation", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Evaluation Execution & Session State Caching
# ---------------------------------------------------------------------------
if run_button and active_payload:
    with st.spinner("Executing multi-dimensional evaluation pipeline (MiniLM, BART NLI, GPT-2, LanguageTool, Groq)..."):
        results = evaluate_payload(
            payload=active_payload,
            groq_api_key=groq_api_key.strip() if groq_api_key else None,
            serpapi_api_key=serpapi_api_key.strip() if serpapi_api_key else None,
        )
        st.session_state["eval_results"] = results
        st.session_state["eval_payload"] = active_payload
        st.session_state["eval_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

results: List[Dict[str, Any]] | None = st.session_state.get("eval_results")


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
if not results:
    if active_payload:
        st.info("👉 Ready to evaluate. Click **'Run Full Evaluation'** in the sidebar to run all modules.")
        with st.expander("Preview Active Dataset Payload", expanded=True):
            st.json(active_payload)
    else:
        st.warning("No input dataset loaded. Select or upload a JSON dataset in the sidebar to begin.")
else:
    tab_trends, tab_detail, tab_export = st.tabs([
        "📊 Overall Trends & Leaderboard",
        "🔍 Individual Response Deep Dive",
        "💾 Export Audit Reports",
    ])

    # =========================================================================
    # TAB 1: Overall Trends & Leaderboard
    # =========================================================================
    with tab_trends:
        st.subheader("Leaderboard & Overall Metric Trends")

        # Top-level metrics
        top_agent = results[0]["agent"]
        top_score = results[0]["overall_score"]
        avg_semantic = np.mean([r["semantic_score"] for r in results])
        avg_fluency = np.mean([r["normalized_perplexity"] for r in results])
        avg_grammar = np.mean([r["grammar_score"] for r in results])
        evaluated_fact = [r["factual_accuracy_score"] for r in results if r.get("factual_accuracy_score") is not None]
        avg_fact_str = f"{np.mean(evaluated_fact):.2f}" if evaluated_fact else "Not evaluated"

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("🏆 Top Performer", f"{top_agent}", f"{top_score:.1f}/100")
        m_col2.metric("🎯 Avg Semantic", f"{avg_semantic:.2f}")
        m_col3.metric("✨ Avg Fluency", f"{avg_fluency:.2f}")
        m_col4.metric("📝 Avg Grammar", f"{avg_grammar:.2f}")
        m_col5.metric("⚖️ Avg Factuality", avg_fact_str)

        # Leaderboard Table
        st.markdown("### 🏅 Candidate Ranking")
        summary_rows = []
        for r in results:
            fact_val = f"{r['factual_accuracy_score']:.2f}" if r.get("factual_accuracy_score") is not None else "Not evaluated"
            summary_rows.append({
                "Agent": r["agent"],
                "Composite Score (0-100)": r["overall_score"],
                "Semantic Sim": f"{r['semantic_score']:.2f}",
                "Prompt Entailment": f"{r['entail_prob']:.2f}",
                "Sentence Coherence": f"{r['coherence']['overall_entailment']:.2f}",
                "Fluency Score": f"{r['normalized_perplexity']:.2f}",
                "Grammar Score": f"{r['grammar_score']:.2f}",
                "Instruction Score": f"{r['instruction_adherence'].get('final_score', 1.0):.2f}",
                "Factual Score": fact_val,
            })
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)

        # Matplotlib 2x2 Trend Charts
        st.markdown("### 📈 Metric Trends Across Candidates")
        n = list(range(1, len(results) + 1))
        agent_labels = [r["agent"] for r in results]

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        # Chart 1: Semantic
        sem_scores = [r["semantic_score"] for r in results]
        axs[0, 0].plot(n, sem_scores, marker="o", color="#4CAF50", linewidth=2)
        axs[0, 0].axhline(y=avg_semantic, linestyle="--", color="#2E7D32", label=f"Avg: {avg_semantic:.2f}")
        axs[0, 0].set_title("Semantic Similarity (MiniLM)")
        axs[0, 0].set_xticks(n)
        axs[0, 0].set_xticklabels(agent_labels, rotation=30)
        axs[0, 0].set_ylim(0, 1.05)
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle=":", alpha=0.6)

        # Chart 2: Factuality
        if evaluated_fact:
            fact_scores = [r["factual_accuracy_score"] if r.get("factual_accuracy_score") is not None else 0.0 for r in results]
            avg_fact_val = float(np.mean(evaluated_fact))
            axs[0, 1].plot(n, fact_scores, marker="s", color="#2196F3", linewidth=2)
            axs[0, 1].axhline(y=avg_fact_val, linestyle="--", color="#1565C0", label=f"Avg: {avg_fact_val:.2f}")
            axs[0, 1].legend()
        else:
            axs[0, 1].text(0.5, 0.5, "Not evaluated\n(Credentials not provided)", ha="center", va="center", transform=axs[0, 1].transAxes, fontsize=11, color="gray")
        axs[0, 1].set_title("Factual Accuracy (SerpAPI + LLM Judge)")
        axs[0, 1].set_xticks(n)
        axs[0, 1].set_xticklabels(agent_labels, rotation=30)
        axs[0, 1].set_ylim(0, 1.05)
        axs[0, 1].grid(True, linestyle=":", alpha=0.6)

        # Chart 3: Fluency
        fluency_scores = [r["normalized_perplexity"] for r in results]
        axs[1, 0].plot(n, fluency_scores, marker="^", color="#9C27B0", linewidth=2)
        axs[1, 0].axhline(y=avg_fluency, linestyle="--", color="#6A1B9A", label=f"Avg: {avg_fluency:.2f}")
        axs[1, 0].set_title("Normalized Fluency (GPT-2 Perplexity)")
        axs[1, 0].set_xticks(n)
        axs[1, 0].set_xticklabels(agent_labels, rotation=30)
        axs[1, 0].set_ylim(0, 1.05)
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle=":", alpha=0.6)

        # Chart 4: Grammar
        gram_scores = [r["grammar_score"] for r in results]
        axs[1, 1].plot(n, gram_scores, marker="d", color="#FF9800", linewidth=2)
        axs[1, 1].axhline(y=avg_grammar, linestyle="--", color="#E65100", label=f"Avg: {avg_grammar:.2f}")
        axs[1, 1].set_title("Grammar Quality Score (LanguageTool)")
        axs[1, 1].set_xticks(n)
        axs[1, 1].set_xticklabels(agent_labels, rotation=30)
        axs[1, 1].set_ylim(0, 1.05)
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    # =========================================================================
    # TAB 2: Individual Response Deep Dive
    # =========================================================================
    with tab_detail:
        agent_names = [r["agent"] for r in results]
        selected_agent = st.selectbox("Select Candidate to Inspect", agent_names, index=0)
        selected_data = next(r for r in results if r["agent"] == selected_agent)

        st.markdown(f"## 📋 Detailed Analysis: **{selected_agent}**")
        st.info(f"**Prompt:** {selected_data['prompt']}")
        with st.expander("View Full Response Text", expanded=True):
            st.markdown(selected_data["agent_response"])

        st.markdown("---")

        # 1.1 Semantic Search Score
        st.subheader("1.1. Semantic Search Score (MiniLM)")
        st.metric("MiniLM Semantic Score (0–1)", f"{selected_data['semantic_score']:.2f}")
        st.caption("Measures sentence embedding cosine similarity between the input prompt and agent response.")

        # 1.2 Prompt-Response Entailment
        st.subheader("1.2. Prompt-Response Entailment (BART NLI)")
        col_nli_1, col_nli_2, col_nli_3 = st.columns(3)
        col_nli_1.metric("Entailment", f"{selected_data['entail_prob']:.2f}")
        col_nli_2.metric("Neutrality", f"{selected_data['neutral_prob']:.2f}")
        col_nli_3.metric("Contradiction", f"{selected_data['contradict_prob']:.2f}")
        st.caption("Evaluates whether the response logically follows, remains neutral, or contradicts the premise.")

        # 1.3 Sequential Sentence Coherence
        st.subheader("1.3. Sequential Sentence Coherence (BART NLI)")
        coherence = selected_data["coherence"]
        col_seq_1, col_seq_2, col_seq_3 = st.columns(3)
        col_seq_1.metric("Overall Entailment", f"{coherence['overall_entailment']:.2f}")
        col_seq_2.metric("Overall Neutrality", f"{coherence['overall_neutrality']:.2f}")
        col_seq_3.metric("Overall Contradiction", f"{coherence['overall_contradiction']:.2f}")

        pairs = coherence["pairs"]
        entail = coherence["entail_scores"]
        neutral = coherence["neutral_scores"]
        contradict = coherence["contradict_scores"]

        if pairs:
            fig_coh, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

            # Stacked bar chart
            ax1.bar(pairs, contradict, label="Contradiction", color="#F44336")
            ax1.bar(pairs, neutral, bottom=contradict, label="Neutral", color="#9E9E9E")
            ax1.bar(
                pairs,
                entail,
                bottom=[c + n for c, n in zip(contradict, neutral)],
                label="Entailment",
                color="#4CAF50",
            )
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Probability")
            ax1.set_xlabel("Adjacent Sentence Pair")
            ax1.set_title("Sequential Sentence Probability Distribution (Stacked)")
            ax1.legend(loc="upper right")
            ax1.tick_params(axis="x", rotation=35)

            # Grouped bar chart
            x = np.arange(len(pairs))
            width = 0.25
            ax2.bar(x - width, contradict, width, label="Contradiction", color="#F44336")
            ax2.bar(x, neutral, width, label="Neutral", color="#9E9E9E")
            ax2.bar(x + width, entail, width, label="Entailment", color="#4CAF50")
            ax2.set_xticks(x)
            ax2.set_xticklabels(pairs, rotation=35)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Probability")
            ax2.set_xlabel("Adjacent Sentence Pair")
            ax2.set_title("Sequential Sentence Relations (Grouped NLI)")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            st.pyplot(fig_coh)
            plt.clf()
        else:
            st.info("Single-sentence response; sequential coherence requires at least two sentences.")

        # 1.4 Text Fluency and Quality
        st.subheader("1.4. Text Fluency & Language Quality")
        col_div1, col_div2, col_ppl1, col_ppl2 = st.columns(4)
        col_div1.metric("Unigram Diversity", f"{selected_data['unigrams_diversity']:.2f}")
        col_div2.metric("Bigram Diversity", f"{selected_data['bigrams_diversity']:.2f}")
        col_ppl1.metric("Normalized Fluency", f"{selected_data['normalized_perplexity']:.2f}")
        col_ppl2.metric("Raw Perplexity (GPT-2)", f"{selected_data['raw_perplexity']:.1f}")

        st.metric("LanguageTool Grammar Score", f"{selected_data['grammar_score']:.2f}")
        grammar_errors = selected_data["grammar_errors"]
        if grammar_errors:
            st.warning(f"Detected {len(grammar_errors)} potential grammar/spelling issues:")
            for err in grammar_errors[:5]:
                st.markdown(
                    f"- **Issue:** {err.get('message', 'Issue')} | "
                    f"**Context:** `{err.get('context', {}).get('text', '')}`"
                )
        else:
            st.success("No grammar or spelling errors flagged.")

        # 1.5 Instruction Adherence Check
        st.subheader("1.5. Instruction Adherence Check")
        inst_eval = selected_data["instruction_adherence"]
        with st.expander("Parsed Constraints", expanded=False):
            st.json(selected_data["parsed_constraints"])

        col_s1, col_s2, col_s3, col_s4, col_s5, col_final = st.columns(6)
        col_s1.metric("Quantity", f"{inst_eval.get('quantity_score', 1.0):.2f}")
        col_s2.metric("Format", f"{inst_eval.get('format_score', 1.0):.2f}")
        col_s3.metric("Length", f"{inst_eval.get('length_score', 1.0):.2f}")
        col_s4.metric("Style", f"{inst_eval.get('style_score', 1.0):.2f}")
        col_s5.metric("Do/Don't", f"{inst_eval.get('do_dont_score', 1.0):.2f}")
        col_final.metric("Adherence Score", f"{inst_eval.get('final_score', 1.0):.2f}")

        st.markdown(f"**Judge Reasoning:** {inst_eval.get('reasoning', 'No reasoning supplied.')}")

        # 1.6 Factual Accuracy Check
        st.subheader("1.6. Factual Accuracy Check")
        if selected_data.get("factual_accuracy_score") is not None:
            st.metric("Factual Accuracy Score", f"{selected_data['factual_accuracy_score']:.2f}")
        else:
            st.metric("Factual Accuracy Score", "Not evaluated")
            st.info(
                "Factual accuracy is **Not evaluated** because Groq or SerpAPI credentials were not provided. "
                "Its 15% weight has been proportionally redistributed across the other active dimensions."
            )

        if selected_data["claims"]:
            with st.expander("Extracted Claims", expanded=False):
                st.json(selected_data["claims"])

        if selected_data["evidences"]:
            with st.expander("Gathered Evidence Snippets (SerpAPI)", expanded=False):
                for idx, ev in enumerate(selected_data["evidences"], 1):
                    st.markdown(f"{idx}. {ev}")

        with st.expander("Fact-Checking Judge Verdict", expanded=True):
            st.json(selected_data["AI_as_judge_results"])

    # =========================================================================
    # TAB 3: Export Audit Reports
    # =========================================================================
    with tab_export:
        st.subheader("Export Evaluation Artifacts")
        st.write("Download machine-readable audit trails and leaderboard summaries.")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download Audit Report (JSON)",
                data=json_str,
                file_name="agentic_evaluation_report.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_exp2:
            csv_data = df_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Leaderboard (CSV)",
                data=csv_data,
                file_name="leaderboard_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
