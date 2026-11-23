# Agentic Evaluation Framework

*A Scalable and Explainable System for Evaluating AI Agent Responses*

------------------------------------------------------------------------

On running the above factcheckhackathon notebook, it directs you towards the streamlit based app shown in the demo video, where input is the json responses of multiple agents for a single prompt

## 1. Introduction

As AI agents become more capable, they are increasingly deployed to
perform tasks like answering questions, summarizing texts, generating
reports, and assisting with decision-making. While these agents are
powerful, their reliability is not guaranteed. They may hallucinate
facts, ignore instructions, or make unwarranted assumptions.

When you have hundreds of agents producing thousands of responses daily,
manually checking them for correctness and quality is impossible. What
we need is an automated evaluation framework that can reliably assess
agent responses across multiple dimensions.

This report describes the **Agentic Evaluation Framework** we developed
for the E6data hackathon. It is a **Streamlit-based system** that can
score AI responses on:

-   Instruction adherence
-   Factual accuracy
-   Assumption control
-   Coherence and fluency

The system produces interpretable dashboards, scores, reasoning, and
visualizations, making it easier to trust and improve AI systems at
scale.

------------------------------------------------------------------------

## 2. Goals of the Framework

Our hackathon challenge was to design a system that:

-   Accepts prompts, responses, and metadata for hundreds of agents.
-   Evaluates responses across key dimensions:
    -   Instruction-following
    -   Factual accuracy / hallucination detection
    -   Assumption control
    -   Coherence and accuracy
-   Provides interpretable outputs: scores, leaderboards, reports.
-   Scales to thousands of responses.
-   Offers explainability, so developers know *why* a response failed.

**Stretch goals included:** - AI-judge integration (LLM-based
evaluators).
- Visualization tools like heatmaps and charts.
- Support for multiple domains such as Q&A, summarization, and
reasoning.

------------------------------------------------------------------------

## 3. System Architecture

The framework is implemented as a **Streamlit dashboard application**
with the following workflow:

### Input:

-   User prompt
-   Agent response
-   API keys (Groq for LLM judges, SerpAPI for factual checks)

### Evaluation Pipeline:

-   Semantic similarity (SentenceTransformers)
-   Logical entailment/contradiction (Facebook BART NLI model)
-   Sequential coherence (sentence-by-sentence NLI to check
    consistency)
-   Fluency & grammar (GPT-2 perplexity + LanguageTool)
-   Instruction adherence (Regex parsing + Groq LLM judge)
-   Factual accuracy (claim extraction + SerpAPI evidence + LLM
    judgment)

### Output:

-   Numerical scores (0--1)
-   Charts (bar graphs for coherence)
-   JSON reports with reasoning
-   Final composite evaluation

------------------------------------------------------------------------

## 4. Methodology & Components

### 4.1 Semantic Similarity

-   **Model:** SentenceTransformers (MiniLM-L6-v2)
-   Cosine similarity → measures alignment between prompt and response
    meaning.
-   **Score range:** 0--1 (closer to 1 = higher relevance).

### 4.2 Entailment & Logical Relations

-   **Model:** facebook/bart-large-mnli (NLI).
-   Checks if response:
    -   *Entails* (consistent)
    -   *Neutral* (partially related)
    -   *Contradicts* (incorrect).

### 4.3 Sequential Sentence Coherence

-   Splits response into sentences.
-   Runs pairwise NLI checks.
-   Charts show entailment, neutrality, contradiction probabilities.
-   Detects response drift/contradictions.

### 4.4 Fluency & Text Quality

-   **Lexical Diversity:** unique tokens ÷ total tokens (detects
    repetition).
-   **Perplexity (Fluency):** GPT-2 → lower = smoother language.
-   **Grammar Check:** LanguageTool API → detects spelling/grammar
    errors.

### 4.5 Instruction Adherence

-   **Regex-based parsing:** detects constraints (word limits,
    formatting, tone, forbidden content).
-   **LLM Judge (Groq):**
    -   Scores per-constraint.
    -   Provides reasoning.
-   Hybrid approach ensures rule + subtle style checks.

### 4.6 Factual Accuracy

-   **Claim Extraction:** LLM extracts claims.
-   **Query Generation & Evidence Gathering:** converts claims into
    queries, fetches evidence via SerpAPI.
-   **Evaluation:** Groq LLM reviews evidence, outputs JSON (reasoning,
    corrections, factuality).
-   **Scoring:**\
    \[ `\text{Factual Score}`{=tex} = 1 -
    `\frac{\text{Incorrect Sentences}}{\text{Total Sentences}}`{=tex} \]

------------------------------------------------------------------------

## 5. What We Delivered

- *STATE-OF-THE-ART*: We developed a robust evaluation framework that integrates multiple research-backed metrics and open-source libraries into a single streamlined workflow.  

- *VISUALIZED RESULTS*: Our framework includes an intuitive, visually appealing dashboard that highlights response accuracies, trends, and performance comparisons across agents.  

- *COST-EFFICIENT*: The solution is completely free to use in its base form, with the option to enhance capabilities through premium API tiers for higher rate limits.  

- *SCALABLE*: Built with scalability in mind, the system supports parallel processing and can efficiently handle thousands of queries from multiple agents in real-world scenarios.

------------------------------------------------------------------------

## 6. Scoring Pipeline

Each module outputs scores between 0 and 1:

-   Semantic Similarity
-   NLI (Entailment/Contradiction)
-   Sentence Coherence
-   Fluency (Perplexity + Lexical Diversity)
-   Grammar Quality
-   Instruction Adherence
-   Factual Accuracy

Scores can be reported individually (diagnostic) or combined into a
**final leaderboard score**.

------------------------------------------------------------------------

## 7. Interpretability & Explainability

-   Instruction checks → parsed constraints shown.
-   Grammar errors → listed with context.
-   Factual checks → reasoning + suggested corrections.
-   Coherence → stacked bar chart visualization.
-   Developers see *why* an agent failed, not just that it failed.

------------------------------------------------------------------------

## 8. Scalability & Robustness

-   **Streamlit caching** avoids repeated model loading.
-   **Batch processing:** CSV/JSON uploads (1000s of responses).
-   **Hybrid approach:** rule-based + ML + LLM judges.
-   **API integrations:** Groq, SerpAPI.

------------------------------------------------------------------------

## 9. Limitations & Future Work

**Current Limitations:** - Dependence on external APIs → latency +
costs.
- Factual checks limited by search engine coverage.

**Future Improvements:** - Offline fact-check databases (reduce reliance
on live searches).
- Ensemble AI judges (multiple LLM evaluators).
- Domain-specific scoring modules (medicine, law, finance).
- **ML-based score weighting:** train ML models on 1000+ labeled
responses to optimize weights (semantic similarity, factuality, fluency,
etc.).

------------------------------------------------------------------------

## 10. Conclusion

We successfully built an **Agentic Evaluation Framework** that:

-   Scales to thousands of responses.
-   Scores agents across multiple dimensions.
-   Provides explainable outputs and visualizations.
-   Uses a hybrid approach: rules + ML + LLMs.

This framework addresses the urgent need for **trustworthy AI
evaluation** and lays the groundwork for industrial applications where
AI reliability is critical.

------------------------------------------------------------------------

### Team METAVERSE

-   Dev Kumar
-   Divya Gandhi
-   Devansh Upadhyay
