# AI Response Evaluation Framework

A transparent, explainable, and multi-dimensional evaluation framework for benchmarking AI agent responses. Includes a complete Streamlit web dashboard, heuristic scoring engine, diagnostic test suite, and exportable audit reports.

---

## 🌟 Key Features

- **Multi-Dimensional Evaluation:**
  - 📏 **Instruction Adherence:** Checks rule constraints including word limits (`min_words`, `max_words`), mandatory keywords (`must_include`), prohibited terms (`must_not_include`), and formatting (`bullets`, `numbered_list`, `json`, `one_sentence`).
  - 🎯 **Prompt Relevance:** Computes TF-IDF cosine similarity between the input prompt and each candidate response.
  - 📑 **Evidence Alignment (Similarity Proxy):** Compares response sentences against supplied factual evidence or reference text.
  - 🔗 **Sentence Continuity:** Evaluates inter-sentence lexical cohesion and penalizes repetitive sentences.
  - ✨ **Surface Fluency:** Surface-level indicators checking sentence lengths, consecutive word repetitions, and terminal punctuation.
- **Dynamic Composite Index:** Automatically normalizes active evaluation dimension weights into a unified 0–100 score.
- **Interactive Streamlit Dashboard (`app.py`):**
  - Dataset preset selector (Customer Support benchmark vs. Linear Regression multi-agent responses).
  - JSON payload upload or live JSON editing.
  - Metric summary cards (top performer, average score, candidate count).
  - Visual charts: Overall composite score comparison and multi-dimension breakdown.
  - Diagnostic expanders with rule-by-rule pass/fail badges.
  - Export evaluation results to JSON and CSV.
- **Flexible Input Schemas:** Seamlessly supports both `"responses"` and `"agent_responses"`.

---

## 📁 Repository Structure

```
├── app.py                      # Interactive Streamlit dashboard
├── evaluator.py                # Core evaluation engine and heuristic metrics
├── sample_input.json           # Sample payload with constraints and evidence
├── input.json                  # Multi-agent response dataset (Linear Regression)
├── requirements.txt            # Python dependencies
├── tests/
│   └── test_evaluator.py       # Comprehensive unit tests
├── factcheckhackathon.ipynb    # Supplementary experimentation notebook
├── HACKATHON_REPORT.md         # Original Hackathon documentation & architecture
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer

### Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Web Dashboard

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Execute the automated test suite with Python's built-in `unittest`:

```bash
python3 -m unittest discover -s tests
```

---

## 📝 Input Format Specification

The framework accepts JSON payloads formatted in either standard or agent-benchmark structures:

### Option A: Standard Format with Constraints & Evidence

```json
{
  "prompt": "Explain why testing AI responses matters in customer support.",
  "constraints": {
    "max_words": 90,
    "min_words": 10,
    "must_include": ["reliability"],
    "must_not_include": ["guaranteed"],
    "expected_format": "bullets"
  },
  "evidence": [
    "AI response evaluations help teams catch inaccurate answers before they reach users."
  ],
  "responses": [
    {
      "agent": "Model A",
      "response": "- Testing improves reliability by catching errors early.\n- Teams can verify adherence before deployment."
    },
    {
      "agent": "Model B",
      "response": "Everything is always guaranteed."
    }
  ]
}
```

### Option B: Array Format (`agent_responses`)

```json
{
  "prompt": "What are the assumptions of Linear Regression?",
  "agent_responses": [
    "Linear regression assumes linearity, homoscedasticity, and normal residuals...",
    "The primary assumption is a straight line relation..."
  ]
}
```

---

## ⚖️ Heuristic Calibration & Limitations

- The metrics provided are transparent heuristic indicators and text-similarity proxies designed for fast filtering, ranking, and human-in-the-loop audit.
- Evidence alignment estimates lexical similarity to supplied reference texts; it is not a substitution for semantic truth verification or live factual retrieval.
- For high-stakes decisions, use these scores as diagnostic signals alongside human verification.
