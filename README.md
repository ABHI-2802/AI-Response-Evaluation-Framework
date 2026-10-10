# AI Response Evaluation Framework

A small Streamlit application for comparing candidate AI responses with transparent, rule-based and text-similarity indicators.

## What it evaluates

- **Instruction adherence:** checks supported constraints such as word limits, required/prohibited terms, and selected output formats.
- **Prompt relevance:** estimates TF-IDF cosine similarity between the prompt and each response.
- **Evidence alignment (similarity proxy):** compares response sentences with evidence text supplied in the input. This is not factual verification.
- **Sentence continuity (proxy):** estimates lexical similarity between adjacent sentences.
- **Surface fluency (proxy):** applies simple surface-level checks for sentence length, consecutive repeated words, and terminal punctuation.

The composite index combines only available dimensions and renormalizes the weights. It is a heuristic comparison score, not a calibrated probability, factuality guarantee, or replacement for human review.

## Repository files

- app.py — Streamlit interface for editing/uploading JSON and viewing/downloadable results.
- evaluator.py — independent scoring functions.
- sample_input.json — example payload to try.
- tests/test_evaluator.py — unit tests for basic behavior.
- requirements.txt — Python dependencies.

## Run locally

Requires Python 3.10 or newer.

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
\`\`\`

On Windows, activate the environment with .venv\\Scripts\\activate instead of the source command.

## Input format

The application accepts a JSON object with a prompt, optional constraints and evidence, and a list of responses:

\`\`\`json
{
  "prompt": "Explain why testing AI responses matters.",
  "constraints": {
    "max_words": 90,
    "must_include": ["reliability"],
    "must_not_include": ["guaranteed"],
    "expected_format": "bullets"
  },
  "evidence": [
    "Testing helps teams identify inaccurate or incomplete responses."
  ],
  "responses": [
    {
      "agent": "Candidate A",
      "response": "- Testing supports reliability by helping teams identify issues."
    },
    {
      "agent": "Candidate B",
      "response": "Responses should be reviewed before deployment."
    }
  ]
}
\`\`\`

Supported expected_format values are json, bullets, numbered_list, and one_sentence. The evaluator only scores constraint types implemented in evaluator.py; unsupported formats are not scored.

## Tests

Run the unit tests with:

\`\`\`bash
python -m unittest discover -s tests
\`\`\`

## Limitations

This baseline does not call an LLM, run a natural-language-inference model, retrieve live web evidence, or verify claims as true or false. Similarity to supplied evidence can be misleading. Scores should be checked against human judgments before use in any consequential workflow.

## Implementation note

The code in this repository is a fresh baseline implementation and does not include copied source files or assets from another repository. Extend and validate it before describing it as production-ready.
