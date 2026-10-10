"""Transparent, heuristic metrics for comparing text responses.

These metrics are proxies, not proof of correctness or a replacement for human review.
"""
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TRANSITION_WORDS = {
    "also", "because", "but", "consequently", "finally", "first",
    "however", "instead", "meanwhile", "next", "so", "therefore",
    "thus", "while"
}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def similarity(left: str, right: str) -> float:
    """Return TF-IDF cosine similarity between two strings (0 to 1)."""
    if not left.strip() or not right.strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(
            [left, right]
        )
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        return 0.0


def evaluate_instruction_adherence(response: str, constraints: dict[str, Any]) -> dict[str, Any] | None:
    """Check explicitly supplied, supported constraints. Unsupported rules are not guessed."""
    checks: list[dict[str, Any]] = []
    word_count = len(tokens(response))

    if "max_words" in constraints:
        limit = int(constraints["max_words"])
        checks.append({
            "rule": f"At most {limit} words",
            "passed": word_count <= limit,
            "detail": f"Response has {word_count} words."
        })

    if "min_words" in constraints:
        limit = int(constraints["min_words"])
        checks.append({
            "rule": f"At least {limit} words",
            "passed": word_count >= limit,
            "detail": f"Response has {word_count} words."
        })

    for term in constraints.get("must_include", []):
        passed = str(term).casefold() in response.casefold()
        checks.append({
            "rule": f"Must include: {term}",
            "passed": passed,
            "detail": "Required text found." if passed else "Required text not found."
        })

    for term in constraints.get("must_not_include", []):
        passed = str(term).casefold() not in response.casefold()
        checks.append({
            "rule": f"Must not include: {term}",
            "passed": passed,
            "detail": "Prohibited text absent." if passed else "Prohibited text found."
        })

    expected_format = str(constraints.get("expected_format", "")).casefold().strip()
    if expected_format:
        if expected_format == "json":
            import json
            try:
                json.loads(response)
                passed = True
                detail = "Response parses as JSON."
            except (json.JSONDecodeError, TypeError):
                passed = False
                detail = "Response is not valid JSON."
        elif expected_format == "bullets":
            passed = bool(re.search(r"(?m)^\s*[-*•]\s+", response))
            detail = "Bullet list marker found." if passed else "No bullet list marker found."
        elif expected_format == "numbered_list":
            passed = bool(re.search(r"(?m)^\s*\d+[.)]\s+", response))
            detail = "Numbered-list marker found." if passed else "No numbered-list marker found."
        elif expected_format == "one_sentence":
            passed = len(sentences(response)) <= 1
            detail = f"Detected {len(sentences(response))} sentence(s)."
        else:
            passed = None
            detail = "Unsupported expected_format; this rule was not scored."
        if passed is not None:
            checks.append({
                "rule": f"Expected format: {expected_format}",
                "passed": passed,
                "detail": detail
            })

    if not checks:
        return None

    score = 100.0 * sum(1 for check in checks if check["passed"]) / len(checks)
    return {"score": round(score, 1), "checks": checks, "passed": sum(c["passed"] for c in checks),
            "total": len(checks)}


def evidence_alignment(response: str, evidence: list[str]) -> float | None:
    """Measure text similarity to supplied evidence; this does not establish truth."""
    usable_evidence = [str(item).strip() for item in evidence if str(item).strip()]
    response_sentences = sentences(response)
    if not usable_evidence or not response_sentences:
        return None
    best_matches = [
        max(similarity(sentence, item) for item in usable_evidence)
        for sentence in response_sentences
    ]
    return round(100.0 * sum(best_matches) / len(best_matches), 1)


def sentence_continuity(response: str) -> float:
    """A limited proxy based on lexical similarity between adjacent sentences."""
    parts = sentences(response)
    if len(parts) < 2:
        return 100.0
    sims = [similarity(a, b) for a, b in zip(parts, parts[1:])]
    duplicate_pairs = sum(1 for a, b in zip(parts, parts[1:]) if a.casefold() == b.casefold())
    score = (sum(sims) / len(sims)) * 100.0
    score -= (duplicate_pairs / len(sims)) * 20.0
    return round(max(0.0, min(100.0, score)), 1)


def surface_fluency(response: str) -> float:
    """Rule-based surface-quality proxy; it does not check grammar semantically."""
    parts = sentences(response)
    if not response.strip():
        return 0.0
    score = 100.0
    words = tokens(response)

    if words and not response.rstrip().endswith((".", "!", "?", ":", ";", '"', "'", ")", "]", "}")):
        score -= 8.0

    if parts:
        lengths = [len(tokens(part)) for part in parts]
        too_long = sum(max(0, length - 35) for length in lengths)
        too_short = sum(1 for length in lengths if length == 1)
        score -= min(24.0, too_long * 0.7)
        score -= min(15.0, too_short * 5.0)

    repeated_consecutive = sum(
        1 for first, second in zip(words, words[1:]) if first == second
    )
    score -= min(30.0, repeated_consecutive * 10.0)
    return round(max(0.0, min(100.0, score)), 1)


def evaluate_response(
    prompt: str,
    response: str,
    constraints: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    agent: str = "Response",
) -> dict[str, Any]:
    constraints = constraints or {}
    evidence = evidence or []

    adherence = evaluate_instruction_adherence(response, constraints)
    relevance = round(100.0 * similarity(prompt, response), 1)
    alignment = evidence_alignment(response, evidence)
    continuity = sentence_continuity(response)
    fluency = surface_fluency(response)

    dimensions: dict[str, float | None] = {
        "Instruction adherence": adherence["score"] if adherence else None,
        "Prompt relevance": relevance,
        "Evidence alignment (similarity proxy)": alignment,
        "Sentence continuity (proxy)": continuity,
        "Surface fluency (proxy)": fluency,
    }

    weights = {
        "Instruction adherence": 0.30,
        "Prompt relevance": 0.30,
        "Evidence alignment (similarity proxy)": 0.25,
        "Sentence continuity (proxy)": 0.10,
        "Surface fluency (proxy)": 0.05,
    }
    active = [(name, value, weights[name]) for name, value in dimensions.items() if value is not None]
    weight_sum = sum(weight for _, _, weight in active)
    overall = sum(float(value) * weight for _, value, weight in active) / weight_sum if weight_sum else 0.0

    return {
        "agent": agent,
        "response": response,
        "overall_score": round(overall, 1),
        "dimensions": dimensions,
        "instruction_checks": adherence["checks"] if adherence else [],
        "instruction_summary": (
            f'{adherence["passed"]}/{adherence["total"]} supported checks passed'
            if adherence else "No supported constraints supplied"
        ),
        "notes": [
            "Scores are heuristic indicators, not calibrated probabilities.",
            "Evidence alignment measures textual similarity to supplied evidence; it does not verify factual truth."
        ],
    }


def evaluate_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(payload.get("prompt", ""))
    constraints = payload.get("constraints") or {}
    evidence = payload.get("evidence") or []
    responses = payload.get("responses") or []
    results = []

    for index, item in enumerate(responses, start=1):
        if isinstance(item, str):
            text = item
            agent = f"Response {index}"
        elif isinstance(item, dict):
            text = str(item.get("response", ""))
            agent = str(item.get("agent", item.get("name", f"Response {index}")))
        else:
            continue
        results.append(evaluate_response(prompt, text, constraints, evidence, agent))

    return sorted(results, key=lambda result: result["overall_score"], reverse=True)
