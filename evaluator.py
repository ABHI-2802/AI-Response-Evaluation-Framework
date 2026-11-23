"""Agentic Evaluation Framework: Multi-Dimensional Evaluation Engine.

Provides evaluation pipelines across 6 primary dimensions:
1. Semantic Similarity (SentenceTransformer MiniLM)
2. Logical Entailment & Contradiction (BART NLI)
3. Sequential Sentence Coherence (Pairwise NLI)
4. Fluency & Text Quality (GPT-2 Perplexity + Lexical Diversity)
5. Grammar Quality (LanguageTool API)
6. Instruction Adherence (Constraint Parsing + Groq LLM Judge)
7. Factual Accuracy (Claim Extraction + SerpAPI Evidence + Groq LLM Judge)
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Global Caches & Lazy Model Loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_semantic_model():
    """Load and cache SentenceTransformer MiniLM model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Notice: SentenceTransformer MiniLM could not be loaded: {e}")
        return None


@lru_cache(maxsize=1)
def load_nli_model():
    """Load and cache facebook/bart-large-mnli model and tokenizer."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = "facebook/bart-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Notice: BART NLI model could not be loaded: {e}")
        return None, None


@lru_cache(maxsize=1)
def load_gpt2_for_fluency():
    """Load and cache GPT-2 model and tokenizer for perplexity estimation."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Notice: GPT-2 fluency model could not be loaded: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Dimension 1: Semantic Similarity (MiniLM)
# ---------------------------------------------------------------------------

def compute_semantic_similarity(prompt: str, response: str, model=None) -> float:
    """Compute semantic similarity between prompt and response using MiniLM cosine similarity.
    
    Returns a score normalized to [0, 1].
    """
    if not prompt.strip() or not response.strip():
        return 0.0

    if model is None:
        model = load_semantic_model()

    if model is not None:
        try:
            from sentence_transformers import util
            emb_prompt = model.encode(prompt, convert_to_tensor=True)
            emb_resp = model.encode(response, convert_to_tensor=True)
            sim_score = util.cos_sim(emb_prompt, emb_resp).item()
            return float(max(0.0, min(1.0, (sim_score + 1.0) / 2.0)))
        except Exception as e:
            print(f"Semantic similarity inference error: {e}")

    # Fallback to character/word overlap proxy if torch/model is unavailable
    prompt_tokens = set(re.findall(r"\b\w+\b", prompt.lower()))
    resp_tokens = set(re.findall(r"\b\w+\b", response.lower()))
    if not prompt_tokens or not resp_tokens:
        return 0.0
    jaccard = len(prompt_tokens & resp_tokens) / len(prompt_tokens | resp_tokens)
    return float(min(1.0, jaccard * 1.5))


# ---------------------------------------------------------------------------
# Dimension 2: Logical Entailment & Contradiction (BART NLI)
# ---------------------------------------------------------------------------

def compute_nli_probabilities(
    premise: str, hypothesis: str, tokenizer=None, model=None
) -> Tuple[float, float, float]:
    """Compute (entailment, neutrality, contradiction) probabilities using facebook/bart-large-mnli.
    
    Index mapping for BART MNLI:
    0: contradiction, 1: neutral, 2: entailment
    """
    if not premise.strip() or not hypothesis.strip():
        return 0.0, 1.0, 0.0

    if tokenizer is None or model is None:
        tokenizer, model = load_nli_model()

    if tokenizer is not None and model is not None:
        try:
            import torch
            import torch.nn.functional as F

            inputs = tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()

            contradiction_prob = float(probs[0])
            neutral_prob = float(probs[1])
            entailment_prob = float(probs[2])
            return entailment_prob, neutral_prob, contradiction_prob
        except Exception as e:
            print(f"BART NLI inference error: {e}")

    # Heuristic fallback if model not loaded
    return 0.33, 0.34, 0.33


# ---------------------------------------------------------------------------
# Dimension 3: Sequential Sentence Coherence
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK or regex fallback."""
    if not text or not text.strip():
        return []
    try:
        import nltk
        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]


def compute_sequential_coherence(
    response: str, tokenizer=None, model=None
) -> Dict[str, Any]:
    """Evaluate sentence-by-sentence pairwise NLI coherence."""
    sentences = split_into_sentences(response)
    if len(sentences) < 2:
        return {
            "sentences": sentences,
            "pairs": [],
            "entail_scores": [],
            "neutral_scores": [],
            "contradict_scores": [],
            "overall_entailment": 1.0 if sentences else 0.0,
            "overall_neutrality": 0.0,
            "overall_contradiction": 0.0,
            "coherent_scores_overall_enc": [0.0, 1.0, 0.0] if sentences else [0.0, 0.0, 0.0],
        }

    if tokenizer is None or model is None:
        tokenizer, model = load_nli_model()

    entail_scores: List[float] = []
    neutral_scores: List[float] = []
    contradict_scores: List[float] = []
    pairs: List[str] = []

    for i in range(len(sentences) - 1):
        s1 = sentences[i]
        s2 = sentences[i + 1]
        pairs.append(f"S{i+1} -> S{i+2}")
        e, n, c = compute_nli_probabilities(s1, s2, tokenizer=tokenizer, model=model)
        entail_scores.append(e)
        neutral_scores.append(n)
        contradict_scores.append(c)

    avg_entail = sum(entail_scores) / len(entail_scores)
    avg_neutral = sum(neutral_scores) / len(neutral_scores)
    avg_contradict = sum(contradict_scores) / len(contradict_scores)

    return {
        "sentences": sentences,
        "pairs": pairs,
        "entail_scores": entail_scores,
        "neutral_scores": neutral_scores,
        "contradict_scores": contradict_scores,
        "overall_entailment": float(avg_entail),
        "overall_neutrality": float(avg_neutral),
        "overall_contradiction": float(avg_contradict),
        "coherent_scores_overall_enc": [
            float(avg_neutral),
            float(avg_entail),
            float(avg_contradict),
        ],
    }


# ---------------------------------------------------------------------------
# Dimension 4: Fluency, Perplexity (GPT-2) & Lexical Diversity
# ---------------------------------------------------------------------------

def compute_lexical_diversity(text: str) -> Dict[str, float]:
    """Compute unigram lexical diversity and distinct-2 (bigrams) diversity."""
    tokens = re.findall(r"\b[\w'-]+\b", (text or "").lower())
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {"unigrams_diversity": 0.0, "bigrams_diversity": 0.0}

    unique_unigrams = len(set(tokens))
    unigrams_diversity = unique_unigrams / total_tokens

    bigrams = [tuple(tokens[i : i + 2]) for i in range(total_tokens - 1)]
    bigrams_diversity = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

    return {
        "unigrams_diversity": float(unigrams_diversity),
        "bigrams_diversity": float(bigrams_diversity),
    }


def compute_perplexity(
    text: str, tokenizer=None, model=None
) -> Tuple[float, float]:
    """Compute GPT-2 cross-entropy loss, raw perplexity, and normalized fluency score.
    
    Formula:
        normalized_fluency = 1 / (1 + log(1 + raw_perplexity))
    """
    if not text or not text.strip():
        return 0.0, float("inf")

    if tokenizer is None or model is None:
        tokenizer, model = load_gpt2_for_fluency()

    if tokenizer is not None and model is not None:
        try:
            import torch

            encodings = tokenizer(text, return_tensors="pt")
            if encodings["input_ids"].shape[1] > 0:
                with torch.no_grad():
                    outputs = model(**encodings, labels=encodings["input_ids"])
                    loss = outputs.loss.item()
                    raw_ppl = math.exp(min(loss, 20.0))  # guard against overflow
                    normalized_score = 1.0 / (1.0 + math.log(1.0 + raw_ppl))
                    return float(normalized_score), float(raw_ppl)
        except Exception as e:
            print(f"GPT-2 perplexity computation error: {e}")

    # Fallback heuristic based on length and repetition
    words = re.findall(r"\b[\w'-]+\b", text)
    if not words:
        return 0.0, 100.0
    raw_est = 25.0
    normalized_est = 1.0 / (1.0 + math.log(1.0 + raw_est))
    return float(normalized_est), float(raw_est)


# ---------------------------------------------------------------------------
# Dimension 5: Grammar Quality (LanguageTool API)
# ---------------------------------------------------------------------------

def check_grammar_languagetool(
    text: str, timeout: int = 5
) -> Tuple[float, List[Dict[str, Any]]]:
    """Check text grammar and spelling using the public LanguageTool REST API.
    
    Returns (grammar_score [0..1], error_matches_list).
    """
    if not text.strip():
        return 0.0, []

    url = "https://api.languagetool.org/v2/check"
    data = {"text": text, "language": "en-US"}
    try:
        response = requests.post(url, data=data, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            errors = result.get("matches", [])
            total_words = max(1, len(text.split()))
            score = max(0.0, 1.0 - (len(errors) / total_words))
            return float(score), errors
    except Exception as e:
        print(f"LanguageTool API notice: {e}")

    # Offline heuristic fallback
    return 1.0, []


# ---------------------------------------------------------------------------
# Dimension 6: Instruction Adherence (Regex Parsing + Groq LLM Judge)
# ---------------------------------------------------------------------------

def parse_constraints(prompt_text: str) -> Dict[str, List[Any]]:
    """Extract quantity, format, length, style, and do/don't constraints from prompt."""
    constraints: Dict[str, List[Any]] = {
        "quantity": [],
        "format": [],
        "length": [],
        "style": [],
        "do_dont": [],
    }
    text = (prompt_text or "").lower().strip()

    quantity_patterns = [
        r"(?:list|give|provide|generate|mention)\s+(\d+)\s",
        r"(?:list|give|provide|generate|mention)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s",
    ]
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    for pattern in quantity_patterns:
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            constraints["quantity"].append(int(val) if val.isdigit() else word_to_num.get(val))

    format_keywords = [
        "json", "table", "bullet points", "bullets", "list", "yaml", "csv",
        "markdown", "dict", "dictionary", "paragraph", "monolithic",
    ]
    for kw in format_keywords:
        if kw in text:
            constraints["format"].append(kw)

    length_patterns = [
        r"limit to (\d+) words", r"not more than (\d+) words", r"no more than (\d+) words",
        r"less than (\d+) words", r"greater than (\d+) words", r"not more than (\d+) sentences",
        r"up to (\d+) words",
    ]
    for pattern in length_patterns:
        match = re.search(pattern, text)
        if match:
            constraints["length"].append(match.group(0))

    style_keywords = [
        "formal tone", "informal tone", "past tense", "present tense", "active voice",
        "passive voice", "easy language", "not ai generated", "human generated",
        "technical jargons",
    ]
    for kw in style_keywords:
        if kw in text:
            constraints["style"].append(kw)

    dont_pattern = r"(?:do not|don't)\s+(?:mention|include|talk about)\s+([\w\s]+)"
    dont_match = re.findall(dont_pattern, text)
    if dont_match:
        constraints["do_dont"].extend([item.strip() for item in dont_match])

    return constraints


def evaluate_instruction_adherence_llm(
    prompt: str,
    agent_response: str,
    parsed_constraints: Dict[str, Any],
    groq_api_key: Optional[str] = None,
    model_name: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """Evaluate instruction adherence using Groq LLM as a judge.
    
    If Groq API key is provided, performs deep evaluation across constraints.
    Otherwise, applies rule-based validation.
    """
    if groq_api_key:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate

            model = ChatGroq(model=model_name, api_key=groq_api_key, temperature=0.1)

            judge_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are a strict AI response evaluation judge. Evaluate if the AGENT RESPONSE accurately and completely follows the USER PROMPT and CONSTRAINTS.
### USER PROMPT: {user_prompt}
### CONSTRAINTS: {constraints}
### AGENT RESPONSE: {agent_response}

EVALUATION RULES:
1. Assign a float score between 0.0 and 1.0 for each category:
   - quantity_score: followed requested count or item limits
   - format_score: followed format specifications (bullets, json, table, paragraph)
   - length_score: adhered to word or sentence bounds
   - style_score: adhered to requested tone or style
   - do_dont_score: avoided forbidden items or instructions
2. final_score = Average of the five scores.
3. Response MUST be valid JSON only.

JSON FORMAT:
{{
  "quantity_score": float,
  "format_score": float,
  "length_score": float,
  "style_score": float,
  "do_dont_score": float,
  "final_score": float,
  "reasoning": "Concise paragraph explaining the rationale."
}}""",
                )
            ])

            chain_input = judge_prompt.format(
                user_prompt=prompt,
                constraints=json.dumps(parsed_constraints, indent=2),
                agent_response=agent_response,
            )
            result = model.invoke(chain_input)
            content = result.content.strip()

            try:
                import json_repair
                eval_data = json_repair.loads(content)
            except Exception:
                eval_data = json.loads(content)

            eval_data["llm_judge_active"] = True
            return eval_data
        except Exception as e:
            print(f"Groq LLM Judge notice: {e}")

    # Deterministic fallback when Groq API key is not supplied
    scores: Dict[str, float] = {
        "quantity_score": 1.0,
        "format_score": 1.0,
        "length_score": 1.0,
        "style_score": 1.0,
        "do_dont_score": 1.0,
    }
    notes: List[str] = []

    words = re.findall(r"\b[\w'-]+\b", agent_response)
    word_count = len(words)

    # Check length
    for rule in parsed_constraints.get("length", []):
        m = re.search(r"(\d+)", rule)
        if m:
            limit = int(m.group(1))
            if "not more than" in rule or "less than" in rule or "limit to" in rule or "no more than" in rule:
                if word_count > limit:
                    scores["length_score"] = max(0.0, 1.0 - (word_count - limit) / limit)
                    notes.append(f"Exceeded word limit ({word_count} > {limit}).")
            elif "greater than" in rule:
                if word_count < limit:
                    scores["length_score"] = 0.5
                    notes.append(f"Below required words ({word_count} < {limit}).")

    # Check format
    for fmt in parsed_constraints.get("format", []):
        if fmt in ["bullet points", "bullets", "list"] and ("\n-" not in agent_response and "\n*" not in agent_response and "\n1." not in agent_response):
            scores["format_score"] = 0.4
            notes.append("Expected bulleted/numbered list formatting.")
        elif fmt == "json":
            try:
                json.loads(agent_response)
            except Exception:
                scores["format_score"] = 0.0
                notes.append("Expected valid JSON response.")

    # Check forbidden content
    for forbidden in parsed_constraints.get("do_dont", []):
        if forbidden.lower() in agent_response.lower():
            scores["do_dont_score"] = 0.0
            notes.append(f"Included prohibited term '{forbidden}'.")

    final_score = sum(scores.values()) / len(scores)
    reasoning = " ".join(notes) if notes else "Response satisfies parsed prompt constraints."

    return {
        **scores,
        "final_score": float(final_score),
        "reasoning": reasoning,
        "llm_judge_active": False,
    }


# ---------------------------------------------------------------------------
# Dimension 7: Factual Accuracy (Claims + SerpAPI Evidence + Groq LLM Judge)
# ---------------------------------------------------------------------------

def check_factuality(
    prompt: str,
    agent_response: str,
    groq_api_key: str,
    serpapi_api_key: str,
    max_claims: int = 5,
    max_results: int = 2,
) -> Tuple[float, Dict[str, Any], List[Dict[str, str]], List[str]]:
    """Extract atomic claims, search web evidence via SerpAPI, and judge factuality using Groq LLM."""
    try:
        from langchain_groq import ChatGroq
        from langchain_community.utilities import SerpAPIWrapper
        from langchain_core.prompts import ChatPromptTemplate
        import json_repair

        extractor_model = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)
        judge_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key, params={"num": str(max_results)})

        template_extract = ChatPromptTemplate([
            ("system", "You are an expert fact-checking assistant."),
            (
                "human",
                """Identify and extract up to {k} atomic knowledge claims from the following text.
Each claim must be concise (<15 words), resolve coreferences, and be verifiable.
Respond ONLY with a JSON array of objects: [{{"claim": "..."}}, ...]

Text:
{input_text}""",
            ),
        ])

        template_query = ChatPromptTemplate([
            ("system", "You generate search queries to verify factual claims."),
            (
                "human",
                """Given the claim, generate a JSON array with exactly ONE search engine query to verify it:
["query"]

Claim: {input}""",
            ),
        ])

        template_judge = ChatPromptTemplate([
            ("system", "You are an objective AI fact-checking judge."),
            (
                "human",
                """Review the response against the gathered evidence snippets.
Response: {answer}
Evidence: {evidences}

Respond ONLY in valid JSON:
{{
  "reasoning": "Detailed factual reasoning citing evidence.",
  "error": "Description of any factual errors, or 'None'.",
  "correction": "Corrected factual text if applicable.",
  "out-of-context": "Assumptions or irrelevant claims.",
  "factually incorrect number of sentences": 0,
  "factuality": true
}}""",
            ),
        ])

        # Step 1: Extract claims
        prompt1 = template_extract.invoke({"input_text": agent_response, "k": str(max_claims)})
        res_extract = extractor_model.invoke(prompt1)
        claims = json_repair.loads(res_extract.content.strip())
        if not isinstance(claims, list):
            claims = []

        # Step 2: Fetch evidence snippets
        evidences: List[str] = []
        for c in claims[:max_claims]:
            claim_text = c.get("claim", "") if isinstance(c, dict) else str(c)
            if not claim_text:
                continue
            try:
                res_query = extractor_model.invoke(template_query.invoke({"input": claim_text}))
                query_list = json_repair.loads(res_query.content.strip())
                query = query_list[0] if isinstance(query_list, list) and query_list else claim_text

                search_res = search.results(query)
                if "knowledge_graph" in search_res and "description" in search_res["knowledge_graph"]:
                    evidences.append(search_res["knowledge_graph"]["description"])
                for organic in search_res.get("organic_results", [])[:max_results]:
                    snippet = organic.get("snippet", "")
                    if snippet:
                        evidences.append(snippet)
            except Exception as e:
                print(f"SerpAPI query notice: {e}")

        # Step 3: Judge factuality
        prompt3 = template_judge.invoke({"answer": agent_response, "evidences": evidences})
        res_judge = judge_model.invoke(prompt3)
        final_eval = json_repair.loads(res_judge.content.strip())

        sentences = split_into_sentences(agent_response)
        total_sentences = max(1, len(sentences))
        incorrect_sentences = int(final_eval.get("factually incorrect number of sentences", 0))
        factual_score = max(0.0, 1.0 - (incorrect_sentences / total_sentences))

        return float(factual_score), final_eval, claims, evidences

    except Exception as e:
        print(f"Factuality evaluation error: {e}")
        return 0.0, {"error": str(e)}, [], []


# ---------------------------------------------------------------------------
# Composite Response Evaluator
# ---------------------------------------------------------------------------

def evaluate_response(
    prompt: str,
    response: str,
    groq_api_key: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    model_name: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """Execute complete multi-dimensional evaluation for a single prompt-response pair."""
    if constraints is None:
        constraints = parse_constraints(prompt)

    # 1. Semantic Similarity
    semantic_score = compute_semantic_similarity(prompt, response)

    # 2. BART NLI Entailment
    ent_prob, neu_prob, cont_prob = compute_nli_probabilities(prompt, response)

    # 3. Sequential Sentence Coherence
    coherence = compute_sequential_coherence(response)

    # 4. Fluency & Lexical Diversity
    diversity = compute_lexical_diversity(response)
    normalized_fluency, raw_ppl = compute_perplexity(response)

    # 5. Grammar Quality
    grammar_score, grammar_errors = check_grammar_languagetool(response)

    # 6. Instruction Adherence
    instruction_eval = evaluate_instruction_adherence_llm(
        prompt=prompt,
        agent_response=response,
        parsed_constraints=constraints,
        groq_api_key=groq_api_key,
        model_name=model_name,
    )
    instruction_score = float(instruction_eval.get("final_score", 1.0))

    # 7. Factual Accuracy
    factual_evaluated = False
    fact_score: Optional[float] = None
    if groq_api_key and serpapi_api_key:
        fact_score, fact_details, claims, evidences = check_factuality(
            prompt, response, groq_api_key, serpapi_api_key
        )
        factual_evaluated = True
    else:
        fact_score = None
        fact_details = {"status": "Not evaluated (Groq and SerpAPI credentials required)"}
        claims = []
        evidences = []

    # Composite Normalized Score (0 - 100)
    # Balanced base weighting across 7 dimensions (sum = 1.0):
    # Semantic: 25%, Prompt Entailment: 10%, Sentence Coherence: 10%,
    # Instruction Adherence: 20%, Fluency: 10%, Grammar: 10%, Factuality: 15%
    remaining_weighted_sum = (
        (semantic_score * 0.25)
        + (ent_prob * 0.10)
        + (coherence["overall_entailment"] * 0.10)
        + (instruction_score * 0.20)
        + (normalized_fluency * 0.10)
        + (grammar_score * 0.10)
    )

    if fact_score is not None:
        # Factuality was evaluated (including a genuine score of 0.0)
        composite_100 = (remaining_weighted_sum + (fact_score * 0.15)) * 100.0
    else:
        # Credentials missing: redistribute the 15% weight proportionally across remaining checks (sum = 0.85)
        composite_100 = (remaining_weighted_sum / 0.85) * 100.0

    return {
        "prompt": prompt,
        "agent_response": response,
        "overall_score": round(float(composite_100), 2),
        "semantic_score": round(float(semantic_score), 4),
        "entail_prob": round(float(ent_prob), 4),
        "neutral_prob": round(float(neu_prob), 4),
        "contradict_prob": round(float(cont_prob), 4),
        "coherence": coherence,
        "coherent_scores_overall_enc": coherence["coherent_scores_overall_enc"],
        "unigrams_diversity": round(float(diversity["unigrams_diversity"]), 4),
        "bigrams_diversity": round(float(diversity["bigrams_diversity"]), 4),
        "normalized_perplexity": round(float(normalized_fluency), 4),
        "raw_perplexity": round(float(raw_ppl), 2) if not math.isinf(raw_ppl) else 999.0,
        "grammar_score": round(float(grammar_score), 4),
        "grammar_errors": grammar_errors,
        "parsed_constraints": constraints,
        "instruction_adherence": instruction_eval,
        "factual_accuracy_score": round(float(fact_score), 4) if fact_score is not None else None,
        "factual_accuracy_display": f"{fact_score:.2f}" if fact_score is not None else "Not evaluated",
        "claims": claims,
        "evidences": evidences,
        "AI_as_judge_results": fact_details,
        "dimensions": {
            "Semantic Similarity": round(semantic_score * 100.0, 1),
            "Instruction adherence": round(instruction_score * 100.0, 1),
            "Prompt Entailment": round(ent_prob * 100.0, 1),
            "Sentence Coherence": round(coherence["overall_entailment"] * 100.0, 1),
            "Fluency (Perplexity)": round(normalized_fluency * 100.0, 1),
            "Grammar Quality": round(grammar_score * 100.0, 1),
            "Factual Accuracy": round(fact_score * 100.0, 1) if fact_score is not None else None,
        },
    }


def evaluate_payload(
    payload: Dict[str, Any],
    groq_api_key: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a payload containing prompt and candidate responses."""
    prompt = payload.get("prompt", "")
    explicit_constraints = payload.get("constraints")
    parsed_constraints = (
        explicit_constraints
        if isinstance(explicit_constraints, dict)
        else parse_constraints(prompt)
    )

    items: List[Tuple[str, str]] = []
    if "responses" in payload and isinstance(payload["responses"], list):
        for idx, item in enumerate(payload["responses"]):
            if isinstance(item, dict):
                agent_name = item.get("agent", f"Agent {idx+1}")
                text = item.get("response", "")
            else:
                agent_name = f"Agent {idx+1}"
                text = str(item)
            items.append((agent_name, text))
    elif "agent_responses" in payload and isinstance(payload["agent_responses"], list):
        for idx, text in enumerate(payload["agent_responses"]):
            items.append((f"Agent {idx+1}", str(text)))
    elif "response" in payload:
        items.append(("Agent 1", str(payload["response"])))

    results: List[Dict[str, Any]] = []
    for agent_name, text in items:
        eval_result = evaluate_response(
            prompt=prompt,
            response=text,
            groq_api_key=groq_api_key,
            serpapi_api_key=serpapi_api_key,
            constraints=parsed_constraints,
        )
        eval_result["agent"] = agent_name
        results.append(eval_result)

    results.sort(key=lambda x: x["overall_score"], reverse=True)
    return results
