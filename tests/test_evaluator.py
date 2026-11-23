import json
import os
import sys
import unittest
from pathlib import Path

# Add project root to path so tests can run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import (
    compute_lexical_diversity,
    compute_nli_probabilities,
    compute_perplexity,
    compute_semantic_similarity,
    compute_sequential_coherence,
    evaluate_instruction_adherence_llm,
    evaluate_payload,
    evaluate_response,
    parse_constraints,
    split_into_sentences,
)


class EvaluatorPipelineTests(unittest.TestCase):
    def test_parse_constraints(self):
        prompt = (
            "List 5 bullet points with a formal tone. "
            "Limit to 50 words. Do not mention competitors."
        )
        constraints = parse_constraints(prompt)
        self.assertIn(5, constraints["quantity"])
        self.assertIn("bullet points", constraints["format"])
        self.assertTrue(any("50 words" in l for l in constraints["length"]))
        self.assertIn("formal tone", constraints["style"])
        self.assertIn("competitors", constraints["do_dont"])

    def test_lexical_diversity(self):
        repetitive = "the the the the"
        diverse = "machine learning models evaluate natural language"
        div_rep = compute_lexical_diversity(repetitive)
        div_div = compute_lexical_diversity(diverse)
        self.assertLess(div_rep["unigrams_diversity"], div_div["unigrams_diversity"])
        self.assertGreaterEqual(div_div["bigrams_diversity"], 0.0)

    def test_split_into_sentences(self):
        text = "First sentence here. Second sentence follows! Third one?"
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 3)

    def test_semantic_similarity_bounds(self):
        score = compute_semantic_similarity(
            "What are assumptions of linear regression?",
            "Linear regression assumes linearity, normality, and homoscedasticity.",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_nli_probabilities_bounds(self):
        ent, neu, cont = compute_nli_probabilities(
            "The sky is blue.", "It is clear and blue outside."
        )
        self.assertGreaterEqual(ent, 0.0)
        self.assertGreaterEqual(neu, 0.0)
        self.assertGreaterEqual(cont, 0.0)
        self.assertAlmostEqual(ent + neu + cont, 1.0, delta=0.05)

    def test_sequential_coherence(self):
        text = "Linear regression models data. It assumes normal residuals. Errors are independent."
        coherence = compute_sequential_coherence(text)
        self.assertEqual(len(coherence["pairs"]), 2)
        self.assertIn("overall_entailment", coherence)
        self.assertIn("overall_contradiction", coherence)

    def test_fluency_and_perplexity_bounds(self):
        score, ppl = compute_perplexity("This is a fluent and grammatical sentence.")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(ppl, 0.0)

    def test_instruction_adherence_fallback(self):
        prompt = "Limit to 20 words. Do not mention error."
        response = "This short response follows the prompt."
        constraints = parse_constraints(prompt)
        res = evaluate_instruction_adherence_llm(
            prompt=prompt, agent_response=response, parsed_constraints=constraints
        )
        self.assertIn("final_score", res)
        self.assertGreaterEqual(res["final_score"], 0.0)
        self.assertLessEqual(res["final_score"], 1.0)

    def test_single_response_evaluation(self):
        result = evaluate_response(
            prompt="What is linear regression?",
            response="Linear regression is a statistical method for modeling linear relationships.",
        )
        self.assertIn("overall_score", result)
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 100.0)
        self.assertIn("dimensions", result)
        self.assertIn("Semantic Similarity", result["dimensions"])
        self.assertIn("Sentence Coherence", result["dimensions"])

    def test_missing_credentials_marks_factuality_unavailable_and_redistributes_weight(self):
        prompt = "What is linear regression?"
        response = "Linear regression models linear relationships between variables."

        # Missing both keys
        result_missing = evaluate_response(
            prompt=prompt,
            response=response,
            groq_api_key=None,
            serpapi_api_key=None,
        )

        # 1. Factuality must be marked as unavailable
        self.assertIsNone(result_missing["factual_accuracy_score"])
        self.assertEqual(result_missing["factual_accuracy_display"], "Not evaluated")
        self.assertIsNone(result_missing["dimensions"]["Factual Accuracy"])

        # 2. 15% weight must be redistributed across remaining 85% checks
        sem = result_missing["semantic_score"]
        ent = result_missing["entail_prob"]
        coh = result_missing["coherence"]["overall_entailment"]
        inst = result_missing["instruction_adherence"]["final_score"]
        fl = result_missing["normalized_perplexity"]
        gram = result_missing["grammar_score"]

        remaining_weighted_sum = (
            (sem * 0.25)
            + (ent * 0.10)
            + (coh * 0.10)
            + (inst * 0.20)
            + (fl * 0.10)
            + (gram * 0.10)
        )
        expected_redistributed_score = round(float((remaining_weighted_sum / 0.85) * 100.0), 2)
        self.assertEqual(result_missing["overall_score"], expected_redistributed_score)

    def test_genuine_factuality_score_zero_must_count(self):
        from unittest.mock import patch

        prompt = "What is linear regression?"
        response = "Linear regression plane turns into a pumpkin at midnight."

        mock_eval = {
            "reasoning": "The response makes completely false and hallucinated claims.",
            "error": "Pumpkin claim is fabricated.",
            "factuality": False,
            "factually incorrect number of sentences": 1,
        }

        with patch("evaluator.check_factuality", return_value=(0.0, mock_eval, [{"claim": "pumpkin"}], ["evidence"])):
            result_zero = evaluate_response(
                prompt=prompt,
                response=response,
                groq_api_key="mock_groq_key",
                serpapi_api_key="mock_serpapi_key",
            )

        # 1. Factuality score 0.0 must be preserved and distinct from None
        self.assertIsNotNone(result_zero["factual_accuracy_score"])
        self.assertEqual(result_zero["factual_accuracy_score"], 0.0)
        self.assertEqual(result_zero["dimensions"]["Factual Accuracy"], 0.0)
        self.assertEqual(result_zero["factual_accuracy_display"], "0.00")

        # 2. Score must count with the 15% weight included (not redistributed)
        sem = result_zero["semantic_score"]
        ent = result_zero["entail_prob"]
        coh = result_zero["coherence"]["overall_entailment"]
        inst = result_zero["instruction_adherence"]["final_score"]
        fl = result_zero["normalized_perplexity"]
        gram = result_zero["grammar_score"]

        remaining_weighted_sum = (
            (sem * 0.25)
            + (ent * 0.10)
            + (coh * 0.10)
            + (inst * 0.20)
            + (fl * 0.10)
            + (gram * 0.10)
        )
        # With factuality = 0.0, it contributes 0.0 * 0.15 = 0.0, and overall score is based on sum = 1.0 (not divided by 0.85)
        expected_score_with_zero = round(float(remaining_weighted_sum + (0.0 * 0.15)) * 100.0, 2)
        self.assertEqual(result_zero["overall_score"], expected_score_with_zero)

        # 3. Overall score with factuality=0.0 MUST be strictly lower than when factuality was unavailable
        expected_score_if_redistributed = round(float((remaining_weighted_sum / 0.85) * 100.0), 2)
        self.assertLess(result_zero["overall_score"], expected_score_if_redistributed)

    def test_evaluate_payload_with_input_json(self):
        input_path = Path(__file__).resolve().parent.parent / "input.json"
        if input_path.exists():
            with open(input_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            results = evaluate_payload(payload)
            self.assertEqual(len(results), len(payload["agent_responses"]))
            self.assertGreaterEqual(results[0]["overall_score"], results[-1]["overall_score"])


if __name__ == "__main__":
    unittest.main()
