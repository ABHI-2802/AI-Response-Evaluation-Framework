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
