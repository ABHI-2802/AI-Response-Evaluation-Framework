import json
import os
import sys
import unittest
from pathlib import Path

# Add project root to path so tests can run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import evaluate_response, evaluate_payload, sentence_continuity, surface_fluency


class EvaluatorTests(unittest.TestCase):
    def test_instruction_checks_and_scores_are_bounded(self):
        result = evaluate_response(
            prompt="Write a short bullet list about reliability.",
            response="- Reliability matters.",
            constraints={"must_include": ["reliability"], "expected_format": "bullets"},
        )
        self.assertEqual(result["dimensions"]["Instruction adherence"], 100.0)
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 100.0)

    def test_missing_required_term_fails(self):
        result = evaluate_response(
            prompt="Explain evaluation.",
            response="It can compare responses.",
            constraints={"must_include": ["reliability"]},
        )
        self.assertEqual(result["dimensions"]["Instruction adherence"], 0.0)

    def test_string_constraint_inputs_handled_safely(self):
        # Single string constraint instead of list should not iterate characters
        result = evaluate_response(
            prompt="Testing constraints",
            response="Model reliability is essential.",
            constraints={"must_include": "reliability", "must_not_include": "fail", "max_words": "50"},
        )
        self.assertEqual(result["dimensions"]["Instruction adherence"], 100.0)

    def test_empty_response(self):
        result = evaluate_response(
            prompt="Testing empty response",
            response="",
        )
        self.assertEqual(result["overall_score"], 0.0)
        self.assertEqual(sentence_continuity(""), 0.0)
        self.assertEqual(surface_fluency(""), 0.0)

    def test_payload_sorts_by_overall_score(self):
        payload = {
            "prompt": "Explain reliability and response evaluation.",
            "constraints": {"must_include": ["reliability"]},
            "responses": [
                {"agent": "Lower", "response": "A response."},
                {"agent": "Higher", "response": "Reliability is important for response evaluation."}
            ]
        }
        results = evaluate_payload(payload)
        self.assertGreaterEqual(results[0]["overall_score"], results[1]["overall_score"])

    def test_agent_responses_format_support(self):
        payload = {
            "prompt": "What are assumptions of linear regression?",
            "agent_responses": [
                "Assumption of linearity and normality of errors.",
                "Line relationship between variables."
            ]
        }
        results = evaluate_payload(payload)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["agent"].startswith("Agent"))

    def test_workspace_input_json_evaluates_cleanly(self):
        input_path = Path(__file__).resolve().parent.parent / "input.json"
        if input_path.exists():
            with open(input_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            results = evaluate_payload(payload)
            self.assertGreater(len(results), 0)
            self.assertIn("overall_score", results[0])


if __name__ == "__main__":
    unittest.main()
