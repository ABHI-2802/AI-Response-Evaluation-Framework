import unittest

from evaluator import evaluate_response, evaluate_payload


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


if __name__ == "__main__":
    unittest.main()
