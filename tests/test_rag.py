import importlib.util
import unittest
from pathlib import Path

from logistics_ai.rag import (
    DEFAULT_CORPUS,
    DEFAULT_DEV,
    DEFAULT_EVAL,
    Retriever,
    answer_question,
    build_answer,
    calibrate_abstention,
    load_documents,
    read_jsonl,
    reciprocal_rank_fusion,
    validate_questions,
)


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents(DEFAULT_CORPUS)
        cls.retriever = Retriever(cls.documents)

    def test_frozen_splits_are_valid_and_disjoint(self):
        validate_questions(read_jsonl(DEFAULT_DEV), read_jsonl(DEFAULT_EVAL), self.documents)

    def test_calibration_rejects_eval_rows(self):
        with self.assertRaisesRegex(ValueError, "development"):
            calibrate_abstention(self.retriever, read_jsonl(DEFAULT_EVAL))

    def test_exact_policy_subject_retrieves_right_document(self):
        hits = self.retriever.search("냉장 화물 온도 이탈 대응", k=3)
        self.assertEqual(hits[0].document.doc_id, "SYN-OPS-001")
        self.assertGreater(hits[0].score, 0)

    def test_unknown_words_lower_coverage(self):
        baseline = self.retriever.lexical.coverage("냉장 화물", 0)
        unsupported = self.retriever.lexical.coverage("냉장 화물 보험료 환불 수수료", 0)
        self.assertGreater(baseline, unsupported)

    def test_rrf_combines_rank_not_incompatible_scores(self):
        scores = reciprocal_rank_fusion([[0, 1], [1, 2]], constant=60)
        self.assertAlmostEqual(scores[1], 1 / 62 + 1 / 61)
        self.assertGreater(scores[1], scores[0])

    def test_provenance_preserves_document_id_version_and_verbatim_text(self):
        hits = self.retriever.search("냉장 화물 온도 이탈 대응")
        answer = answer_question("냉장 화물 온도 이탈 대응", hits, threshold=0)
        self.assertFalse(answer["abstained"])
        self.assertEqual(answer["evidence"][0]["citation"], "[SYN-OPS-001@1.0]")
        self.assertEqual(answer["evidence"][0]["text"], self.documents[0].text)
        self.assertIsNone(answer["model_draft"])

    def test_real_company_question_refused_even_with_strong_match(self):
        query = "CJ올리브네트웍스의 실제 냉장 화물 온도 이탈 대응"
        answer = answer_question(query, self.retriever.search(query), threshold=0)
        self.assertTrue(answer["abstained"])
        self.assertEqual(answer["reason"], "actual_company_policy_not_available")
        self.assertEqual(answer["evidence"], [])

    def test_abstention_does_not_call_generator(self):
        class MustNotGenerate:
            def generate(self, *args, **kwargs):
                raise AssertionError("Generator was called for refused query")

        answer = answer_question("알 수 없는 질문", [], threshold=0, generator=MustNotGenerate())
        self.assertTrue(answer["abstained"])

    def test_default_is_fail_closed_without_calibration(self):
        hits = self.retriever.search("냉장 화물 온도 이탈 대응")
        self.assertTrue(answer_question("냉장 화물 온도 이탈 대응", hits)["abstained"])

    def test_empty_query_and_invalid_k(self):
        self.assertEqual(self.retriever.search("   "), [])
        self.assertTrue(build_answer("   ")["abstained"])
        with self.assertRaises(ValueError):
            self.retriever.search("냉장", k=0)

    def test_unsupported_modes_and_duplicate_documents_fail(self):
        with self.assertRaises(ValueError):
            Retriever(self.documents, mode="fake_dense")
        with self.assertRaises(ValueError):
            Retriever([self.documents[0], self.documents[0]])

    def test_calibration_is_deterministic_and_uses_only_dev(self):
        dev = read_jsonl(DEFAULT_DEV)
        first = calibrate_abstention(self.retriever, dev)
        second = calibrate_abstention(self.retriever, dev)
        self.assertEqual(first, second)
        self.assertEqual(first["split"], "dev")
        self.assertGreaterEqual(first["threshold"], 0)
        self.assertLessEqual(first["threshold"], 1)

    def test_true_recall_counts_all_required_documents(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rag.py"
        spec = importlib.util.spec_from_file_location("evaluate_rag_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.retrieval_metrics(["a", "wrong", "b"], ["a", "b"])
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr@3"], 1.0)
        self.assertIsNone(module.retrieval_metrics(["a"], [])["recall@1"])


if __name__ == "__main__":
    unittest.main()
