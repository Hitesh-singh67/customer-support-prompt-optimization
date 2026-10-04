import pytest
from src.llm_client import LLMClient
from src.evaluator import BenchmarkEvaluator


def test_evaluator_judge_scoring():
    client = LLMClient()
    evaluator = BenchmarkEvaluator(client)

    cust_msg = "My ₹140 speaker broke."
    resp = "I am so sorry to hear that! I have authorized a full refund of ₹140.00. Use GOODWILL10 code."
    scores = evaluator.evaluate_response_with_judge(cust_msg, resp)

    assert "empathy_score" in scores
    assert "actionability_score" in scores
    assert "policy_compliance_score" in scores
    assert "conciseness_score" in scores
    assert "overall_csat_score" in scores
    assert 1.0 <= scores["overall_csat_score"] <= 5.0


def test_evaluator_dataset_loading():
    client = LLMClient()
    evaluator = BenchmarkEvaluator(client)
    dataset = evaluator.load_dataset()
    
    assert isinstance(dataset, list)
    assert len(dataset) == 50
    assert "id" in dataset[0]
    assert "customer_message" in dataset[0]
