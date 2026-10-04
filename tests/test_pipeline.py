import pytest
from src.llm_client import LLMClient
from src.pipeline import BaselinePipeline, OptimizedPipeline


def test_baseline_pipeline_execution():
    client = LLMClient()
    pipeline = BaselinePipeline(client)
    res = pipeline.run("What is your return policy?")
    
    assert res["pipeline_type"] == "Baseline"
    assert isinstance(res["response"], str)
    assert len(res["response"]) > 0
    assert res["latency_ms"] >= 0
    assert res["input_tokens"] > 0
    assert res["output_tokens"] > 0
    assert res["token_cost"] >= 0.0


def test_optimized_pipeline_execution():
    client = LLMClient()
    pipeline = OptimizedPipeline(client)
    res = pipeline.run("My ₹140 Bluetooth speaker broke after 3 days. Order #APX-9821.")
    
    assert res["pipeline_type"] == "Optimized"
    assert isinstance(res["response"], str)
    assert "customer_state" in res
    assert "guardrail_result" in res
    assert "stage_latencies" in res
    assert res["latency_ms"] >= 0
    assert res["input_tokens"] > 0
    assert res["output_tokens"] > 0
