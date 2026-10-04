import json
import pytest
from src.llm_client import LLMClient
from src.pipeline import OptimizedPipeline


def test_classifier_mock_output_parsing():
    client = LLMClient()
    pipeline = OptimizedPipeline(client)
    
    test_msg = "My ₹140 Bluetooth speaker broke after 3 days. I am extremely pissed off! Order #APX-9821."
    res = pipeline.run(test_msg)
    
    customer_state = res["customer_state"]
    assert isinstance(customer_state, dict)
    assert customer_state["intent"] in ["REFUND_REQUEST", "DAMAGED_GOODS", "RETURN_POLICY", "BILLING_DISPUTE", "GENERAL_INQUIRY"]
    assert customer_state["sentiment"] in ["FRUSTRATED", "NEUTRAL", "SATISFIED"]
    assert customer_state["urgency"] in ["HIGH", "MEDIUM", "LOW"]
    assert customer_state["key_entities"]["order_id"] == "APX-9821"
    assert customer_state["key_entities"]["monetary_amount"] == 140.0


def test_classifier_damaged_goods():
    client = LLMClient()
    pipeline = OptimizedPipeline(client)

    test_msg = "The glass coffee pot in order APX-1029 arrived completely shattered!"
    res = pipeline.run(test_msg)

    state = res["customer_state"]
    assert state["intent"] in ["DAMAGED_GOODS", "REFUND_REQUEST"]
    assert state["key_entities"]["order_id"] == "APX-1029"
