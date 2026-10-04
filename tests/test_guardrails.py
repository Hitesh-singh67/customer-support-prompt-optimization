import pytest
from src.guardrails import check_rule_based_violations, validate_response
from src.llm_client import LLMClient


def test_guardrail_unauthorized_refund_interception():
    cust_input = "I bought a 4K Gaming Monitor for ₹650 (Order #APX-4410). Give me cash back right now!"
    cand_resp = "I have immediately refunded ₹650 back to your account cash back right now."
    cust_state = {
        "intent": "REFUND_REQUEST",
        "sentiment": "FRUSTRATED",
        "urgency": "HIGH",
        "key_entities": {"order_id": "APX-4410", "monetary_amount": 650.0}
    }
    
    result = check_rule_based_violations(cust_input, cand_resp, cust_state)
    assert not result.is_safe
    assert "UNAUTHORIZED_REFUND_PROMISE" in result.violations
    assert "SUP-ESCALATE" in result.sanitized_response
    assert "₹650.00" in result.sanitized_response


def test_guardrail_robotic_disclaimer_interception():
    cust_input = "Can I return a jacket after 10 days?"
    cand_resp = "As an AI language model, I can tell you that our return window is 30 days."
    cust_state = {"intent": "RETURN_POLICY", "key_entities": {}}

    result = check_rule_based_violations(cust_input, cand_resp, cust_state)
    assert not result.is_safe
    assert "ROBOTIC_DISCLAIMER" in result.violations
    assert "As an AI language model" not in result.sanitized_response


def test_guardrail_safe_response():
    cust_input = "I ordered headphones #APX-3390."
    cand_resp = "I am so sorry to hear your headphones arrived damaged. I have issued a full refund of ₹85.00."
    cust_state = {"intent": "REFUND_REQUEST", "key_entities": {"order_id": "APX-3390", "monetary_amount": 85.0}}

    result = check_rule_based_violations(cust_input, cand_resp, cust_state)
    assert result.is_safe
    assert len(result.violations) == 0
