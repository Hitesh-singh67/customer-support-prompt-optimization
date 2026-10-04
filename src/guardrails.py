import json
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.config import GUARDRAIL_PROMPT_PATH, load_prompt_template
from src.llm_client import LLMClient


class GuardrailResult(BaseModel):
    is_safe: bool = Field(description="Whether the candidate response complies with all safety & policy rules.")
    violations: List[str] = Field(default_factory=list, description="List of rule violation identifiers if any.")
    reasoning: str = Field(description="Audit explanation for the decision.")
    sanitized_response: str = Field(description="Policy-compliant version of response (or original if is_safe is True).")


def check_rule_based_violations(
    customer_input: str,
    candidate_response: str,
    customer_state: Dict[str, Any]
) -> GuardrailResult:
    """
    Fast rule-based guardrail checking key policy rules prior to or alongside LLM audit.
    """
    violations = []
    reasoning_parts = []
    
    cand_lower = candidate_response.lower()
    cust_lower = customer_input.lower()
    
    # Extract monetary amount if present
    entities = customer_state.get("key_entities", {})
    amount = entities.get("monetary_amount")
    if amount is None:
        price_match = re.search(r"(?:₹|\$)(\d+(?:\.\d+)?)", customer_input)
        if price_match:
            try:
                amount = float(price_match.group(1))
            except ValueError:
                amount = 0.0

    # Rule 1: Unauthorized refund promise for amounts > ₹200
    if amount and amount > 200:
        promises_instant = any(w in cand_lower for w in ["immediately refunded", "instant refund", "cash back right now", "processed your full refund of ₹", "processed your full refund of $"])
        mentions_escalation = any(w in cand_lower for w in ["supervisor", "escalated", "review", "sup-escalate"])
        
        if promises_instant and not mentions_escalation:
            violations.append("UNAUTHORIZED_REFUND_PROMISE")
            reasoning_parts.append(f"Response promises an instant full refund for ₹{amount:.2f} (> ₹200 limit) without required supervisor escalation.")

    # Rule 2: Robotic AI disclaimer
    if any(phrase in cand_lower for phrase in ["as an ai", "i am an ai", "i am just a chatbot", "computer program"]):
        violations.append("ROBOTIC_DISCLAIMER")
        reasoning_parts.append("Response contains forbidden robotic AI disclaimer phrasing.")

    # Rule 3: Hostile or rude language
    if any(phrase in cand_lower for phrase in ["stop complaining", "calm down", "not my problem", "too bad"]):
        violations.append("HOSTILE_OR_DISCOURTEOUS_TONE")
        reasoning_parts.append("Response uses discourteous or hostile language.")

    if violations:
        # Create sanitized response for refund violation if needed
        if "UNAUTHORIZED_REFUND_PROMISE" in violations:
            order_id = entities.get("order_id") or "your order"
            sanitized = (
                f"I am truly sorry for the frustration with {order_id}! Because refunds over ₹200 require supervisor verification, "
                f"I have flagged your request for priority supervisor review under ticket #SUP-ESCALATE. "
                f"A supervisor will review and finalize your ₹{amount:.2f} refund within 24 business hours."
            )
        else:
            sanitized = re.sub(r"(?i)as an ai (language model|assistant|chatbot)?", "", candidate_response).strip()
            
        return GuardrailResult(
            is_safe=False,
            violations=violations,
            reasoning="; ".join(reasoning_parts),
            sanitized_response=sanitized
        )

    return GuardrailResult(
        is_safe=True,
        violations=[],
        reasoning="Passed rule-based policy validation.",
        sanitized_response=candidate_response
    )


def validate_response(
    client: LLMClient,
    customer_input: str,
    candidate_response: str,
    customer_state: Dict[str, Any]
) -> GuardrailResult:
    """
    Executes guardrail evaluation using rule-based checks combined with LLM Guardrail audit.
    """
    # 1. Fast deterministic check
    rule_res = check_rule_based_violations(customer_input, candidate_response, customer_state)
    if not rule_res.is_safe:
        return rule_res

    # 2. LLM Audit Check
    try:
        guardrail_template = load_prompt_template(GUARDRAIL_PROMPT_PATH)
        prompt = (
            guardrail_template
            .replace("{customer_input}", customer_input)
            .replace("{candidate_response}", candidate_response)
            .replace("{customer_state}", json.dumps(customer_state, indent=2))
        )
        
        result = client.generate(prompt=prompt, mode_context="guardrail", temperature=0.0)
        parsed = json.loads(result["content"])
        
        is_safe = parsed.get("is_safe", True)
        violations = parsed.get("violations", [])
        reasoning = parsed.get("reasoning", "Passed LLM Guardrail audit.")
        sanitized = parsed.get("sanitized_response") or candidate_response
        
        if not is_safe and not sanitized:
            sanitized = rule_res.sanitized_response

        return GuardrailResult(
            is_safe=is_safe,
            violations=violations,
            reasoning=reasoning,
            sanitized_response=sanitized if not is_safe else candidate_response
        )
    except Exception as e:
        print(f"[Guardrails] LLM Guardrail call fallback due to: {e}")
        return rule_res
