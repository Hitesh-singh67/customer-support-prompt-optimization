import json
import time
from typing import Dict, Any, Optional
from src.config import (
    BASELINE_PROMPT_PATH,
    CLASSIFIER_PROMPT_PATH,
    OPTIMIZED_PROMPT_PATH,
    INPUT_TOKEN_COST_PER_1K,
    OUTPUT_TOKEN_COST_PER_1K,
    load_prompt_template,
)
from src.llm_client import LLMClient
from src.guardrails import validate_response, GuardrailResult


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_TOKEN_COST_PER_1K / 1000.0) + (output_tokens * OUTPUT_TOKEN_COST_PER_1K / 1000.0)


class BaselinePipeline:
    """
    Naive single-stage assistant pipeline using unoptimized baseline prompt.
    """
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self.prompt_template = load_prompt_template(BASELINE_PROMPT_PATH)

    def run(self, customer_input: str) -> Dict[str, Any]:
        start_time = time.time()
        full_prompt = f"{self.prompt_template}\n\nCustomer Input:\n\"{customer_input}\""
        
        result = self.client.generate(
            prompt=full_prompt,
            mode_context="baseline",
            temperature=0.7,
            max_output_tokens=300
        )
        
        elapsed_ms = result.get("latency_ms", int((time.time() - start_time) * 1000))
        in_tokens = result.get("input_tokens", len(full_prompt) // 4)
        out_tokens = result.get("output_tokens", len(result["content"]) // 4)
        cost = calculate_cost(in_tokens, out_tokens)

        return {
            "pipeline_type": "Baseline",
            "response": result["content"].strip(),
            "latency_ms": elapsed_ms,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "token_cost": round(cost, 6),
            "is_mock": result.get("is_mock", False),
            "prompts_used": {
                "system_prompt": self.prompt_template,
                "full_prompt": full_prompt
            }
        }


class OptimizedPipeline:
    """
    Multi-stage prompt engineered pipeline featuring:
    1. Intent/Sentiment/Urgency Classifier
    2. Dynamic XML-Structured Context Injection Generation
    3. Rule-based & Prompt-based Policy Guardrails
    """
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self.classifier_template = load_prompt_template(CLASSIFIER_PROMPT_PATH)
        self.optimized_template = load_prompt_template(OPTIMIZED_PROMPT_PATH)

    def run(self, customer_input: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # --- Stage 1: Classifier ---
        stage1_start = time.time()
        classifier_prompt = self.classifier_template.replace("{customer_input}", customer_input)
        classifier_res = self.client.generate(
            prompt=classifier_prompt,
            mode_context="classifier",
            temperature=0.0
        )
        classifier_ms = int((time.time() - stage1_start) * 1000)
        
        # Parse Classifier JSON
        try:
            # Clean markdown formatting if present
            raw_cls = classifier_res["content"].strip()
            if raw_cls.startswith("```json"):
                raw_cls = raw_cls.split("```json")[1].split("```")[0].strip()
            elif raw_cls.startswith("```"):
                raw_cls = raw_cls.split("```")[1].split("```")[0].strip()
                
            customer_state = json.loads(raw_cls)
        except Exception as e:
            print(f"[OptimizedPipeline] Classifier JSON parse error ({e}). Using default state.")
            customer_state = {
                "intent": "GENERAL_INQUIRY",
                "sentiment": "NEUTRAL",
                "urgency": "MEDIUM",
                "key_entities": {"order_id": None, "monetary_amount": None},
                "customer_summary": customer_input[:100]
            }

        # --- Stage 2: Dynamic Context Injection Generation ---
        stage2_start = time.time()
        intent = customer_state.get("intent", "GENERAL_INQUIRY")
        sentiment = customer_state.get("sentiment", "NEUTRAL")
        urgency = customer_state.get("urgency", "MEDIUM")
        key_entities = json.dumps(customer_state.get("key_entities", {}))
        customer_summary = customer_state.get("customer_summary", "")

        generation_prompt = (
            self.optimized_template
            .replace("{intent}", str(intent))
            .replace("{sentiment}", str(sentiment))
            .replace("{urgency}", str(urgency))
            .replace("{key_entities}", key_entities)
            .replace("{customer_summary}", str(customer_summary))
            .replace("{customer_input}", customer_input)
        )

        gen_res = self.client.generate(
            prompt=generation_prompt,
            mode_context="optimized",
            temperature=0.2,
            max_output_tokens=500
        )
        candidate_response = gen_res["content"].strip()
        generation_ms = int((time.time() - stage2_start) * 1000)

        # --- Stage 3: Guardrail Verification ---
        stage3_start = time.time()
        guardrail_res: GuardrailResult = validate_response(
            client=self.client,
            customer_input=customer_input,
            candidate_response=candidate_response,
            customer_state=customer_state
        )
        guardrail_ms = int((time.time() - stage3_start) * 1000)

        final_response = guardrail_res.sanitized_response
        total_latency = int((time.time() - start_time) * 1000)

        # Token metrics aggregation
        total_in_tokens = (
            classifier_res.get("input_tokens", 0) +
            gen_res.get("input_tokens", 0) +
            (len(candidate_response) // 4)
        )
        total_out_tokens = (
            classifier_res.get("output_tokens", 0) +
            gen_res.get("output_tokens", 0) +
            (len(final_response) // 4)
        )
        total_cost = calculate_cost(total_in_tokens, total_out_tokens)

        return {
            "pipeline_type": "Optimized",
            "response": final_response,
            "raw_candidate_response": candidate_response,
            "customer_state": customer_state,
            "guardrail_result": guardrail_res.model_dump(),
            "stage_latencies": {
                "classifier_ms": classifier_ms,
                "generation_ms": generation_ms,
                "guardrail_ms": guardrail_ms
            },
            "latency_ms": total_latency,
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
            "token_cost": round(total_cost, 6),
            "is_mock": gen_res.get("is_mock", False),
            "prompts_used": {
                "classifier_prompt": classifier_prompt,
                "generation_prompt": generation_prompt
            }
        }
