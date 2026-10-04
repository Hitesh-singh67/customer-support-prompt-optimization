import json
import os
import re
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Try importing official google-genai package
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class LLMClient:
    """
    Unified LLM Client using Google Gemini API (`google-genai`).
    Supports seamless deterministic mock execution when GEMINI_API_KEY is not configured.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.is_mock = not bool(self.api_key and GENAI_AVAILABLE)
        
        if not self.is_mock and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[LLMClient] Warning: GenAI client initialization failed ({e}). Falling back to Mock Engine.")
                self.is_mock = True

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1000,
        mode_context: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate completion response. Returns dict with 'content', 'latency_ms', 'input_tokens', 'output_tokens'.
        """
        start_time = time.time()
        
        if self.is_mock:
            response_text = self._mock_generation(prompt, system_instruction, mode_context)
            latency_ms = int((time.time() - start_time + 0.08 + (len(prompt) % 15) / 100.0) * 1000)
            in_tokens = max(10, len(prompt) // 4)
            out_tokens = max(10, len(response_text) // 4)
            return {
                "content": response_text,
                "latency_ms": latency_ms,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "is_mock": True
            }

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if system_instruction:
                config.system_instruction = system_instruction
                
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=config,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract usage metadata if present
            usage = getattr(response, "usage_metadata", None)
            in_tokens = getattr(usage, "prompt_token_count", len(prompt) // 4) if usage else len(prompt) // 4
            out_tokens = getattr(usage, "candidates_token_count", len(response.text) // 4) if usage else len(response.text) // 4

            return {
                "content": response.text,
                "latency_ms": latency_ms,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "is_mock": False
            }
        except Exception as e:
            print(f"[LLMClient] API call failed ({e}). Falling back to Mock Engine for this request.")
            response_text = self._mock_generation(prompt, system_instruction, mode_context)
            latency_ms = int((time.time() - start_time + 0.1) * 1000)
            return {
                "content": response_text,
                "latency_ms": latency_ms,
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(response_text) // 4,
                "is_mock": True
            }

    def _mock_generation(self, prompt: str, system_instruction: Optional[str], mode_context: str) -> str:
        """
        High-fidelity deterministic mock response generator for classifier, baseline, optimized, guardrail, and judge calls.
        """
        prompt_lower = prompt.lower()
        sys_lower = (system_instruction or "").lower()

        # 1. Classifier Mock
        if "expected output schema" in prompt_lower or "customer support query classifier" in prompt_lower or mode_context == "classifier":
            intent = "GENERAL_INQUIRY"
            sentiment = "NEUTRAL"
            urgency = "LOW"
            amount = None
            order_id = None

            # Extract order ID if present
            order_match = re.search(r"APX-\d+", prompt, re.IGNORECASE)
            if order_match:
                order_id = order_match.group(0).upper()

            # Extract price if present
            price_match = re.search(r"(?:₹|\$)(\d+(?:\.\d+)?)", prompt)
            if price_match:
                try:
                    amount = float(price_match.group(1))
                except ValueError:
                    pass

            if "refund" in prompt_lower or "money back" in prompt_lower or "payout" in prompt_lower:
                intent = "REFUND_REQUEST"
            elif "damaged" in prompt_lower or "broken" in prompt_lower or "shattered" in prompt_lower or "crushed" in prompt_lower or "torn" in prompt_lower:
                intent = "DAMAGED_GOODS"
            elif "return" in prompt_lower or "exchange" in prompt_lower or "days ago" in prompt_lower or "clearance" in prompt_lower:
                intent = "RETURN_POLICY"
            elif "charged" in prompt_lower or "billing" in prompt_lower or "subscription" in prompt_lower or "invoice" in prompt_lower or "overcharged" in prompt_lower or "tax" in prompt_lower:
                intent = "BILLING_DISPUTE"
            elif "socks" in prompt_lower or "cat" in prompt_lower or "incompetent" in prompt_lower or "lost my shoes" in prompt_lower:
                intent = "GENERAL_INQUIRY"

            if any(w in prompt_lower for w in ["pissed", "furious", "upset", "ridiculous", "unacceptable", "sue", "demand", "angry", "idiots", "useless", "scam"]):
                sentiment = "FRUSTRATED"
            elif "thanks" in prompt_lower or "great" in prompt_lower:
                sentiment = "SATISFIED"

            if amount and amount > 200 or sentiment == "FRUSTRATED" or "immediately" in prompt_lower:
                urgency = "HIGH"
            elif intent in ["REFUND_REQUEST", "DAMAGED_GOODS", "BILLING_DISPUTE"]:
                urgency = "MEDIUM"

            return json.dumps({
                "intent": intent,
                "sentiment": sentiment,
                "urgency": urgency,
                "key_entities": {
                    "order_id": order_id,
                    "product_name": "Customer Product",
                    "monetary_amount": amount
                },
                "customer_summary": "Customer raised an inquiry regarding " + intent.replace("_", " ").lower()
            }, indent=2)

        # 2. Guardrail Mock
        if "guardrail" in prompt_lower or "audit a candidate chatbot response" in prompt_lower or mode_context == "guardrail":
            # Check if candidate response promises unauthorized refunds > ₹200 instantly
            is_unauthorized = False
            if "cash back right now" in prompt_lower or ("instant" in prompt_lower and any(sym in prompt_lower for sym in ["₹", "$"]) and "refund" in prompt_lower):
                # check if amount > 200
                amounts = [float(a) for a in re.findall(r"(?:₹|\$)(\d+)", prompt) if float(a) > 200]
                if amounts and ("guaranteed" in prompt_lower or "instant" in prompt_lower or "processed immediately" in prompt_lower):
                    is_unauthorized = True

            if is_unauthorized:
                return json.dumps({
                    "is_safe": False,
                    "violations": ["UNAUTHORIZED_REFUND_PROMISE"],
                    "reasoning": "Response promises instant cash refund exceeding ₹200 auto-approval threshold without supervisor escalation.",
                    "sanitized_response": "I am so sorry for the frustration. Because refunds over ₹200 require supervisor verification, I have flagged your order for priority supervisor review (#SUP-ESCALATE). A supervisor will finalize your refund within 24 business hours."
                }, indent=2)
            else:
                return json.dumps({
                    "is_safe": True,
                    "violations": [],
                    "reasoning": "Response adheres to company policy and safety guidelines.",
                    "sanitized_response": ""
                }, indent=2)

        # 3. LLM-as-a-Judge Mock Evaluator
        if "scoring rubric" in prompt_lower or "evaluating customer support chatbot responses" in prompt_lower or mode_context == "judge":
            bot_resp = prompt.split("Actual Chatbot Response:")[-1].lower() if "Actual Chatbot Response:" in prompt else prompt_lower
            is_optimized = any(k in bot_resp for k in ["goodwill10", "sup-escalate", "store credit", "authorized a full refund", "3-5 business days", "24 business hours", "priority supervisor review", "truly sorry to hear", "so sorry to hear"])
            is_frustrated = "pissed" in prompt_lower or "furious" in prompt_lower or "demand" in prompt_lower or "upset" in prompt_lower or "ridiculous" in prompt_lower
            
            if is_optimized:
                emp = 4.8 if is_frustrated else 4.6
                act = 4.7
                pol = 4.8
                con = 4.5
                summary = "The optimized response shows high empathy, clear step-by-step actionability, precise policy adherence, and concise structure."
            else:
                # Baseline
                emp = 2.7 if is_frustrated else 3.4
                act = 3.2
                pol = 3.6
                con = 3.8
                summary = "The baseline response is polite but generic, lacking emotional validation for frustrated tone and missing explicit policy escalation details."

            overall = round((emp * 0.30) + (act * 0.30) + (pol * 0.25) + (con * 0.15), 2)

            return json.dumps({
                "empathy_score": emp,
                "actionability_score": act,
                "policy_compliance_score": pol,
                "conciseness_score": con,
                "overall_csat_score": overall,
                "judge_rationale": summary
            }, indent=2)

        # 4. Baseline Pipeline Response Mock
        if mode_context == "baseline" or "answer the customer's inquiry politely based on general customer service" in sys_lower:
            if "pissed" in prompt_lower or "broken" in prompt_lower or "shattered" in prompt_lower or "furious" in prompt_lower:
                return "Hello. Thank you for reaching out to customer support. I apologize for the issue with your item. You can return the item according to our return policy on our website or contact our support team during business hours for assistance."
            elif "4k gaming monitor" in prompt_lower or "₹650" in prompt_lower or "$650" in prompt_lower or "sue" in prompt_lower:
                return "We apologize for the inconvenience with your monitor. Please visit our returns portal online to submit a return claim. Refunds will be processed according to standard processing times."
            elif "30 days" in prompt_lower or "42 days" in prompt_lower or "return" in prompt_lower:
                return "Our standard return policy allows returns within 30 days of purchase. Please check our terms and conditions on our website for more information."
            elif "charged" in prompt_lower or "double" in prompt_lower:
                return "I apologize for any billing confusion. Please check your bank account or send us your receipt so we can investigate."
            else:
                return "Hello, thank you for contacting customer support. We are happy to help you with your inquiry. Please let us know if you need anything else."

        # 5. Optimized Pipeline Response Mock
        if mode_context == "optimized" or "you are nova, an elite customer support ai specialist" in sys_lower:
            # Extract customer message inside prompt
            cust_msg = prompt
            if "Customer Message:" in prompt:
                cust_msg = prompt.split("Customer Message:")[-1].strip()

            order_match = re.search(r"APX-\d+", cust_msg, re.IGNORECASE)
            order_id = order_match.group(0).upper() if order_match else "your order"

            price_match = re.search(r"(?:₹|\$)(\d+(?:\.\d+)?)", cust_msg)
            amount = float(price_match.group(1)) if price_match else 0.0

            if "broken" in cust_msg.lower() or "shattered" in cust_msg.lower() or "crushed" in cust_msg.lower() or "damaged" in cust_msg.lower() or "ruined" in cust_msg.lower():
                if amount > 200:
                    return f"I am truly sorry to hear that your item in order {order_id} arrived damaged! I completely understand how disappointing and frustrating this is.\n\nBecause refunds over ₹200 require supervisor verification, I have escalated your request to our senior support team under ticket #SUP-ESCALATE. A supervisor will review and finalize your ₹{amount:.2f} refund or replacement within 24 business hours.\n\nWe have also emailed you a prepaid return shipping label and a 10% discount code (GOODWILL10) for your next purchase."
                else:
                    amt_str = f"₹{amount:.2f}" if amount > 0 else "full"
                    return f"I am so sorry to hear that your item in order {order_id} arrived damaged, and I completely understand your frustration!\n\nSince your item is under ₹200, I have immediately authorized a full refund of {amt_str} back to your original payment method (3-5 business days processing). If you prefer a free replacement instead, just let me know!\n\nI've also emailed you a prepaid return shipping label along with a 10% discount code (GOODWILL10) for your next order."

            elif "refund" in cust_msg.lower() or "money back" in cust_msg.lower():
                if amount > 200:
                    return f"I understand your urgent concern regarding the refund for order {order_id} (₹{amount:.2f}). I am so sorry for the hassle!\n\nAs refunds exceeding ₹200 require human supervisor authorization, I have submitted your order for priority supervisor review (#SUP-ESCALATE). A supervisor will finalize your ₹{amount:.2f} refund within 24 business hours.\n\nThank you for your patience while we ensure this is handled smoothly!"
                else:
                    amt_str = f"₹{amount:.2f}" if amount > 0 else "full"
                    return f"I am very sorry for the issue with order {order_id}. I understand you want a refund right away!\n\nI have immediately approved your refund of {amt_str} back to your original payment method (takes 3-5 business days). I have also sent a prepaid return shipping label to your email, along with a 10% goodwill voucher (GOODWILL10)."

            elif "42 days" in cust_msg.lower() or "35 days" in cust_msg.lower() or "55 days" in cust_msg.lower():
                return f"Thank you for contacting us regarding order {order_id}! While our standard cash refund window is 30 days, items returned between 31 and 60 days are eligible for Apex Store Credit.\n\nI can issue a prepaid return shipping label right now for you to return the item for full store credit. Would you like me to proceed with generating the store credit label?"

            elif "charged" in cust_msg.lower() or "double" in cust_msg.lower() or "subscription" in cust_msg.lower():
                return f"I am so sorry for the billing confusion! I completely understand how frustrating unexpected charges can be.\n\nI have reviewed your account and immediately issued a full refund of the duplicate charge back to your payment method (processing takes 3-5 business days). I've also emailed you a transaction receipt confirmation for your records."

            elif "90 days" in cust_msg.lower() or "4 years" in cust_msg.lower():
                return "Thank you for reaching out. Under Apex Store policy, items purchased beyond 60 days are outside our return window and cannot be returned or refunded. We appreciate your understanding and would be happy to assist you with any questions about our current product lineup."

            else:
                return f"I am glad to help you with your inquiry regarding order {order_id}! Our team is committed to making sure your experience with Apex Store is outstanding.\n\nI have noted your request and provided detailed options in your account dashboard. Please let me know if you would like me to assist with anything else today!"

        # General Fallback
        return "Thank you for contacting Apex Customer Support. We have processed your request according to company policy."
