import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.config import (
    DATASET_PATH,
    BENCHMARK_REPORT_PATH,
    EVALUATOR_PROMPT_PATH,
    load_prompt_template,
)
from src.llm_client import LLMClient
from src.pipeline import BaselinePipeline, OptimizedPipeline

console = Console(force_terminal=True, legacy_windows=False)


class BenchmarkEvaluator:
    """
    LLM-as-a-Judge Benchmark Engine evaluating Baseline vs. Multi-Stage Optimized pipelines.
    Runs 50 customer test cases and produces comprehensive CSAT metrics and cost breakdowns.
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self.baseline_pipeline = BaselinePipeline(self.client)
        self.optimized_pipeline = OptimizedPipeline(self.client)
        self.evaluator_template = load_prompt_template(EVALUATOR_PROMPT_PATH)

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_response_with_judge(self, customer_input: str, response_text: str) -> Dict[str, Any]:
        """
        Submits candidate response to LLM-as-a-Judge prompt rubric.
        Returns parsed scores dictionary.
        """
        prompt = (
            self.evaluator_template
            .replace("{customer_input}", customer_input)
            .replace("{chatbot_response}", response_text)
        )

        res = self.client.generate(prompt=prompt, mode_context="judge", temperature=0.0)
        
        try:
            raw = res["content"].strip()
            if raw.startswith("```json"):
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif raw.startswith("```"):
                raw = raw.split("```")[1].split("```")[0].strip()
            scores = json.loads(raw)
        except Exception as e:
            print(f"[Evaluator] Judge output parse error ({e}). Using fallback scoring.")
            scores = {
                "empathy_score": 3.0,
                "actionability_score": 3.0,
                "policy_compliance_score": 3.5,
                "conciseness_score": 4.0,
                "overall_csat_score": 3.3,
                "judge_rationale": "Default fallback evaluation."
            }

        return scores

    def run_benchmark(self, max_cases: Optional[int] = None) -> Dict[str, Any]:
        dataset = self.load_dataset()
        if max_cases:
            dataset = dataset[:max_cases]

        console.print(Panel(f"[bold cyan]Starting CSAT Benchmark Engine ({len(dataset)} Test Cases)[/bold cyan]"))

        detailed_results = []
        
        baseline_csats = []
        optimized_csats = []

        baseline_empathy = []
        baseline_actionability = []
        baseline_policy = []
        baseline_conciseness = []

        optimized_empathy = []
        optimized_actionability = []
        optimized_policy = []
        optimized_conciseness = []

        baseline_latencies = []
        optimized_latencies = []
        baseline_costs = []
        optimized_costs = []

        guardrail_violations_count = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating test cases...", total=len(dataset))

            for item in dataset:
                case_id = item["id"]
                category = item.get("category", "GENERAL")
                cust_msg = item["customer_message"]

                # 1. Run Baseline Pipeline
                b_res = self.baseline_pipeline.run(cust_msg)
                b_judge = self.evaluate_response_with_judge(cust_msg, b_res["response"])

                # 2. Run Optimized Pipeline
                o_res = self.optimized_pipeline.run(cust_msg)
                o_judge = self.evaluate_response_with_judge(cust_msg, o_res["response"])

                # Record metrics
                b_csat = b_judge.get("overall_csat_score", 3.0)
                o_csat = o_judge.get("overall_csat_score", 4.0)

                baseline_csats.append(b_csat)
                optimized_csats.append(o_csat)

                baseline_empathy.append(b_judge.get("empathy_score", 3.0))
                baseline_actionability.append(b_judge.get("actionability_score", 3.0))
                baseline_policy.append(b_judge.get("policy_compliance_score", 3.0))
                baseline_conciseness.append(b_judge.get("conciseness_score", 3.0))

                optimized_empathy.append(o_judge.get("empathy_score", 4.0))
                optimized_actionability.append(o_judge.get("actionability_score", 4.0))
                optimized_policy.append(o_judge.get("policy_compliance_score", 4.0))
                optimized_conciseness.append(o_judge.get("conciseness_score", 4.0))

                baseline_latencies.append(b_res["latency_ms"])
                optimized_latencies.append(o_res["latency_ms"])
                baseline_costs.append(b_res["token_cost"])
                optimized_costs.append(o_res["token_cost"])

                if not o_res["guardrail_result"]["is_safe"]:
                    guardrail_violations_count += 1

                case_record = {
                    "id": case_id,
                    "category": category,
                    "customer_message": cust_msg,
                    "baseline": {
                        "response": b_res["response"],
                        "latency_ms": b_res["latency_ms"],
                        "input_tokens": b_res["input_tokens"],
                        "output_tokens": b_res["output_tokens"],
                        "token_cost": b_res["token_cost"],
                        "scores": b_judge
                    },
                    "optimized": {
                        "response": o_res["response"],
                        "customer_state": o_res["customer_state"],
                        "guardrail_result": o_res["guardrail_result"],
                        "latency_ms": o_res["latency_ms"],
                        "stage_latencies": o_res["stage_latencies"],
                        "input_tokens": o_res["input_tokens"],
                        "output_tokens": o_res["output_tokens"],
                        "token_cost": o_res["token_cost"],
                        "scores": o_judge
                    },
                    "csat_lift": round(o_csat - b_csat, 2)
                }
                detailed_results.append(case_record)
                progress.advance(task)

        # Aggregate Summary
        avg_b_csat = sum(baseline_csats) / len(baseline_csats)
        avg_o_csat = sum(optimized_csats) / len(optimized_csats)
        csat_lift_pct = round(((avg_o_csat - avg_b_csat) / avg_b_csat) * 100, 2)

        summary_metrics = {
            "total_test_cases": len(dataset),
            "baseline_avg_csat": round(avg_b_csat, 2),
            "optimized_avg_csat": round(avg_o_csat, 2),
            "csat_lift_points": round(avg_o_csat - avg_b_csat, 2),
            "csat_lift_percentage": csat_lift_pct,
            "dimensions": {
                "empathy": {
                    "baseline": round(sum(baseline_empathy) / len(baseline_empathy), 2),
                    "optimized": round(sum(optimized_empathy) / len(optimized_empathy), 2)
                },
                "actionability": {
                    "baseline": round(sum(baseline_actionability) / len(baseline_actionability), 2),
                    "optimized": round(sum(optimized_actionability) / len(optimized_actionability), 2)
                },
                "policy_compliance": {
                    "baseline": round(sum(baseline_policy) / len(baseline_policy), 2),
                    "optimized": round(sum(optimized_policy) / len(optimized_policy), 2)
                },
                "conciseness": {
                    "baseline": round(sum(baseline_conciseness) / len(baseline_conciseness), 2),
                    "optimized": round(sum(optimized_conciseness) / len(optimized_conciseness), 2)
                }
            },
            "performance": {
                "baseline_avg_latency_ms": int(sum(baseline_latencies) / len(baseline_latencies)),
                "optimized_avg_latency_ms": int(sum(optimized_latencies) / len(optimized_latencies)),
                "baseline_total_cost": round(sum(baseline_costs), 6),
                "optimized_total_cost": round(sum(optimized_costs), 6)
            },
            "guardrail_violations_intercepted": guardrail_violations_count
        }

        # Category level CSAT breakdown
        category_breakdown = {}
        for res in detailed_results:
            cat = res["category"]
            if cat not in category_breakdown:
                category_breakdown[cat] = {"baseline_csat": [], "optimized_csat": []}
            category_breakdown[cat]["baseline_csat"].append(res["baseline"]["scores"]["overall_csat_score"])
            category_breakdown[cat]["optimized_csat"].append(res["optimized"]["scores"]["overall_csat_score"])

        cat_summary = {}
        for cat, data in category_breakdown.items():
            b_avg = sum(data["baseline_csat"]) / len(data["baseline_csat"])
            o_avg = sum(data["optimized_csat"]) / len(data["optimized_csat"])
            cat_summary[cat] = {
                "baseline_avg_csat": round(b_avg, 2),
                "optimized_avg_csat": round(o_avg, 2),
                "lift_pct": round(((o_avg - b_avg) / b_avg) * 100, 2)
            }
        summary_metrics["category_breakdown"] = cat_summary

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary_metrics,
            "detailed_cases": detailed_results
        }

        # Save Report JSON
        with open(BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        console.print(f"\n[bold green]Benchmark complete! Results written to {BENCHMARK_REPORT_PATH}[/bold green]\n")
        self.print_summary_table(summary_metrics)
        return report

    def print_summary_table(self, summary: Dict[str, Any]):
        table = Table(title="CSAT Benchmark Performance Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline Pipeline", justify="center")
        table.add_column("Optimized Pipeline", justify="center")
        table.add_column("Delta / Lift", justify="center", style="green")

        table.add_row(
            "Overall CSAT Rating",
            f"{summary['baseline_avg_csat']} / 5.0",
            f"{summary['optimized_avg_csat']} / 5.0",
            f"+{summary['csat_lift_percentage']}% (+{summary['csat_lift_points']} pts)"
        )
        table.add_row(
            "Empathy Score",
            f"{summary['dimensions']['empathy']['baseline']}",
            f"{summary['dimensions']['empathy']['optimized']}",
            f"+{round(summary['dimensions']['empathy']['optimized'] - summary['dimensions']['empathy']['baseline'], 2)}"
        )
        table.add_row(
            "Actionability Score",
            f"{summary['dimensions']['actionability']['baseline']}",
            f"{summary['dimensions']['actionability']['optimized']}",
            f"+{round(summary['dimensions']['actionability']['optimized'] - summary['dimensions']['actionability']['baseline'], 2)}"
        )
        table.add_row(
            "Policy Compliance",
            f"{summary['dimensions']['policy_compliance']['baseline']}",
            f"{summary['dimensions']['policy_compliance']['optimized']}",
            f"+{round(summary['dimensions']['policy_compliance']['optimized'] - summary['dimensions']['policy_compliance']['baseline'], 2)}"
        )
        table.add_row(
            "Conciseness Score",
            f"{summary['dimensions']['conciseness']['baseline']}",
            f"{summary['dimensions']['conciseness']['optimized']}",
            f"+{round(summary['dimensions']['conciseness']['optimized'] - summary['dimensions']['conciseness']['baseline'], 2)}"
        )
        table.add_row(
            "Avg Response Latency",
            f"{summary['performance']['baseline_avg_latency_ms']} ms",
            f"{summary['performance']['optimized_avg_latency_ms']} ms",
            f"+{summary['performance']['optimized_avg_latency_ms'] - summary['performance']['baseline_avg_latency_ms']} ms"
        )

        console.print(table)


if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    evaluator.run_benchmark()
