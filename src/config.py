import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prompt Paths
BASELINE_PROMPT_PATH = PROMPTS_DIR / "baseline_prompt.txt"
CLASSIFIER_PROMPT_PATH = PROMPTS_DIR / "classifier_prompt.txt"
OPTIMIZED_PROMPT_PATH = PROMPTS_DIR / "optimized_system_prompt.txt"
GUARDRAIL_PROMPT_PATH = PROMPTS_DIR / "guardrail_prompt.txt"
EVALUATOR_PROMPT_PATH = PROMPTS_DIR / "evaluator_judge_prompt.txt"

# Dataset & Benchmark Paths
DATASET_PATH = DATA_DIR / "customer_test_dataset.json"
BENCHMARK_REPORT_PATH = RESULTS_DIR / "benchmark_report.json"

# Pricing Assumptions (Per 1k tokens - Gemini Flash reference)
INPUT_TOKEN_COST_PER_1K = 0.00015
OUTPUT_TOKEN_COST_PER_1K = 0.00060


def load_prompt_template(path: Path) -> str:
    """Safely read prompt text from template file."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing at {path}")
    return path.read_text(encoding="utf-8")
