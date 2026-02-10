"""
LLM Benchmark Runner — Compare models on standardized tasks.

Usage:
    python benchmarks/benchmark_runner.py --models gpt-4o-mini gpt-3.5-turbo
"""

import os
import time
import json
import argparse
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ── Benchmark Tasks ─────────────────────────────────────────────────────────────

BENCHMARK_TASKS = [
    {
        "name": "Reasoning",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step.",
        "expected_contains": ["0.05", "5 cents", "$0.05", "five cents"],
    },
    {
        "name": "Code Generation",
        "prompt": "Write a Python function that checks if a string is a palindrome. Include docstring and type hints.",
        "expected_contains": ["def ", "palindrome", "return"],
    },
    {
        "name": "Summarization",
        "prompt": "Summarize in 2 sentences: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems improve their performance through experience. Common approaches include supervised learning, unsupervised learning, and reinforcement learning.",
        "expected_contains": ["learn", "data"],
    },
    {
        "name": "Structured Output",
        "prompt": 'Return a JSON object with fields "name", "type", "color" for this description: A red Toyota Camry sedan. Return ONLY JSON.',
        "expected_contains": ["{", "Toyota", "red"],
    },
    {
        "name": "Instruction Following",
        "prompt": "List exactly 3 benefits of exercise. Number them 1, 2, 3. Each should be one sentence.",
        "expected_contains": ["1.", "2.", "3."],
    },
]


@dataclass
class BenchmarkResult:
    model: str
    task: str
    response: str
    latency_ms: float
    tokens_used: int
    passed: bool


def run_benchmark(model: str, task: dict, client: OpenAI) -> BenchmarkResult:
    """Run a single benchmark task against a model."""
    start = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            temperature=0,
            max_tokens=500,
        )
        elapsed = (time.time() - start) * 1000
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens

        # Check if response contains expected substrings
        passed = any(
            exp.lower() in content.lower()
            for exp in task["expected_contains"]
        )

        return BenchmarkResult(
            model=model,
            task=task["name"],
            response=content[:200],
            latency_ms=round(elapsed, 1),
            tokens_used=tokens,
            passed=passed,
        )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return BenchmarkResult(
            model=model,
            task=task["name"],
            response=f"ERROR: {e}",
            latency_ms=round(elapsed, 1),
            tokens_used=0,
            passed=False,
        )


def run_all_benchmarks(models: list[str]) -> list[BenchmarkResult]:
    """Run all tasks against all models."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = []

    for model in models:
        print(f"\n🤖 Benchmarking: {model}")
        for task in BENCHMARK_TASKS:
            print(f"  📝 {task['name']}...", end=" ", flush=True)
            result = run_benchmark(model, task, client)
            print(f"{'✅' if result.passed else '❌'} ({result.latency_ms}ms)")
            results.append(result)

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print a formatted summary table."""
    models = list(dict.fromkeys(r.model for r in results))
    tasks = list(dict.fromkeys(r.task for r in results))

    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)

    # Header
    header = f"{'Task':<25}"
    for model in models:
        short = model.split("/")[-1][:15]
        header += f" | {short:>15}"
    print(header)
    print("-" * len(header))

    # Rows
    for task in tasks:
        row = f"{task:<25}"
        for model in models:
            r = next(r for r in results if r.model == model and r.task == task)
            status = f"{'✅' if r.passed else '❌'} {r.latency_ms:>6}ms"
            row += f" | {status:>15}"
        print(row)

    # Summary
    print("-" * len(header))
    summary_row = f"{'PASS RATE':<25}"
    for model in models:
        model_results = [r for r in results if r.model == model]
        passed = sum(1 for r in model_results if r.passed)
        total = len(model_results)
        summary_row += f" | {f'{passed}/{total} ({100*passed/total:.0f}%)':>15}"
    print(summary_row)

    # Average latency
    latency_row = f"{'AVG LATENCY':<25}"
    for model in models:
        model_results = [r for r in results if r.model == model]
        avg = sum(r.latency_ms for r in model_results) / len(model_results)
        latency_row += f" | {f'{avg:.0f}ms':>15}"
    print(latency_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark Runner")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-3.5-turbo"])
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY in .env file first!")
        exit(1)

    results = run_all_benchmarks(args.models)
    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
