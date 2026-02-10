"""
Model Comparison Benchmark — Compare different LLM models side-by-side.

Tests models on speed, cost, and quality across standard prompts.

Usage:
    python experiments/model_comparison.py
"""

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()


# Standard benchmark prompts
BENCHMARK_PROMPTS = {
    "reasoning": {
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does he have left?",
        "expected": "9",
        "category": "Logic",
    },
    "coding": {
        "prompt": "Write a Python one-liner to find the second largest element in a list.",
        "expected": "sorted(set(lst))[-2]",
        "category": "Code Generation",
    },
    "summarization": {
        "prompt": (
            "Summarize in one sentence: Machine learning is a subset of artificial intelligence "
            "that provides systems the ability to automatically learn and improve from experience "
            "without being explicitly programmed. It focuses on the development of computer programs "
            "that can access data and use it to learn for themselves."
        ),
        "expected": None,  # Subjective
        "category": "Summarization",
    },
    "extraction": {
        "prompt": (
            "Extract the person's name, age, and city from this text:\n"
            "'John Smith, 34 years old, lives in San Francisco and works as a software engineer.'\n"
            "Return as JSON."
        ),
        "expected": '{"name": "John Smith", "age": 34, "city": "San Francisco"}',
        "category": "Extraction",
    },
    "creativity": {
        "prompt": "Write a haiku about Python programming.",
        "expected": None,  # Subjective
        "category": "Creative Writing",
    },
}

# Model configurations
MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "input_cost_per_1m": 0.50,
        "output_cost_per_1m": 1.50,
    },
}


def call_openai(model: str, prompt: str) -> dict:
    """Call OpenAI API and measure performance."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed"}

    client = OpenAI()

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    latency = time.time() - start

    usage = response.usage
    return {
        "response": response.choices[0].message.content,
        "latency_ms": round(latency * 1000),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def estimate_cost(model_config: dict, input_tokens: int, output_tokens: int) -> float:
    """Estimate API call cost in USD."""
    input_cost = (input_tokens / 1_000_000) * model_config["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * model_config["output_cost_per_1m"]
    return round(input_cost + output_cost, 6)


def run_benchmark(models_to_test: list[str] = None) -> dict:
    """Run full benchmark across models and prompts."""
    models_to_test = models_to_test or list(MODELS.keys())
    results = {}

    for model_name in models_to_test:
        if model_name not in MODELS:
            print(f"  ⚠️ Unknown model: {model_name}")
            continue

        config = MODELS[model_name]
        model_results = {}

        print(f"\n{'─'*50}")
        print(f"🤖 Testing: {model_name}")
        print(f"{'─'*50}")

        for task_name, task in BENCHMARK_PROMPTS.items():
            print(f"  📝 {task['category']:20s}", end=" ", flush=True)

            try:
                result = call_openai(config["model"], task["prompt"])

                if "error" in result:
                    print(f"❌ {result['error']}")
                    continue

                cost = estimate_cost(config, result["input_tokens"], result["output_tokens"])

                model_results[task_name] = {
                    "response": result["response"][:100],
                    "latency_ms": result["latency_ms"],
                    "tokens": result["total_tokens"],
                    "cost_usd": cost,
                    "category": task["category"],
                }

                print(f"✅ {result['latency_ms']:>5d}ms | {result['total_tokens']:>4d} tokens | ${cost:.6f}")

            except Exception as e:
                print(f"❌ {str(e)[:50]}")
                model_results[task_name] = {"error": str(e)}

        results[model_name] = model_results

    return results


def print_comparison_table(results: dict):
    """Print a side-by-side comparison table."""
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")

    models = list(results.keys())
    if not models:
        print("  No results to display.")
        return

    # Header
    header = f"{'Metric':<25s}"
    for model in models:
        header += f" {model:>15s}"
    print(header)
    print("─" * (25 + 16 * len(models)))

    # Avg latency
    row = f"{'Avg Latency (ms)':<25s}"
    for model in models:
        latencies = [r["latency_ms"] for r in results[model].values() if "latency_ms" in r]
        avg = round(sum(latencies) / len(latencies)) if latencies else 0
        row += f" {avg:>14d}ms"
    print(row)

    # Avg tokens
    row = f"{'Avg Tokens':<25s}"
    for model in models:
        tokens = [r["tokens"] for r in results[model].values() if "tokens" in r]
        avg = round(sum(tokens) / len(tokens)) if tokens else 0
        row += f" {avg:>15d}"
    print(row)

    # Total cost
    row = f"{'Total Cost':<25s}"
    for model in models:
        costs = [r["cost_usd"] for r in results[model].values() if "cost_usd" in r]
        total = sum(costs)
        row += f" ${total:>13.6f}"
    print(row)

    # Cost per 1K calls
    row = f"{'Cost per 1K calls':<25s}"
    for model in models:
        costs = [r["cost_usd"] for r in results[model].values() if "cost_usd" in r]
        avg_cost = sum(costs) / len(costs) if costs else 0
        row += f" ${avg_cost * 1000:>12.4f}"
    print(row)

    # Tasks completed
    row = f"{'Tasks Completed':<25s}"
    for model in models:
        completed = sum(1 for r in results[model].values() if "error" not in r)
        row += f" {completed:>11d}/{len(BENCHMARK_PROMPTS)}"
    print(row)


if __name__ == "__main__":
    print("=" * 60)
    print("⚡ LLM Model Comparison Benchmark")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set. Running in demo mode.\n")
        print("📋 Available benchmark tasks:")
        for name, task in BENCHMARK_PROMPTS.items():
            print(f"  • {task['category']:20s} — {task['prompt'][:60]}...")
        print(f"\n📋 Models configured: {', '.join(MODELS.keys())}")
        print("\nSet OPENAI_API_KEY in .env to run the full benchmark.")
    else:
        # Run with just the cheapest model for demo (remove the filter for full comparison)
        results = run_benchmark(["gpt-4o-mini"])
        print_comparison_table(results)

        print("\n💡 To compare all models, run:")
        print("   results = run_benchmark(['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'])")
