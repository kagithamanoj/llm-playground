# llm-playground

Experimental repository for working with Large Language Models. This toolkit includes prompt engineering patterns, fine-tuning scripts, model benchmarking, and tokenization utilities.

## Directory Structure

```text
llm-playground/
├── prompts/              # Production-proven prompting techniques
├── fine-tuning/          # LoRA and QLoRA fine-tuning pipelines
├── benchmarks/           # Multi-model latency and accuracy testing
├── experiments/          # Tokenizer analysis and attention visualization
└── docs/                 # Implementation guides
```

## Modules

- **Prompt Patterns**: Implementations of Chain-of-Thought, Few-Shot, and Structured Output.
- **Fine-Tuning**: Hardware-efficient fine-tuning using PEFT and bitsandbytes.
- **Benchmarks**: Head-to-head comparison of model performance and response times.
- **Tokenizer Explorer**: Visualization of tokenization boundaries and API cost estimation.

## Setup

```bash
# Dependencies
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

## Quick Start

Test the prompt patterns directly:

```bash
python prompts/prompt_patterns.py
```

Run a benchmark between models:

```bash
python benchmarks/benchmark_runner.py --models gpt-4o-mini gpt-3.5-turbo
```

---

**Manoj Kumar Kagitha**
[GitHub](https://github.com/kagithamanoj)