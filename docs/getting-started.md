# Getting Started with LLM Playground

## Prerequisites
- Python 3.10+
- OpenAI API key (for prompt patterns and benchmarks)
- Hugging Face token (optional, for fine-tuning gated models)
- GPU with 8GB+ VRAM (optional, for fine-tuning)

## Setup

```bash
git clone https://github.com/kagithamanoj/llm-playground.git
cd llm-playground
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Module Guide

### Prompt Patterns (`prompts/`)
Start here — no GPU needed, just an OpenAI key.

```bash
python prompts/prompt_patterns.py
```

Includes 6 patterns: Chain-of-Thought, Few-Shot, Role Prompting, Structured Output, Self-Consistency, Prompt Chaining.

### Benchmarks (`benchmarks/`)
Compare models head-to-head on standardized tasks.

```bash
python benchmarks/benchmark_runner.py --models gpt-4o-mini gpt-3.5-turbo
```

### Tokenizer Explorer (`experiments/`)
Understand how text becomes tokens and estimate API costs.

```bash
python experiments/tokenizer_explorer.py --text "Your text here"
```

### Fine-Tuning (`fine-tuning/`)
Requires a GPU. Uses LoRA/QLoRA for memory-efficient fine-tuning.

```bash
python fine-tuning/lora_finetune.py --model meta-llama/Llama-3.2-1B --dataset tatsu-lab/alpaca
```
