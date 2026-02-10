# 🧪 LLM Playground

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green)](https://openai.com/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/)

> Interactive experiments with LLMs — prompt engineering, fine-tuning, benchmarking, and tokenizer exploration.

## 🎯 What's Inside

| Module | File | Description |
|--------|------|-------------|
| 🎯 **Prompt Patterns** | `prompts/prompt_patterns.py` | 6 proven prompting techniques with working code |
| 🏋️ **Fine-Tuning** | `fine-tuning/lora_finetune.py` | LoRA/QLoRA pipeline for any Hugging Face model |
| 📊 **Benchmarks** | `benchmarks/benchmark_runner.py` | Compare models on standardized tasks |
| 🔤 **Tokenizer** | `experiments/tokenizer_explorer.py` | Visualize tokenization + estimate API costs |

## 🏗️ Architecture

```
llm-playground/
├── prompts/
│   └── prompt_patterns.py      # CoT, few-shot, role, structured, self-consistency, chaining
├── fine-tuning/
│   └── lora_finetune.py        # QLoRA fine-tuning with PEFT + bitsandbytes
├── benchmarks/
│   └── benchmark_runner.py     # Multi-model comparison with pass/fail + latency
├── experiments/
│   └── tokenizer_explorer.py   # tiktoken + HF tokenizer comparison + cost calc
├── docs/
│   └── getting-started.md      # Setup and usage guide
├── .env.example                # API key template
├── requirements.txt
└── LICENSE
```

## 🚀 Quick Start

```bash
git clone https://github.com/kagithamanoj/llm-playground.git
cd llm-playground
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

## 🎯 Prompt Patterns

Six production-proven techniques:

```python
from prompts.prompt_patterns import chain_of_thought, few_shot_classify, structured_output

# Chain of Thought — step-by-step reasoning
answer = chain_of_thought("If 3 apples cost $1.50, how much do 7 cost?")

# Few-Shot — teach by example
sentiment = few_shot_classify("This product changed my life!")

# Structured Output — force JSON responses
data = structured_output("Meeting at 3pm tomorrow to discuss Q4 budget")
```

**All patterns**: Chain-of-Thought, Few-Shot, Role Prompting, Structured Output, Self-Consistency, Prompt Chaining

## 🏋️ LoRA Fine-Tuning

Fine-tune any model with minimal GPU memory:

```bash
python fine-tuning/lora_finetune.py \
  --model meta-llama/Llama-3.2-1B \
  --dataset tatsu-lab/alpaca \
  --lora-r 16 \
  --epochs 3
```

**Features**: QLoRA 4-bit quantization, configurable LoRA rank/alpha, automatic dataset formatting, checkpoint saving

## 📊 Benchmarks

Compare models head-to-head:

```bash
python benchmarks/benchmark_runner.py --models gpt-4o-mini gpt-3.5-turbo --output results.json
```

```
Task                      |    gpt-4o-mini |  gpt-3.5-turbo
---------------------------------------------------------
Reasoning                 |  ✅   820ms    |  ✅   650ms
Code Generation           |  ✅   1200ms   |  ✅   900ms
PASS RATE                 |     5/5 (100%) |    4/5 (80%)
AVG LATENCY               |         950ms  |        780ms
```

## 🔤 Tokenizer Explorer

```bash
python experiments/tokenizer_explorer.py --text "Hello, how are you?"
```

Compare tokenizers, see token IDs, estimate API costs.

## 📖 Documentation

See [docs/getting-started.md](docs/getting-started.md) for detailed setup and usage.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by** [Manoj Kumar Kagitha](https://github.com/kagithamanoj) 🚀