"""
Tokenizer Explorer — Visualize how different tokenizers break down text.

Usage:
    python experiments/tokenizer_explorer.py --text "Hello, how are you?"
"""

import argparse


def explore_tiktoken(text: str):
    """Explore OpenAI's tiktoken tokenizer."""
    import tiktoken

    encodings = {
        "cl100k_base (GPT-4/4o)": tiktoken.get_encoding("cl100k_base"),
        "o200k_base (GPT-4o-mini)": tiktoken.get_encoding("o200k_base"),
    }

    print("=" * 60)
    print(f"📝 Input: \"{text}\"")
    print(f"   Characters: {len(text)}")
    print("=" * 60)

    for name, enc in encodings.items():
        tokens = enc.encode(text)
        decoded = [enc.decode([t]) for t in tokens]

        print(f"\n🔤 {name}")
        print(f"   Token count: {len(tokens)}")
        print(f"   Token IDs: {tokens}")
        print(f"   Tokens: {decoded}")
        print(f"   Chars/token ratio: {len(text)/len(tokens):.1f}")


def explore_hf_tokenizer(text: str, model_name: str = "meta-llama/Llama-3.2-1B"):
    """Explore a Hugging Face tokenizer."""
    try:
        from transformers import AutoTokenizer
        import os

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.getenv("HF_TOKEN")
        )
        tokens = tokenizer.encode(text)
        decoded = tokenizer.convert_ids_to_tokens(tokens)

        print(f"\n🦙 {model_name}")
        print(f"   Token count: {len(tokens)}")
        print(f"   Token IDs: {tokens}")
        print(f"   Tokens: {decoded}")
        print(f"   Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"\n⚠️  Could not load {model_name}: {e}")


def cost_calculator(text: str, model: str = "gpt-4o-mini"):
    """Estimate API cost for processing text."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    tokens = len(enc.encode(text))

    # Pricing per 1M tokens (as of 2024)
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    if model in pricing:
        input_cost = (tokens / 1_000_000) * pricing[model]["input"]
        print(f"\n💰 Cost Estimate ({model})")
        print(f"   Tokens: {tokens:,}")
        print(f"   Input cost: ${input_cost:.6f}")
        print(f"   (~${pricing[model]['input']:.2f} per 1M tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer Explorer")
    parser.add_argument("--text", type=str, default="Hello, how are you? I'm learning about LLMs and tokenization!")
    args = parser.parse_args()

    explore_tiktoken(args.text)
    cost_calculator(args.text)
