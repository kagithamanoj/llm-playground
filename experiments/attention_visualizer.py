"""
Attention Mechanism Visualizer — Understand how transformer attention works.

Usage:
    python experiments/attention_visualizer.py
"""

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix (batch, seq_len, d_k)
        K: Key matrix (batch, seq_len, d_k)
        V: Value matrix (batch, seq_len, d_v)
        mask: Optional attention mask

    Returns:
        output: Attention output
        weights: Attention weights (for visualization)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute raw attention scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    # Step 2: Apply mask (if causal/padding)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 3: Softmax to get attention weights
    weights = softmax(scores, axis=-1)

    # Step 4: Weighted sum of values
    output = weights @ V

    return output, weights


def multi_head_attention(X, n_heads=4, d_model=None):
    """
    Multi-head self-attention from scratch.

    Args:
        X: Input embeddings (batch, seq_len, d_model)
        n_heads: Number of attention heads
        d_model: Model dimension (default: inferred from X)
    """
    batch, seq_len, d_model = X.shape if d_model is None else (X.shape[0], X.shape[1], d_model)
    d_k = d_model // n_heads

    # Initialize projection matrices
    np.random.seed(42)
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    W_O = np.random.randn(d_model, d_model) * 0.1

    # Project to Q, K, V
    Q = X @ W_Q  # (batch, seq_len, d_model)
    K = X @ W_K
    V = X @ W_V

    # Split into heads
    Q = Q.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_k)
    K = K.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)

    # Attention per head
    all_outputs = []
    all_weights = []

    for h in range(n_heads):
        output, weights = scaled_dot_product_attention(
            Q[:, h:h+1, :, :].reshape(batch, seq_len, d_k),
            K[:, h:h+1, :, :].reshape(batch, seq_len, d_k),
            V[:, h:h+1, :, :].reshape(batch, seq_len, d_k),
        )
        all_outputs.append(output)
        all_weights.append(weights)

    # Concatenate heads
    concat = np.concatenate(all_outputs, axis=-1)  # (batch, seq_len, d_model)

    # Final projection
    output = concat @ W_O

    return output, all_weights


def create_causal_mask(seq_len):
    """Create a causal (autoregressive) attention mask."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def visualize_attention_ascii(weights, tokens, title="Attention Weights"):
    """Print attention weights as ASCII heatmap."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

    seq_len = len(tokens)
    max_token_len = max(len(t) for t in tokens)

    # Header
    header = " " * (max_token_len + 2)
    for t in tokens:
        header += f"{t:>6s} "
    print(header)
    print(" " * (max_token_len + 2) + "─" * (7 * seq_len))

    # Rows
    blocks = " ░▒▓█"
    for i, token in enumerate(tokens):
        row = f"{token:>{max_token_len}s}  "
        for j in range(seq_len):
            w = weights[0, i, j]  # First batch
            block_idx = min(int(w * 5), 4)
            row += f" {blocks[block_idx]}  {w:.2f}".ljust(7)
        print(row)


if __name__ == "__main__":
    print("-" * 60)
    print("Attention Mechanism Visualizer")
    print("-" * 60)

    # Simulate a simple sentence
    tokens = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(tokens)
    d_model = 16

    # Create random embeddings (in practice, these come from an embedding layer)
    np.random.seed(42)
    X = np.random.randn(1, seq_len, d_model)  # batch=1

    # --- Demo 1: Basic Self-Attention ---
    print("\nDemo 1: Self-Attention (no mask)")
    Q = X
    K = X
    V = X
    output, weights = scaled_dot_product_attention(Q, K, V)
    visualize_attention_ascii(weights, tokens, "Self-Attention Weights")
    print(f"\nOutput shape: {output.shape}")

    # --- Demo 2: Causal (GPT-style) Attention ---
    print("\nDemo 2: Causal Attention (GPT-style)")
    mask = create_causal_mask(seq_len)
    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=mask[np.newaxis, :, :])
    visualize_attention_ascii(weights_causal, tokens, "Causal Attention Weights")
    print("  ↑ Each token can only attend to previous tokens (lower triangle)")

    # --- Demo 3: Multi-Head Attention ---
    print("\nDemo 3: Multi-Head Attention (4 heads)")
    mha_output, head_weights = multi_head_attention(X, n_heads=4)
    print(f"\nOutput shape: {mha_output.shape}")
    print(f"Number of heads: {len(head_weights)}")

    for h, hw in enumerate(head_weights):
        visualize_attention_ascii(hw, tokens, f"Head {h+1} Attention")

    # === Summary ===
    print("\n" + "=" * 60)
    print("💡 Key Concepts:")
    print("  • Self-attention: each token attends to ALL tokens")
    print("  • Causal mask: each token only sees PREVIOUS tokens (GPT)")
    print("  • Multi-head: multiple attention patterns in parallel")
    print("  • Scaling by √d_k prevents softmax saturation")
    print("=" * 60)
