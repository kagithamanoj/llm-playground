"""
Embedding Explorer — Visualize and compare text embeddings.

Explore how text gets converted to vectors and find semantic similarity.

Usage:
    python experiments/embedding_explorer.py
"""

import numpy as np


def simple_embedding(text: str, dim: int = 64) -> np.ndarray:
    """
    Create a simple character-level embedding (for demonstration).
    In production, use OpenAI/Sentence-Transformers embeddings.
    """
    # Hash-based embedding (reproducible)
    np.random.seed(hash(text.lower().strip()) % (2**31))
    embedding = np.random.randn(dim)
    # Normalize to unit vector
    return embedding / (np.linalg.norm(embedding) + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def compute_similarity_matrix(texts: list[str], dim: int = 64) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    embeddings = [simple_embedding(t, dim) for t in texts]
    n = len(embeddings)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    return matrix


def find_most_similar(query: str, candidates: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """Find the most similar texts to a query."""
    query_emb = simple_embedding(query)
    scores = []
    for candidate in candidates:
        cand_emb = simple_embedding(candidate)
        score = cosine_similarity(query_emb, cand_emb)
        scores.append((candidate, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def pca_2d(embeddings: list[np.ndarray]) -> np.ndarray:
    """Simple PCA to reduce embeddings to 2D for visualization."""
    X = np.array(embeddings)
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (len(X) - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top 2 components
    top_2 = eigenvectors[:, -2:][:, ::-1]  # Largest first
    return X_centered @ top_2


def visualize_similarity_ascii(texts: list[str], matrix: np.ndarray):
    """Print similarity matrix as ASCII table."""
    print("\n📊 Cosine Similarity Matrix:")
    max_len = max(len(t[:15]) for t in texts)

    # Header
    header = " " * (max_len + 2)
    for i in range(len(texts)):
        header += f"  [{i}]  "
    print(header)

    # Rows
    for i, text in enumerate(texts):
        row = f"{text[:15]:>{max_len}s}  "
        for j in range(len(texts)):
            val = matrix[i][j]
            if val > 0.8:
                indicator = "██"
            elif val > 0.5:
                indicator = "▓▓"
            elif val > 0.2:
                indicator = "▒▒"
            else:
                indicator = "░░"
            row += f"{indicator}{val:.2f} "
        print(row)

    print(f"\n  Legend: ██ >0.8 | ▓▓ >0.5 | ▒▒ >0.2 | ░░ <0.2")


def visualize_2d_ascii(coords: np.ndarray, labels: list[str], width: int = 60, height: int = 20):
    """Plot 2D points as ASCII scatter plot."""
    # Normalize to grid
    x = coords[:, 0]
    y = coords[:, 1]
    x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-8) * (width - 1)).astype(int)
    y_norm = ((y - y.min()) / (y.max() - y.min() + 1e-8) * (height - 1)).astype(int)

    # Create grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for i, (xi, yi) in enumerate(zip(x_norm, y_norm)):
        marker = str(i) if i < 10 else chr(55 + i)  # 0-9, then A, B, C...
        grid[height - 1 - yi][xi] = marker

    print("\n📈 2D Embedding Space (PCA):")
    print("  ┌" + "─" * width + "┐")
    for row in grid:
        print("  │" + "".join(row) + "│")
    print("  └" + "─" * width + "┘")

    print("\n  Key:")
    for i, label in enumerate(labels):
        marker = str(i) if i < 10 else chr(55 + i)
        print(f"    {marker} = {label}")


if __name__ == "__main__":
    print("=" * 60)
    print("🔮 Embedding Explorer")
    print("=" * 60)

    # === Demo 1: Semantic Similarity ===
    print("\n📌 Demo 1: Semantic Grouping")

    texts = [
        # Group 1: Programming
        "Python is a programming language",
        "JavaScript is used for web development",
        "Code review improves software quality",
        # Group 2: Food
        "Pizza is a popular Italian dish",
        "Sushi is a Japanese delicacy",
        "Cooking requires patience and skill",
        # Group 3: Space
        "Mars is the fourth planet from the sun",
        "NASA launched a new satellite",
        "The universe is constantly expanding",
    ]

    matrix = compute_similarity_matrix(texts)
    visualize_similarity_ascii(texts, matrix)

    # === Demo 2: Nearest Neighbor Search ===
    print("\n📌 Demo 2: Nearest Neighbor Search")

    knowledge_base = [
        "How to install Python packages with pip",
        "Setting up a virtual environment in Python",
        "Introduction to machine learning algorithms",
        "Understanding neural networks and deep learning",
        "Web scraping with BeautifulSoup",
        "Building REST APIs with FastAPI",
        "Data visualization with matplotlib",
        "Natural language processing with transformers",
        "Docker containerization best practices",
        "Git version control workflow",
    ]

    queries = [
        "How do I set up my Python project?",
        "I want to learn about AI",
        "How to make charts from data?",
    ]

    for query in queries:
        results = find_most_similar(query, knowledge_base, top_k=3)
        print(f"\n  🔍 Query: \"{query}\"")
        for i, (doc, score) in enumerate(results):
            print(f"     {i+1}. [{score:.3f}] {doc}")

    # === Demo 3: 2D Visualization ===
    print("\n📌 Demo 3: Embedding Space Visualization")

    embeddings = [simple_embedding(t) for t in texts]
    coords = pca_2d(embeddings)
    visualize_2d_ascii(coords, texts)

    # === Summary ===
    print("\n" + "=" * 60)
    print("💡 Key Concepts:")
    print("  • Embeddings convert text to dense vectors")
    print("  • Cosine similarity measures semantic closeness")
    print("  • PCA reduces high-dim vectors to 2D for visualization")
    print("  • Real embeddings (OpenAI, SBERT) capture meaning much better")
    print("  • This demo uses hash-based embeddings for illustration")
    print("=" * 60)
