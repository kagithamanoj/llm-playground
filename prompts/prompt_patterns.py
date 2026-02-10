"""
Prompt Patterns — A collection of proven prompting techniques for LLMs.
Each pattern includes the template, an example, and when to use it.

Usage:
    python prompts/prompt_patterns.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(prompt: str, system: str = "You are a helpful assistant.", model: str = "gpt-4o-mini") -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 1: Chain-of-Thought (CoT)
# ═══════════════════════════════════════════════════════════════════════════════

COT_TEMPLATE = """Solve this step by step.

Question: {question}

Think through this step by step:
1. First, identify what we know
2. Then, determine the approach
3. Finally, calculate the answer

Show your reasoning at each step."""


def chain_of_thought(question: str) -> str:
    """Forces the LLM to reason step-by-step before answering."""
    return call_llm(COT_TEMPLATE.format(question=question))


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 2: Few-Shot Learning
# ═══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_TEMPLATE = """Classify the sentiment of the text as Positive, Negative, or Neutral.

Examples:
Text: "This product is amazing, best purchase ever!"
Sentiment: Positive

Text: "Terrible quality, broke after one day."
Sentiment: Negative

Text: "It arrived on time."
Sentiment: Neutral

Now classify:
Text: "{text}"
Sentiment:"""


def few_shot_classify(text: str) -> str:
    """Uses examples to teach the LLM a task pattern."""
    return call_llm(FEW_SHOT_TEMPLATE.format(text=text))


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 3: Role Prompting
# ═══════════════════════════════════════════════════════════════════════════════

ROLE_TEMPLATE = """You are a {role} with {years} years of experience.
Your communication style is {style}.
You specialize in {specialty}.

User question: {question}

Provide your expert response:"""


def role_prompt(question: str, role: str = "senior ML engineer",
                years: int = 15, style: str = "technical but clear",
                specialty: str = "deploying ML systems at scale") -> str:
    """Sets a persona to shape the LLM's response style and expertise."""
    return call_llm(ROLE_TEMPLATE.format(
        role=role, years=years, style=style,
        specialty=specialty, question=question,
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 4: Structured Output
# ═══════════════════════════════════════════════════════════════════════════════

STRUCTURED_TEMPLATE = """Analyze the following text and return a JSON object with these fields:
- "summary": A one-sentence summary
- "sentiment": "positive", "negative", or "neutral"
- "key_topics": An array of 3-5 key topics
- "action_items": An array of any actionable items mentioned

Text: "{text}"

Return ONLY valid JSON, no other text."""


def structured_output(text: str) -> str:
    """Forces the LLM to return structured data."""
    return call_llm(STRUCTURED_TEMPLATE.format(text=text))


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 5: Self-Consistency (multiple samples + majority vote)
# ═══════════════════════════════════════════════════════════════════════════════

def self_consistency(question: str, n_samples: int = 3) -> dict:
    """Generate multiple answers and pick the most common one."""
    answers = []
    for i in range(n_samples):
        answer = call_llm(
            f"Answer concisely in one word or number.\n\nQuestion: {question}",
        )
        answers.append(answer.strip())

    # Count occurrences
    from collections import Counter
    counts = Counter(answers)
    best = counts.most_common(1)[0]

    return {
        "answer": best[0],
        "confidence": best[1] / n_samples,
        "all_answers": answers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 6: Prompt Chaining (decompose complex tasks)
# ═══════════════════════════════════════════════════════════════════════════════

def prompt_chain(topic: str) -> dict:
    """Chain multiple prompts where each step feeds the next."""

    # Step 1: Research
    research = call_llm(f"List 5 key facts about: {topic}")

    # Step 2: Outline
    outline = call_llm(f"Create a blog post outline using these facts:\n{research}")

    # Step 3: Write
    article = call_llm(f"Write a short blog post following this outline:\n{outline}")

    return {
        "research": research,
        "outline": outline,
        "article": article,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY in .env file first!")
        print("   cp .env.example .env && edit .env")
        exit(1)

    print("=" * 60)
    print("🎯 LLM Prompt Patterns — Demo")
    print("=" * 60)

    print("\n--- 1. Chain of Thought ---")
    result = chain_of_thought("If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is the average speed?")
    print(result[:300])

    print("\n--- 2. Few-Shot Classification ---")
    result = few_shot_classify("The update is okay, nothing special.")
    print(f"Sentiment: {result}")

    print("\n--- 3. Structured Output ---")
    result = structured_output("We need to migrate the database to PostgreSQL by Friday. The team should also set up monitoring dashboards.")
    print(result[:300])

    print("\n✅ Demo complete! See the source code for all 6 patterns.")
