"""
Step 3: The "Plan" step of the agent loop.
"""

from step1_explore import load_and_describe, schema_summary
from step2_llm import call_llm

PLAN_SYSTEM_PROMPT = """You are a data analyst agent. Given a dataset schema \
and a business question, produce a short, concrete plan (3-5 steps) for \
analyzing the data to answer the question.

Rules:
- Each step must be something that can be done with a single pandas/matplotlib \
operation (e.g. "group by category and sum profit", not "understand the business").
- Reference only columns that actually exist in the schema.
- Return ONLY a numbered list, one step per line, no preamble or explanation.
"""


def build_plan(schema_text: str, question: str) -> list[str]:
    prompt = f"""Dataset schema:
{schema_text}

Business question: {question}

Produce the analysis plan now."""

    raw_response = call_llm(prompt, system=PLAN_SYSTEM_PROMPT)

    steps = []
    for line in raw_response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = line.lstrip("0123456789.)- ").strip()
        if cleaned:
            steps.append(cleaned)

    return steps


if __name__ == "__main__":
    df = load_and_describe("sample_sales.csv")
    schema_text = schema_summary(df)

    question = "Why is the Furniture category losing money?"

    print("\n=== Generated Plan ===")
    plan = build_plan(schema_text, question)
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")