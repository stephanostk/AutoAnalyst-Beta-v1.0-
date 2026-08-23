"""
Step 6: Agent loop v2 — persistent namespace + self-correction retries.

Two fixes over step5_agent_loop.py:

1. FILTER LEAKAGE FIX: all steps now execute in the SAME namespace
   (a shared dict), so a variable created in step 1 (e.g. furniture_df)
   is still there and usable in step 2, instead of each step getting a
   fresh, disconnected namespace. We also tell the LLM explicitly which
   variables already exist, so it reuses them instead of re-deriving
   from scratch.

2. SELF-CORRECTION: if a step's generated code fails, the error is fed
   back to the LLM and it gets up to MAX_RETRIES attempts to fix its
   own code before we give up and move on.
"""

import io
import os
import base64
import contextlib
import traceback

from step1_explore import load_and_describe, schema_summary
from step2_llm import call_llm

MAX_RETRIES = 2

CODE_SYSTEM_PROMPT = """You are a data analyst agent that writes Python code.

You will be given:
- A dataset schema
- A list of variables already defined from previous steps (if any) — REUSE
  these instead of recreating them from `df` when they already contain
  what you need (e.g. if `furniture_df` already exists and is the
  Furniture subset, filter/group ON `furniture_df`, not on `df` again).
- A single analysis step to perform

Assume:
- A dataframe called `df` is already loaded (the FULL, unfiltered dataset).
- pandas is already imported as `pd`.
- matplotlib.pyplot is already imported as `plt`.

Rules:
- Print any numeric/tabular results with print() so they show up in output.
- If producing a chart, save it with plt.savefig('output_chart.png') instead of plt.show().
- If you create a new variable that later steps might need, give it a clear, reusable name.
- Return ONLY executable Python code. No markdown fences, no explanation.
"""

FIX_SYSTEM_PROMPT = """You are a data analyst agent. Your previous code failed.
You will be given the code you wrote, the error it produced, and the
available variables. Fix the code so it runs successfully and still
accomplishes the original step. Return ONLY the corrected executable
Python code, no markdown fences, no explanation.
"""


def strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = [l for l in code.split("\n") if not l.strip().startswith("```")]
        code = "\n".join(lines)
    return code.strip()


def generate_code(schema_text: str, step_description: str, known_vars: list[str]) -> str:
    vars_note = (
        f"\n\nVariables already defined from previous steps: {', '.join(known_vars)}"
        if known_vars else "\n\nNo variables from previous steps yet — this is the first step."
    )
    prompt = f"""Dataset schema:
{schema_text}
{vars_note}

Analysis step to perform:
{step_description}

Write the code now."""

    raw = call_llm(prompt, system=CODE_SYSTEM_PROMPT)
    return strip_fences(raw)


def fix_code(schema_text: str, step_description: str, bad_code: str, error: str, known_vars: list[str]) -> str:
    vars_note = f"\nAvailable variables: {', '.join(known_vars)}" if known_vars else ""
    prompt = f"""Dataset schema:
{schema_text}
{vars_note}

Original step: {step_description}

Code that failed:
{bad_code}

Error produced:
{error}

Fix the code now."""

    raw = call_llm(prompt, system=FIX_SYSTEM_PROMPT)
    return strip_fences(raw)



CHART_FILENAME = "output_chart.png"


def _capture_chart_if_present() -> str | None:
    """
    If the executed code called plt.savefig('output_chart.png'), read it,
    base64-encode it so it can travel over JSON, and delete the file so
    the next step doesn't accidentally reuse a stale chart.
    """
    if os.path.exists(CHART_FILENAME):
        with open(CHART_FILENAME, "rb") as f:
            chart_bytes = f.read()
        os.remove(CHART_FILENAME)
        return base64.b64encode(chart_bytes).decode("utf-8")
    return None

def execute_in_namespace(code: str, namespace: dict) -> dict:
    """
    Executes code in a SHARED, PERSISTENT namespace (mutated in place),
    so variables created in one call are visible in the next.
    """
    output_buffer = io.StringIO()
    result = {"success": False, "output": "", "error": None}
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, namespace)
        result["success"] = True
        result["output"] = output_buffer.getvalue()
        result["chart_base64"] = _capture_chart_if_present()
    except Exception:
        result["output"] = output_buffer.getvalue()
        result["error"] = traceback.format_exc()
        result["chart_base64"] = None
    return result


def run_step_with_retries(schema_text, step_description, namespace, known_vars):
    code = generate_code(schema_text, step_description, known_vars)
    attempt = 0

    while True:
        print(f"Attempt {attempt + 1} — generated code:\n{code}")
        result = execute_in_namespace(code, namespace)

        if result["success"]:
            print("Result: SUCCESS")
            print(result["output"])
            return {"description": step_description, "success": True, "output": result["output"], "attempts": attempt + 1, "chart_base64": result.get("chart_base64")}

        print(f"Result: FAILED (attempt {attempt + 1})")
        print(result["error"][-500:])  # last part of traceback is usually most useful

        attempt += 1
        if attempt > MAX_RETRIES:
            print(f"Giving up on this step after {attempt} attempts.")
            return {"description": step_description, "success": False, "output": result["error"], "attempts": attempt, "chart_base64": None}

        print("Retrying with a fix...")
        code = fix_code(schema_text, step_description, code, result["error"], known_vars)


def run_agent(csv_path: str, question: str):
    df = load_and_describe(csv_path)
    schema_text = schema_summary(df)

    # Import pandas/matplotlib once, into the persistent shared namespace
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace = {"df": df, "pd": pd, "plt": plt}

    from step3_plan import build_plan
    plan = build_plan(schema_text, question)

    print(f"\n=== Question: {question} ===")
    print("\n=== Plan ===")
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")

    step_records = []
    known_vars = []  # names of variables the LLM has created so far

    for i, step_description in enumerate(plan, 1):
        print(f"\n--- Running step {i}: {step_description} ---")
        record = run_step_with_retries(schema_text, step_description, namespace, known_vars)
        record["step_num"] = i
        step_records.append(record)

        # Track any new variables the executed code added to the namespace,
        # so the NEXT step's prompt knows they're available for reuse.
        current_vars = [k for k in namespace.keys() if k not in ("df", "pd", "plt", "__builtins__")]
        known_vars = current_vars

    # Final report (same as step5)
    from step5_agent_loop import REPORT_SYSTEM_PROMPT
    steps_summary = "\n\n".join(
        f"Step {r['step_num']}: {r['description']}\n"
        f"Status: {'SUCCESS' if r['success'] else 'FAILED'} (took {r['attempts']} attempt(s))\n"
        f"Output:\n{r['output']}"
        for r in step_records
    )
    report_prompt = f"""Original question: {question}

Steps and results:
{steps_summary}

Write the report now."""

    print("\n=== Generating Final Report ===")
    report = call_llm(report_prompt, system=REPORT_SYSTEM_PROMPT, max_tokens=2048)
    print(report)
    return report


if __name__ == "__main__":
    run_agent("sample_sales.csv", "Why is the Furniture category losing money?")