"""
Step 4 & 5: Generate code for a single plan step, then execute it.

Given one step from the plan (e.g. "Filter the dataset for rows where
category equals 'Furniture'"), ask the LLM to write pandas code that
accomplishes it, then run that code against the real dataframe and
capture the output.
"""

import io
import contextlib
import traceback

from step1_explore import load_and_describe, schema_summary
from step2_llm import call_llm

CODE_SYSTEM_PROMPT = """You are a data analyst agent that writes Python code.

You will be given:
- A dataset schema
- A single analysis step to perform

Write pandas/matplotlib code that accomplishes ONLY this step. Assume:
- A dataframe called `df` is already loaded and available.
- pandas is already imported as `pd`.
- matplotlib.pyplot is already imported as `plt`.

Rules:
- Print any numeric/tabular results with print() so they show up in output.
- If producing a chart, save it with plt.savefig('output_chart.png') instead of plt.show().
- Return ONLY executable Python code. No markdown fences, no explanation, no comments about what you're doing.
"""


def generate_code(schema_text: str, step_description: str) -> str:
    prompt = f"""Dataset schema:
{schema_text}

Analysis step to perform:
{step_description}

Write the code now."""

    raw_response = call_llm(prompt, system=CODE_SYSTEM_PROMPT)

    # Strip markdown code fences if the LLM includes them despite instructions
    code = raw_response.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(lines)

    return code.strip()


def execute_code(code: str, df) -> dict:
    """
    Executes the generated code in a restricted namespace, capturing
    stdout and any error. This is a PROTOTYPE execution method — plain
    exec(), not yet sandboxed. Good enough to prove the loop works;
    real isolation (subprocess/Docker) comes in a later step.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, avoids GUI popups
    import matplotlib.pyplot as plt

    namespace = {"df": df, "pd": pd, "plt": plt}
    output_buffer = io.StringIO()

    result = {"success": False, "output": "", "error": None}

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, namespace)
        result["success"] = True
        result["output"] = output_buffer.getvalue()
    except Exception:
        result["output"] = output_buffer.getvalue()
        result["error"] = traceback.format_exc()

    return result


if __name__ == "__main__":
    df = load_and_describe("sample_sales.csv")
    schema_text = schema_summary(df)

    step_description = "Filter the dataset for rows where category equals 'Furniture'."

    print("\n=== Generated Code ===")
    code = generate_code(schema_text, step_description)
    print(code)

    print("\n=== Execution Result ===")
    result = execute_code(code, df)
    if result["success"]:
        print("SUCCESS")
        print(result["output"])
    else:
        print("FAILED")
        print(result["error"])