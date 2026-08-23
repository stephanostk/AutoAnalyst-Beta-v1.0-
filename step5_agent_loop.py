"""
Step 5 & 6: Chain through all plan steps, then generate a final report.

This ties everything together: Plan -> for each step, Generate Code ->
Execute -> Observe (feeding results forward as context) -> finally,
Report.

This is still the PROTOTYPE version (no retries/self-correction yet —
that's next). If a step fails, we record the failure and move on, so
one bad step doesn't kill the whole run.
"""

from step1_explore import load_and_describe, schema_summary
from step2_llm import call_llm
from step3_plan import build_plan
from step4_generate_and_execute import generate_code, execute_code

REPORT_SYSTEM_PROMPT = """You are a data analyst agent. You have just run a \
multi-step analysis to answer a business question. You will be given the \
original question, and for each step: what was done, the code that ran, \
and its output (or error).

Write a short, business-style report (3-5 paragraphs) that:
- Directly answers the original question using the actual numbers observed.
- Explains what likely drives the pattern found.
- Notes anything unreliable or incomplete in the findings (e.g. a step that
  failed or had missing data).

Do not restate the code. Write for a manager who wants the takeaway, not
the mechanics.
"""


def run_agent(csv_path: str, question: str):
    df = load_and_describe(csv_path)
    schema_text = schema_summary(df)

    print(f"\n=== Question: {question} ===")

    plan = build_plan(schema_text, question)
    print("\n=== Plan ===")
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")

    # This is what gets fed to the report step at the end
    step_records = []

    for i, step_description in enumerate(plan, 1):
        print(f"\n--- Running step {i}: {step_description} ---")

        # Feed prior steps' results into the code-gen prompt as context,
        # so later steps can build on earlier ones (e.g. reference
        # "the furniture rows filtered in step 1").
        context_note = ""
        if step_records:
            context_note = "\n\nContext from prior steps:\n" + "\n".join(
                f"Step {r['step_num']} ({r['description']}): "
                f"{'succeeded' if r['success'] else 'FAILED'}. "
                f"Output: {r['output'][:300]}"
                for r in step_records
            )

        code = generate_code(schema_text, step_description + context_note)
        print("Generated code:\n" + code)

        result = execute_code(code, df)
        record = {
            "step_num": i,
            "description": step_description,
            "code": code,
            "success": result["success"],
            "output": result["output"] if result["success"] else result["error"],
        }
        step_records.append(record)

        if result["success"]:
            print("Result: SUCCESS")
            print(result["output"])
        else:
            print("Result: FAILED (continuing to next step)")
            print(result["error"])

    # Final step: synthesize everything into a narrative report
    print("\n=== Generating Final Report ===")
    steps_summary = "\n\n".join(
        f"Step {r['step_num']}: {r['description']}\n"
        f"Status: {'SUCCESS' if r['success'] else 'FAILED'}\n"
        f"Output:\n{r['output']}"
        for r in step_records
    )

    report_prompt = f"""Original question: {question}

Steps and results:
{steps_summary}

Write the report now."""

    report = call_llm(report_prompt, system=REPORT_SYSTEM_PROMPT, max_tokens=2048)
    print(report)

    return report


if __name__ == "__main__":
    run_agent("sample_sales.csv", "Why is the Furniture category losing money?")