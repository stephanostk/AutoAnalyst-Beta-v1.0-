"""
FastAPI backend wrapping the AutoAnalyst agent loop.

Two endpoints:
- POST /analyze         : synchronous, returns one full JSON response (original)
- POST /analyze-stream  : streams progress events as the agent works, so the
                          frontend can show live status instead of a blank
                          wait. Uses newline-delimited JSON over a plain
                          streaming HTTP response (simpler than WebSockets
                          for this one-directional use case).
"""

import os
import json
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from step1_explore import load_and_describe, schema_summary
from step2_llm import call_llm
from step3_plan import build_plan
from step4_generate_and_execute import generate_code, execute_code
from step6_agent_loop_v2 import run_step_with_retries


def _looks_degenerate(text: str) -> bool:
    """
    Detects the 'stuck in a loop' failure mode where an LLM repeats the
    same short character/token thousands of times instead of writing
    normal text. Cheap heuristic: if the text is unusually long for a
    report AND a huge fraction of it is a single repeated character,
    treat it as broken output worth retrying.
    """
    if len(text) > 3000:
        from collections import Counter
        counts = Counter(text)
        most_common_char, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(text) > 0.5:
            return True
    return False


def _generate_report_safely(report_prompt: str, system_prompt: str, max_attempts: int = 3) -> str:
    """
    Calls the LLM for the final report, retrying if the output looks
    degenerate (stuck-in-a-loop repetition) instead of a real report.
    """
    for attempt in range(1, max_attempts + 1):
        report = call_llm(report_prompt, system=system_prompt, max_tokens=1024)
        if not _looks_degenerate(report):
            return report
        print(f"[report generation looked degenerate, retrying: attempt {attempt}/{max_attempts}]")
    # If every attempt looked broken, return the last one anyway rather than
    # failing the whole run — better to show something than nothing.
    return report

app = FastAPI(title="AutoAnalyst API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only
    allow_methods=["*"],
    allow_headers=["*"],
)


class StepResult(BaseModel):
    step_num: int
    description: str
    success: bool
    output: str
    attempts: int
    chart_base64: str | None = None


class AnalysisResponse(BaseModel):
    question: str
    plan: list[str]
    steps: list[StepResult]
    report: str


@app.get("/health")
def health():
    key_present = bool(os.environ.get("GROQ_API_KEY"))
    return {"status": "ok", "groq_key_configured": key_present}


def _run_full_analysis(tmp_path: str, question: str):
    """Shared logic: runs the whole agent loop, returns the final structured result."""
    df = load_and_describe(tmp_path)
    schema_text = schema_summary(df)

    plan = build_plan(schema_text, question)
    if not plan:
        raise RuntimeError("The agent failed to produce a plan for this question.")

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    namespace = {"df": df, "pd": pd, "plt": plt}

    step_records = []
    known_vars = []

    for i, step_description in enumerate(plan, 1):
        record = run_step_with_retries(schema_text, step_description, namespace, known_vars)
        record["step_num"] = i
        step_records.append(record)
        known_vars = [k for k in namespace.keys() if k not in ("df", "pd", "plt", "__builtins__")]

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
    report = _generate_report_safely(report_prompt, REPORT_SYSTEM_PROMPT)

    return plan, step_records, report


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile, question: str = Form(...)):
    """Original synchronous endpoint — kept for simple testing via /docs."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported right now.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        plan, step_records, report = _run_full_analysis(tmp_path, question)
        return AnalysisResponse(
            question=question,
            plan=plan,
            steps=[StepResult(**r) for r in step_records],
            report=report,
        )
    except Exception:
        raise HTTPException(status_code=500, detail=f"Agent run failed:\n{traceback.format_exc()}")
    finally:
        os.unlink(tmp_path)


def _sse_event(event_type: str, data: dict) -> str:
    """Formats one line of newline-delimited JSON the frontend can parse as it streams in."""
    payload = json.dumps({"type": event_type, **data})
    return payload + "\n"


async def _stream_analysis(tmp_path: str, question: str, dataset_name: str):
    """
    Generator that yields progress events as the agent works, then cleans
    up the temp file when done (success or failure).
    """
    try:
        df = load_and_describe(tmp_path)
        schema_text = schema_summary(df)

        yield _sse_event("status", {"message": "Planning the analysis..."})
        plan = build_plan(schema_text, question)
        if not plan:
            yield _sse_event("error", {"message": "Failed to produce a plan for this question."})
            return
        yield _sse_event("plan", {"plan": plan})

        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        namespace = {"df": df, "pd": pd, "plt": plt}

        step_records = []
        known_vars = []

        for i, step_description in enumerate(plan, 1):
            yield _sse_event("step_start", {"step_num": i, "description": step_description})

            record = run_step_with_retries(schema_text, step_description, namespace, known_vars)
            record["step_num"] = i
            step_records.append(record)
            known_vars = [k for k in namespace.keys() if k not in ("df", "pd", "plt", "__builtins__")]

            yield _sse_event("step_done", {
                "step_num": i,
                "description": step_description,
                "success": record["success"],
                "output": record["output"],
                "attempts": record["attempts"],
                "chart_base64": record.get("chart_base64"),
            })

        yield _sse_event("status", {"message": "Writing the final report..."})

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
        report = _generate_report_safely(report_prompt, REPORT_SYSTEM_PROMPT)

        yield _sse_event("report", {"report": report})
        yield _sse_event("done", {})

    except Exception:
        yield _sse_event("error", {"message": traceback.format_exc()})
    finally:
        os.unlink(tmp_path)





SUMMARY_SYSTEM_PROMPT = """You are a data analyst agent. Given a dataset's \
schema and basic statistics, write a short, plain-English overview (3-5 \
sentences) covering: what the dataset appears to contain, its size, any \
notable columns, and any data quality issues (missing values, unusual \
types). Do not answer any specific business question — just describe \
what is here. Write for someone who has not seen the data yet."""


@app.post("/summarize")
async def summarize(file: UploadFile):
    """
    Quick endpoint: no question needed. Returns a fast plain-English
    overview of the dataset's shape, columns, and data quality — without
    running the full multi-step agent loop.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported right now.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name

    try:
        df = load_and_describe(tmp_path)
        schema_text = schema_summary(df)

        stats_text = df.describe(include="all").to_string()

        prompt = f"""Dataset schema:
{schema_text}

Basic statistics:
{stats_text}

Write the overview now."""

        summary = call_llm(prompt, system=SUMMARY_SYSTEM_PROMPT, max_tokens=512)

        return {
            "filename": file.filename,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "summary": summary,
        }
    except Exception:
        raise HTTPException(status_code=500, detail=f"Could not summarize dataset:\n{traceback.format_exc()}")
    finally:
        os.unlink(tmp_path)

@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile, question: str = Form(...)):
    """
    Streaming version: returns progress events as newline-delimited JSON,
    one JSON object per line, as the agent works through the plan.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported right now.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    return StreamingResponse(
        _stream_analysis(tmp_path, question, file.filename),
        media_type="application/x-ndjson",
    )