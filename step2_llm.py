"""
Step 2: Minimal LLM call wrapper (Groq version).

Requires: pip install groq
Requires: set GROQ_API_KEY as an environment variable

Groq's free tier allows far more requests per day than Gemini's free tier,
so this swap avoids the daily quota wall we hit with Gemini.
"""

import os
import time
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# llama-3.3-70b-versatile is a strong, fast, general-purpose model well
# suited to writing pandas code and business-style reports.
MODEL_NAME = "openai/gpt-oss-120b"


def call_llm(prompt: str, system: str = None, max_tokens: int = 2048) -> str:
    """
    Sends a single prompt to Groq and returns the text response.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_attempts:
                wait_time = 15 * attempt
                print(f"[rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_attempts}]")
                time.sleep(wait_time)
                continue
            raise

    raise RuntimeError("Exceeded max retry attempts due to rate limiting.")


if __name__ == "__main__":
    result = call_llm("Reply with exactly: LLM connection working.")
    print(result)