"""
shared/router_client.py — OpenAI-compatible client for the router API.
Reads ROUTER_BASE_URL and ROUTER_API_KEY from environment.
Provides retry with exponential backoff on 429/5xx.
"""

import os
import time
import json
from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client pointing at the router."""
    base_url = os.environ.get("ROUTER_BASE_URL")
    api_key = os.environ.get("ROUTER_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "ROUTER_BASE_URL and ROUTER_API_KEY must be set. "
            "Load .secrets/router.env first."
        )
    return OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)


def chat_completion(
    client: OpenAI,
    messages: list[dict],
    model: str = "google/gemini-2.0-flash-001",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    extra_body: dict | None = None,
) -> dict:
    """
    Call the router's chat completion endpoint with retry logic.
    Returns the full response object as a dict.
    """
    backoff = initial_backoff
    last_exc = None

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            # Extract usage info
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }

            content = ""
            if response.choices and response.choices[0].message:
                msg = response.choices[0].message
                content = msg.content or ""
                # Qwen3 thinking mode: if content is empty, check reasoning field
                if not content and hasattr(msg, "reasoning") and msg.reasoning:
                    content = msg.reasoning

            return {
                "content": content,
                "usage": usage,
                "model": response.model if hasattr(response, "model") else model,
            }
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            # Retry on rate limit, server errors, or timeouts
            if "429" in err_str or "500" in err_str or "502" in err_str or \
               "503" in err_str or "504" in err_str or "rate" in err_str or \
               "timeout" in err_str or "timed out" in err_str:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            raise

    raise RuntimeError(
        f"Router call failed after {max_retries} retries: {last_exc}"
    )


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  model: str = "google/gemini-2.0-flash-001") -> float:
    """Rough cost estimate in USD. Adjust rates per model."""
    # Approximate rates (per 1M tokens)
    rates = {
        "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
        "google/gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.60},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.08},
        "anthropic/claude-4.5-haiku": {"input": 0.80, "output": 4.00},
    }
    r = rates.get(model, {"input": 0.50, "output": 1.50})
    cost = (prompt_tokens * r["input"] + completion_tokens * r["output"]) / 1_000_000
    return round(cost, 6)
