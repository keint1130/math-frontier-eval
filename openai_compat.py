"""Compatibility fixes for OpenAI models not yet known by llm-provider."""

from __future__ import annotations


def install_gpt6_compatibility() -> None:
    """Teach llm-provider's Chat Completions adapter about GPT-6 parameters."""
    from llm_provider.providers import openai_api

    current = openai_api._prepare_kwargs
    if getattr(current, "_math_frontier_gpt6_compat", False):
        return

    def prepare_kwargs(model_id: str, kwargs: dict):
        normalized = dict(kwargs)
        if model_id.startswith("gpt-6"):
            if "max_tokens" in normalized and "max_completion_tokens" not in normalized:
                normalized["max_completion_tokens"] = normalized.pop("max_tokens")
            normalized.setdefault("reasoning_effort", "none")
        return current(model_id, normalized)

    prepare_kwargs._math_frontier_gpt6_compat = True
    openai_api._prepare_kwargs = prepare_kwargs

