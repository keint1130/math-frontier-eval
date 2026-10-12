from llm_provider.providers import openai_api

from openai_compat import install_gpt6_compatibility


def test_gpt6_uses_current_chat_completion_parameters():
    install_gpt6_compatibility()

    kwargs, use_cache = openai_api._prepare_kwargs(
        "gpt-6-luna",
        {"max_tokens": 512, "temperature": 0.0},
    )

    assert kwargs["max_completion_tokens"] == 512
    assert kwargs["reasoning_effort"] == "none"
    assert "max_tokens" not in kwargs
    assert use_cache is True


def test_compatibility_install_is_idempotent():
    install_gpt6_compatibility()
    first = openai_api._prepare_kwargs
    install_gpt6_compatibility()
    assert openai_api._prepare_kwargs is first

