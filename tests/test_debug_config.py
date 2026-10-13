from unittest.mock import Mock

from benchmark import generate_parallel, resolve_eval_params


def test_debug_allows_complete_olympiad_solution():
    config = {"eval": {"n_samples": 3, "max_tokens": 16384, "concurrency": 48}}
    n_samples, concurrency, max_tokens = resolve_eval_params(config, debug=True)

    assert n_samples == 1
    assert concurrency == 2
    assert max_tokens == 4096


def test_single_prompt_usage_is_attributed_to_result():
    llm = Mock()
    llm.total_input_tokens = 10
    llm.total_output_tokens = 20

    def generate(prompts, **kwargs):
        llm.total_input_tokens = 114
        llm.total_output_tokens = 276
        return ["answer"]

    llm.generate.side_effect = generate
    _, _, total_in, total_out, usage = generate_parallel(llm, ["problem"], 0.0, 4096)

    assert (total_in, total_out) == (104, 256)
    assert usage == [{"input_tokens": 104, "output_tokens": 256}]


def test_reasoning_effort_omits_temperature():
    llm = Mock()
    llm.total_input_tokens = 0
    llm.total_output_tokens = 0
    llm.generate.return_value = ["answer"]

    generate_parallel(
        llm,
        ["problem"],
        None,
        4096,
        use_cache=False,
        reasoning_effort="medium",
    )

    kwargs = llm.generate.call_args.kwargs
    assert kwargs["reasoning_effort"] == "medium"
    assert "temperature" not in kwargs

