from unittest.mock import patch

import benchmark


def test_no_memory_does_not_load_private_prompt_dataset():
    with patch.object(benchmark, "load_prompt_dataset") as load_prompts:
        prompts = benchmark.load_prompt_dataset_for_systems(["no_memory"])

    assert prompts == {}
    load_prompts.assert_not_called()


def test_retrieval_system_loads_private_prompt_dataset():
    expected = {("bm25_retrieval", "problem"): "prompt"}
    with patch.object(benchmark, "load_prompt_dataset", return_value=expected) as load_prompts:
        prompts = benchmark.load_prompt_dataset_for_systems(["bm25_retrieval"])

    assert prompts == expected
    load_prompts.assert_called_once_with(["bm25_retrieval"])

