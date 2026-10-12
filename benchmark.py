"""Frontier retriever benchmark — prompts loaded from pre-built HF dataset.

Prompts are pre-computed; no corpus or retriever code runs at eval time.

Usage:
    uv run benchmark.py run                            # all retrievers + datasets from config
    uv run benchmark.py run --retriever no_memory      # single retriever
    uv run benchmark.py run --dataset imo_answerbench  # single dataset
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import typer
import yaml
from data.eval_datasets import load_eval_problems
from grading import grade_proofs
from grading import verify as _verify_answer
from llm_provider import LLM
from openai_compat import install_gpt6_compatibility

os.environ.setdefault("LOCAL_BASE_URL", "http://iris-hgx-2:30000/v1")
app = typer.Typer(pretty_exceptions_enable=False)

install_gpt6_compatibility()

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
HF_PROMPTS_REPO = "yoonholee/math-frontier-prompts"
LOCAL_PROMPTS_CACHE = BASE / "prompts_dataset.parquet"

# Official prompt templates (from QED-Nano eval)
ANSWER_TEMPLATE = "{problem_statement}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
PROOF_TEMPLATE = (
    "Generate a rigorous proof to the following question:\n\n{problem_statement}"
)

# Datasets that use proof grading (no groundtruth answer)
PROOF_DATASETS = {"usamo", "imo_proofbench"}


def load_config(path: Path | None = None) -> dict:
    with open(path or (BASE / "config.yaml")) as f:
        return yaml.safe_load(f)


def resolve_eval_params(
    config: dict,
    *,
    n_samples: int | None = None,
    concurrency: int | None = None,
    max_tokens: int | None = None,
    debug: bool = False,
    use_val: bool = False,
) -> tuple[int, int, int]:
    """Resolve n_samples, concurrency, max_tokens from config + overrides."""
    eval_cfg = config["eval"]
    if debug:
        ns = 1
    elif n_samples:
        ns = n_samples
    elif use_val:
        ns = eval_cfg.get("val_n_samples", eval_cfg["n_samples"])
    else:
        ns = eval_cfg["n_samples"]
    mt = max_tokens or eval_cfg["max_tokens"]
    cc = concurrency or eval_cfg.get("concurrency", 32)
    if debug:
        # Olympiad solutions routinely exceed 256 tokens. Keep debug mode to
        # one problem/sample, but allow a complete derivation and boxed answer.
        mt = min(mt, 4096)
        cc = min(cc, 2)
    return ns, cc, mt


def resolve_systems(
    memory: str | None, config: dict, *, debug: bool = False
) -> list[str]:
    """Resolve retriever system names from --memory flag or config."""
    if memory:
        names = [s.strip() for s in memory.split(",")]
    else:
        retrievers = config["retrievers"]
        names = retrievers["baselines"] + retrievers.get(
            "frontier", retrievers.get("proposed", [])
        )
    if debug:
        names = names[:10]
    return names


_prompt_dataset_cache: dict[tuple[str, ...] | None, dict] = {}


def load_prompt_dataset(retrievers: list[str] | None = None) -> dict:
    """Load pre-built prompt dataset. Returns {(retriever, problem_id): prompt}.

    Tries local cache first (full parquet), then loads per-retriever subsets from HuggingFace.
    """
    cache_key = tuple(sorted(set(retrievers))) if retrievers else None
    if cache_key in _prompt_dataset_cache:
        return _prompt_dataset_cache[cache_key]

    import pandas as pd

    if LOCAL_PROMPTS_CACHE.exists():
        print(f"  Loading prompts from local cache: {LOCAL_PROMPTS_CACHE}")
        df = pd.read_parquet(LOCAL_PROMPTS_CACHE)
        prompt_dataset = dict(
            zip(zip(df["retriever"], df["problem_id"]), df["prompt"])
        )
        if retrievers:
            wanted = set(retrievers)
            prompt_dataset = {
                key: value for key, value in prompt_dataset.items() if key[0] in wanted
            }
    else:
        print(f"  Downloading prompts from {HF_PROMPTS_REPO}...")
        from datasets import get_dataset_config_names, load_dataset

        prompt_dataset = {}
        configs = get_dataset_config_names(HF_PROMPTS_REPO)
        # The published repository currently stores every retriever in rows of
        # one `default` config. Retain support for an older multi-config layout.
        if configs == ["default"] or not configs:
            hf_ds = load_dataset(HF_PROMPTS_REPO, split="train")
            df = hf_ds.to_pandas()
            if retrievers:
                df = df[df["retriever"].isin(set(retrievers))]
            prompt_dataset.update(
                zip(zip(df["retriever"], df["problem_id"]), df["prompt"])
            )
        else:
            subsets = (
                [name for name in dict.fromkeys(retrievers) if name in configs]
                if retrievers
                else configs
            )
            for subset in subsets:
                hf_ds = load_dataset(HF_PROMPTS_REPO, subset, split="train")
                df = hf_ds.to_pandas()
                prompt_dataset.update(
                    zip(zip(df["retriever"], df["problem_id"]), df["prompt"])
                )
        if retrievers and not prompt_dataset:
            raise RuntimeError(
                "No retrieval prompts were loaded for: " + ", ".join(retrievers)
            )

    _prompt_dataset_cache[cache_key] = prompt_dataset
    print(f"  Loaded {len(prompt_dataset)} (retriever, problem) prompts.")
    return prompt_dataset


def load_prompt_dataset_for_systems(system_names: list[str]) -> dict:
    """Load retrieval prompts only when a requested system needs them."""
    if set(system_names) == {"no_memory"}:
        return {}
    requested = [name for name in system_names if name != "no_memory"]
    return load_prompt_dataset(requested)


def _problem_id(p: dict) -> str:
    return p.get("problem_id") or hashlib.sha256(p["problem"].encode()).hexdigest()[:12]


def _load_eval_problems(
    dataset_names: list[str],
) -> tuple[dict[str, list[dict]], list[dict], int]:
    """Load all eval problems for each requested dataset."""
    all_problems_by_ds: dict[str, list[dict]] = {}
    merged: list[dict] = []

    for ds_name in dataset_names:
        problems = load_eval_problems(ds_name)
        for p in problems:
            p["_ds_tag"] = ds_name
        all_problems_by_ds[ds_name] = problems
        merged.extend(problems)

    return all_problems_by_ds, merged, len(merged)


def _make_prompt(context: str, problem: str, ds_tag: str) -> str:
    """Compose final prompt: retriever context + official template."""
    template = PROOF_TEMPLATE if ds_tag in PROOF_DATASETS else ANSWER_TEMPLATE
    official = template.format(problem_statement=problem)
    if context:
        return context.rstrip() + "\n\n" + official
    return official


def build_prompts_from_dataset(
    retriever_name: str,
    problems: list[dict],
    n_samples: int,
    prompt_ds: dict,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Look up pre-built prompts for each problem. Returns (problem_prompts, flat_prompts)."""
    problem_prompts = []
    flat_prompts = []
    missing = 0
    for p in problems:
        pid = _problem_id(p)
        retriever_prompt = prompt_ds.get((retriever_name, pid))
        ds_tag = p.get("_ds_tag", "")
        if retriever_name == "no_memory" or retriever_prompt is None:
            if retriever_name != "no_memory" and retriever_prompt is None:
                missing += 1
            context = ""
        else:
            context = retriever_prompt
        prompt = _make_prompt(context, p["problem"], ds_tag)
        problem_prompts.append(prompt)
        for _ in range(n_samples):
            flat_prompts.append(prompt)
    if missing:
        if strict:
            raise RuntimeError(
                f"{retriever_name}: {missing}/{len(problems)} prompts missing from dataset"
            )
        print(
            f"  [WARN] {retriever_name}: {missing}/{len(problems)} prompts missing from dataset",
            file=sys.stderr,
        )
    return problem_prompts, flat_prompts


def generate_parallel(
    llm: LLM,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
    *,
    use_cache: bool = True,
) -> tuple[list[list[str]], float, int, int, list[dict]]:
    """Run prompts through the provided LLM and return results + token stats."""
    t0 = time.monotonic()
    in_before = llm.total_input_tokens
    out_before = llm.total_output_tokens

    raw_results = llm.generate(
        prompts,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=use_cache,
    )

    def _normalize(resp):
        if isinstance(resp, str):
            return [resp]
        if isinstance(resp, list):
            return resp if resp else [""]
        return [str(resp)] if resp is not None else [""]

    results = [_normalize(r) for r in raw_results]
    if len(results) < len(prompts):
        results.extend([[""]] * (len(prompts) - len(results)))
    elif len(results) > len(prompts):
        results = results[: len(prompts)]

    elapsed = time.monotonic() - t0
    total_in = llm.total_input_tokens - in_before
    total_out = llm.total_output_tokens - out_before
    # llm-provider exposes cumulative usage. It can be attributed exactly for
    # the one-prompt debug path; larger batched runs retain aggregate totals.
    per_prompt_usage = [{} for _ in prompts]
    if len(prompts) == 1:
        per_prompt_usage[0] = {
            "input_tokens": total_in,
            "output_tokens": total_out,
        }
    return results, elapsed, total_in, total_out, per_prompt_usage


def score_responses(
    problems: list[dict],
    problem_prompts: list[str],
    results: list[list[str]],
    per_prompt_usage: list[dict],
    n_samples: int,
    judge_model: str,
    max_concurrent: int,
) -> tuple[list[dict], float, float]:
    """Score LLM responses. Returns (details, mean_score, pass_score)."""

    def _to_response(x):
        return x[0] if isinstance(x, list) and x else ""

    # Split flat results into blocks per problem; keep usage bookkeeping per block.
    responses_by_problem = []
    usage_by_problem = []
    for i in range(len(problems)):
        start = i * n_samples
        end = start + n_samples
        block = results[start:end]
        responses_by_problem.append([_to_response(r) for r in block])
        block_usage = per_prompt_usage[start:end]
        pad = [{"input_tokens": 0, "output_tokens": 0}] * (n_samples - len(block_usage))
        usage_by_problem.append(block_usage + pad)

    proof_tasks = []  # (problem_idx, response_idx, grading_item)
    for i, p in enumerate(problems):
        if p["groundtruth"] is not None or not p.get("grading_guidelines"):
            continue
        for j, ans in enumerate(responses_by_problem[i]):
            proof_tasks.append(
                (
                    i,
                    j,
                    {
                        "problem": p["problem"],
                        "student_answer": ans,
                        "solution": p.get("solution", ""),
                        "grading_guidelines": p["grading_guidelines"],
                    },
                )
            )

    proof_grades: dict[tuple[int, int], float] = {}
    if proof_tasks:
        print(f"  Grading {len(proof_tasks)} proofs in parallel...")
        raw_scores = grade_proofs(
            [t[2] for t in proof_tasks],
            model=judge_model,
            max_concurrent=max_concurrent,
        )
        for (pi, ri, _), raw in zip(proof_tasks, raw_scores):
            proof_grades[(pi, ri)] = (raw or 0) / 7.0

    details = []
    for i, p in enumerate(problems):
        responses = responses_by_problem[i]
        usage = usage_by_problem[i]
        in_tokens = sum(u.get("input_tokens", 0) for u in usage)
        out_tokens = sum(u.get("output_tokens", 0) for u in usage)
        if p["groundtruth"] is not None:
            scores = [_verify_answer(r, p["groundtruth"]) for r in responses]
        elif p.get("grading_guidelines"):
            scores = [proof_grades.get((i, j), 0.0) for j in range(len(responses))]
        else:
            scores = [0.0] * len(responses)
        if p["groundtruth"] is None:
            passed = max(scores) >= 6.0 / 7.0
        else:
            passed = max(scores) > 0
        detail = {
            "problem": p["problem"],
            "groundtruth": p["groundtruth"],
            "source": p.get("source", ""),
            "prompt": problem_prompts[i],
            "scores": scores,
            "mean": sum(scores) / len(scores),
            "passed": passed,
            "responses": responses,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }
        if "_ds_tag" in p:
            detail["_ds_tag"] = p["_ds_tag"]
        details.append(detail)

    mean_score = sum(d["mean"] for d in details) / len(details) * 100
    pass_score = sum(1 for d in details if d["passed"]) / len(details) * 100
    return details, mean_score, pass_score


def eval_system(
    retriever: str,
    problems: list[dict],
    llm: LLM,
    prompt_ds: dict,
    n_samples: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    judge_model: str = "local/gpt-oss-20b",
) -> dict:
    """Evaluate a retriever on a list of problems. Returns {mean, pass, details}."""
    problem_prompts, flat_prompts = build_prompts_from_dataset(
        retriever, problems, n_samples, prompt_ds
    )
    results, elapsed, total_in, total_out, per_prompt_usage = generate_parallel(
        llm,
        flat_prompts,
        temperature,
        max_tokens,
    )
    details, mean_score, pass_score = score_responses(
        problems,
        problem_prompts,
        results,
        per_prompt_usage,
        n_samples,
        judge_model,
        llm.max_concurrent,
    )
    return {
        "mean": mean_score,
        "pass": pass_score,
        "elapsed_seconds": round(elapsed, 1),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "details": details,
    }


@app.command()
def run(
    config_path: Path = typer.Option(None, "--config", help="Experiment YAML config"),
    memory: str = typer.Option(
        None,
        "--memory",
        "--retriever",
        help="Comma-separated retrievers to run (default: all from config)",
    ),
    model: str = typer.Option(None, help="Override model from config"),
    dataset: str = typer.Option(
        None,
        help="Eval dataset name (single dataset). Default: all datasets from config",
    ),
    n_samples: int = typer.Option(None, help="Override n_samples from config"),
    concurrency: int = typer.Option(None, help="Override concurrency from config"),
    skip_existing: bool = typer.Option(True, help="Skip if result file exists"),
    max_tokens: int = typer.Option(None, help="Override max_tokens from config"),
    problem_ids_file: Path = typer.Option(
        None,
        help="JSON list of problem IDs to evaluate in the listed order",
    ),
    output_dir: Path = typer.Option(None, help="Override results output directory"),
    debug: bool = typer.Option(False, "--debug", help="Run a tiny smoke test"),
):
    """Run evaluation for memory systems."""
    config_path = config_path.resolve() if config_path else None
    config = load_config(config_path)
    eval_cfg = config["eval"]
    config_base = config_path.parent if config_path else BASE
    if problem_ids_file is None and eval_cfg.get("problem_ids_file"):
        problem_ids_file = config_base / eval_cfg["problem_ids_file"]
    if output_dir is None:
        configured_output = eval_cfg.get("output_dir")
        output_dir = config_base / configured_output if configured_output else RESULTS_DIR
    output_dir = output_dir.resolve()
    n_samples, concurrency, max_tokens = resolve_eval_params(
        config,
        n_samples=n_samples,
        concurrency=concurrency,
        max_tokens=max_tokens,
        debug=debug,
    )
    if debug:
        skip_existing = False
        print("[DEBUG] tiny smoke test mode enabled")

    # Resolve models
    if model is None:
        models = [entry["model"] for entry in config["models"]]
    else:
        models = [config.get("model_aliases", {}).get(model, model)]
    if debug:
        models = models[:1]

    system_names = resolve_systems(memory, config, debug=debug)
    # The no-memory baseline uses only the benchmark's official problem
    # template. Avoid requiring access to the private pre-computed retrieval
    # prompts when it is the only requested system.
    prompt_ds = load_prompt_dataset_for_systems(system_names)

    # Build dataset list
    dataset_names = [dataset] if dataset else eval_cfg.get("datasets", ["aime"])
    all_problems_by_ds, merged, total = _load_eval_problems(dataset_names)
    if problem_ids_file:
        with open(problem_ids_file) as f:
            requested_ids = json.load(f)
        if not isinstance(requested_ids, list) or not all(
            isinstance(item, str) for item in requested_ids
        ):
            raise ValueError("problem_ids_file must contain a JSON list of strings")
        problem_by_id = {_problem_id(problem): problem for problem in merged}
        missing_ids = [problem_id for problem_id in requested_ids if problem_id not in problem_by_id]
        if missing_ids:
            raise ValueError(f"Unknown problem IDs: {missing_ids}")
        merged = [problem_by_id[problem_id] for problem_id in requested_ids]
        selected = {_problem_id(problem) for problem in merged}
        all_problems_by_ds = {
            name: [problem for problem in problems if _problem_id(problem) in selected]
            for name, problems in all_problems_by_ds.items()
        }
        total = len(merged)
    if debug:
        trimmed_by_ds: dict[str, list[dict]] = {}
        merged = []
        for ds_name, ds_problems in all_problems_by_ds.items():
            one = ds_problems[:1]
            trimmed_by_ds[ds_name] = one
            merged.extend(one)
        all_problems_by_ds = trimmed_by_ds
        total = len(merged)
    print(f"\nTotal: {total} problems across {len(all_problems_by_ds)} datasets")

    for model_name in models:
        model_short = model_name.split("/")[-1]

        # Figure out which systems still need eval
        todo = []
        for sys_name in system_names:
            result_file = output_dir / model_short / sys_name / f"n{n_samples}.json"
            if skip_existing and result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                print(
                    f"[SKIP] {sys_name}/{model_short}: Mean@{n_samples}={result['mean']:.1f}%"
                )
                continue
            todo.append((sys_name, result_file))

        if not todo:
            continue

        judge = eval_cfg.get("judge_model", "local/gpt-oss-20b")
        use_cache = bool(eval_cfg.get("cache", True))

        # Evaluate systems separately so token and latency accounting remains
        # exact for task-specific versus baseline comparisons.
        for sys_name, result_file in todo:
            problem_prompts, flat_prompts = build_prompts_from_dataset(
                sys_name, merged, n_samples, prompt_ds, strict=bool(problem_ids_file) or debug
            )
            print(f"\n  {len(flat_prompts)} prompts for {sys_name}")
            with LLM(model=model_name, max_concurrent=concurrency) as llm:
                sys_results, elapsed, sys_in, sys_out, sys_usage = generate_parallel(
                    llm,
                    flat_prompts,
                    eval_cfg["temperature"],
                    max_tokens,
                    use_cache=use_cache,
                )

            details, mean_score, pass_score = score_responses(
                merged,
                problem_prompts,
                sys_results,
                sys_usage,
                n_samples,
                judge,
                concurrency,
            )

            tok_rate = sys_out / max(elapsed, 0.1)

            print(f"\n{'=' * 60}")
            print(f"[EVAL] {sys_name} / {model_short} (Mean@{n_samples})")
            print(
                f"  {sys_in // 1000}k in | {sys_out // 1000}k out | {tok_rate:.0f} tok/s | {elapsed:.0f}s"
            )

            # Save
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(
                    {
                        "mean": mean_score,
                        "pass": pass_score,
                        "elapsed_seconds": round(elapsed, 1),
                        "total_input_tokens": sys_in,
                        "total_output_tokens": sys_out,
                        "model": model_name,
                        "retriever": sys_name,
                        "prompt_dataset": HF_PROMPTS_REPO,
                        "datasets": list(all_problems_by_ds.keys()),
                        "n_problems": len(merged),
                        "problem_ids": [_problem_id(problem) for problem in merged],
                        "temperature": eval_cfg["temperature"],
                        "max_tokens": max_tokens,
                        "cache": use_cache,
                        "details": details,
                    },
                    f,
                    indent=2,
                )

    # Print summary table with per-dataset breakdown
    rows = []  # (sys_name, model, mean, pass, {ds: mean})
    all_ds = set()
    for result_file in sorted(output_dir.glob(f"**/n{n_samples}.json")):
        parts = result_file.relative_to(output_dir).parts
        if len(parts) != 3:
            continue
        with open(result_file) as f:
            res = json.load(f)
        ds_scores = {}
        for d in res.get("details", []):
            tag = d.get("_ds_tag", "")
            if tag:
                ds_scores.setdefault(tag, []).append(d["mean"])
        ds_means = {ds: sum(v) / len(v) * 100 for ds, v in ds_scores.items()}
        all_ds.update(ds_means.keys())
        rows.append((parts[1], parts[0], res["mean"], res["pass"], ds_means))
    if rows:
        rows.sort(key=lambda x: -x[2])
        ds_list = sorted(all_ds)
        ds_hdrs = "".join(f" {ds:>8}" for ds in ds_list)
        print(f"\n{'=' * 60}")
        print(f"{'System':<30} {'Mean':>6} {'Pass':>6}{ds_hdrs}")
        print("-" * (46 + 9 * len(ds_list)))
        for sys_name, _, mean, pass_, ds_means in rows:
            ds_cols = "".join(f" {ds_means.get(ds, 0):>7.1f}%" for ds in ds_list)
            print(f"{sys_name:<30} {mean:>5.1f}% {pass_:>5.1f}%{ds_cols}")


if __name__ == "__main__":
    app()
