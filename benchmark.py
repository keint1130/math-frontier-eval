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


def load_config() -> dict:
    with open(BASE / "config.yaml") as f:
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


_prompt_dataset_cache: dict | None = None


def load_prompt_dataset() -> dict:
    """Load pre-built prompt dataset. Returns {(retriever, problem_id): prompt}.

    Tries local cache first (full parquet), then loads per-retriever subsets from HuggingFace.
    """
    global _prompt_dataset_cache
    if _prompt_dataset_cache is not None:
        return _prompt_dataset_cache

    import pandas as pd

    if LOCAL_PROMPTS_CACHE.exists():
        print(f"  Loading prompts from local cache: {LOCAL_PROMPTS_CACHE}")
        df = pd.read_parquet(LOCAL_PROMPTS_CACHE)
        _prompt_dataset_cache = dict(
            zip(zip(df["retriever"], df["problem_id"]), df["prompt"])
        )
    else:
        print(f"  Downloading prompts from {HF_PROMPTS_REPO}...")
        from datasets import get_dataset_config_names, load_dataset

        subsets = get_dataset_config_names(HF_PROMPTS_REPO)
        _prompt_dataset_cache = {}
        for subset in subsets:
            hf_ds = load_dataset(HF_PROMPTS_REPO, subset, split="train")
            df = hf_ds.to_pandas()
            _prompt_dataset_cache.update(
                zip(zip(df["retriever"], df["problem_id"]), df["prompt"])
            )

    print(f"  Loaded {len(_prompt_dataset_cache)} (retriever, problem) prompts.")
    return _prompt_dataset_cache


def load_prompt_dataset_for_systems(system_names: list[str]) -> dict:
    """Load retrieval prompts only when a requested system needs them."""
    if set(system_names) == {"no_memory"}:
        return {}
    return load_prompt_dataset()


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
) -> tuple[list[list[str]], float, int, int, list[dict]]:
    """Run prompts through the provided LLM and return results + token stats."""
    t0 = time.monotonic()
    in_before = llm.total_input_tokens
    out_before = llm.total_output_tokens

    raw_results = llm.generate(prompts, temperature=temperature, max_tokens=max_tokens)

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
    debug: bool = typer.Option(False, "--debug", help="Run a tiny smoke test"),
):
    """Run evaluation for memory systems."""
    config = load_config()
    eval_cfg = config["eval"]
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
            result_file = RESULTS_DIR / model_short / sys_name / f"n{n_samples}.json"
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

        # Phase 1: Look up pre-built prompts for all systems (instant)
        sys_prompts = {}  # sys_name -> (problem_prompts, flat_prompts)
        for sys_name, _ in todo:
            problem_prompts, flat_prompts = build_prompts_from_dataset(
                sys_name, merged, n_samples, prompt_ds
            )
            sys_prompts[sys_name] = (problem_prompts, flat_prompts)

        # Phase 2: Generate all prompts in one parallel batch
        all_flat = []
        sys_ranges = {}  # sys_name -> (start, end) index into all_flat
        for sys_name, _ in todo:
            if sys_name not in sys_prompts:
                continue
            start = len(all_flat)
            all_flat.extend(sys_prompts[sys_name][1])
            sys_ranges[sys_name] = (start, len(all_flat))

        n_sys = len(sys_ranges)
        print(f"\n  {len(all_flat)} prompts across {n_sys} systems")

        judge = eval_cfg.get("judge_model", "local/gpt-oss-20b")
        with LLM(model=model_name, max_concurrent=concurrency) as llm:
            results, elapsed, total_in, total_out, per_prompt_usage = generate_parallel(
                llm,
                all_flat,
                eval_cfg["temperature"],
                max_tokens,
            )

        # Phase 3: Partition results back per system, score, and save
        for sys_name, result_file in todo:
            if sys_name not in sys_ranges:
                continue
            start, end = sys_ranges[sys_name]
            sys_results = results[start:end]
            sys_usage = per_prompt_usage[start:end]
            problem_prompts = sys_prompts[sys_name][0]

            details, mean_score, pass_score = score_responses(
                merged,
                problem_prompts,
                sys_results,
                sys_usage,
                n_samples,
                judge,
                concurrency,
            )

            # Per-system token counts
            sys_in = sum(u.get("input_tokens", 0) for u in sys_usage)
            sys_out = sum(u.get("output_tokens", 0) for u in sys_usage)
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
                        "details": details,
                    },
                    f,
                    indent=2,
                )

    # Print summary table with per-dataset breakdown
    rows = []  # (sys_name, model, mean, pass, {ds: mean})
    all_ds = set()
    for result_file in sorted(RESULTS_DIR.glob(f"**/n{n_samples}.json")):
        parts = result_file.relative_to(RESULTS_DIR).parts
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
