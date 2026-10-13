"""Robust math eval: string -> math-verify -> sympy -> numeric cascade."""

import multiprocessing
import re
from pathlib import Path
from typing import Optional

import sympy
from latex2sympy2_extended import latex2sympy

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _brace_extract(s: str, start: int) -> Optional[tuple[str, int]]:
    """From position start (at '{'), return (inner_content, end_pos) or None."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    return None


def extract_boxed(s: str) -> Optional[str]:
    """Content of last \\boxed{} or \\fbox{}, with brace-counting for nesting."""
    if not isinstance(s, str):
        return None
    for tag in ("\\boxed", "\\fbox"):
        idx = s.rfind(tag)
        if idx < 0:
            continue
        rest = s[idx + len(tag) :]
        if not rest or rest[0] != "{":
            content = rest.lstrip().split("$")[0].split("\n")[0].strip()
            return content or None
        result = _brace_extract(s, idx + len(tag))
        if result:
            return result[0]
    return None


def extract_all_boxed(s: str) -> list[str]:
    """All \\boxed{} and \\fbox{} contents, brace-counted."""
    if not isinstance(s, str):
        return []
    results = []
    for tag in ("\\boxed", "\\fbox"):
        pos = 0
        while True:
            idx = s.find(tag, pos)
            if idx < 0:
                break
            brace_pos = idx + len(tag)
            if brace_pos >= len(s) or s[brace_pos] != "{":
                pos = brace_pos
                continue
            result = _brace_extract(s, brace_pos)
            if not result:
                break
            results.append(result[0])
            pos = result[1]
    return results


def get_answer_expr(s: str) -> str:
    """Last \\boxed{} content, or last non-empty line as fallback."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    boxed = extract_boxed(s)
    if boxed is not None:
        return boxed
    for line in reversed(s.split("\n")):
        if line.strip():
            return line.strip()
    return s


def _remove_right_units(s: str) -> str:
    if "\\text{ " in s:
        parts = s.split("\\text{ ")
        if len(parts) == 2:
            return parts[0]
    return s


def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for p in parts[1:]:
        if p.startswith("{"):
            out += "\\sqrt" + p
        elif len(p) >= 1:
            out += f"\\sqrt{{{p[0]}}}" + p[1:]
        else:
            return s
    return out


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    if len(parts) <= 1:
        return s
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p.startswith("{"):
            out += p
        elif len(p) >= 2:
            out += f"{{{p[0]}}}{{{p[1]}}}" + p[2:]
        else:
            return s
    return out


def _fix_a_slash_b(s: str) -> str:
    parts = s.split("/")
    if len(parts) != 2:
        return s
    try:
        a, b = int(parts[0]), int(parts[1])
        if s == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
    except ValueError:
        pass
    return s


def normalize(string: str) -> str:
    """Normalize a LaTeX answer string for comparison."""
    for old, new in [
        ("\n", ""),
        ("\\!", ""),
        ("\\\\", "\\"),
        ("tfrac", "frac"),
        ("dfrac", "frac"),
        ("\\left", ""),
        ("\\right", ""),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
        ("\\$", ""),
        ("$", ""),
    ]:
        string = string.replace(old, new)
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace(r"\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    parts = string.split("=")
    if len(parts) == 2 and len(parts[0]) <= 2:
        string = parts[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def _string_check(gold: str, pred: str) -> Optional[bool]:
    return normalize(gold) == normalize(pred)


def _math_verify_check(gold: str, pred: str) -> Optional[bool]:
    try:
        from math_verify import parse, verify

        gp, pp = parse(gold), parse(pred)
        if not gp or not pp:
            return None
        return verify(gp, pp)
    except Exception:
        return None


def _sympy_worker(gold: str, pred: str) -> bool:
    try:
        return bool(sympy.simplify(latex2sympy(gold) - latex2sympy(pred)) == 0)
    except Exception:
        return False


def _sympy_check(gold: str, pred: str, timeout: float = 5.0) -> Optional[bool]:
    """SymPy equivalence with subprocess timeout, direct-call fallback."""
    try:
        pool = multiprocessing.get_context("fork").Pool(1)
        try:
            val = pool.apply_async(_sympy_worker, (gold, pred)).get(timeout=timeout)
            return val or None
        except (multiprocessing.TimeoutError, Exception):
            pass
        finally:
            pool.terminate()
            pool.join()
    except Exception:
        pass
    try:
        val = _sympy_worker(gold, pred)
        return val or None
    except Exception:
        return None


def _numeric_check(gold: str, pred: str, tol: float = 1e-6) -> Optional[bool]:
    try:
        return abs(float(gold) - float(pred)) < tol
    except (ValueError, TypeError):
        return None


# LLM judge + proof grading


def _llm_judge_answer(
    gold: str, pred: str, model: str = "openrouter/openai/gpt-oss-20b"
) -> float:
    try:
        from llm_provider import LLM

        prompt = _load_prompt("answerbench.txt").format(
            problem_statement="N/A", student_answer=pred, gold_answer=gold
        )
        with LLM(model) as llm:
            resp = llm.chat([{"role": "user", "content": prompt}])
        boxed = extract_boxed(resp)
        return (
            1.0
            if boxed and "correct" in boxed.lower() and "incorrect" not in boxed.lower()
            else 0.0
        )
    except Exception:
        return 0.0


def _remove_self_evaluation(text: str) -> str:
    for pat in [
        r"(?i)#+\s*self[- ]?evaluation.*",
        r"(?i)#+\s*self[- ]?reflection.*",
        r"(?i)#+\s*self[- ]?assessment.*",
    ]:
        text = re.split(pat, text)[0]
    return text.strip()


def _format_guidelines(guidelines) -> str:
    """Normalize grading guidelines to string. Handles plain str or list-of-dicts."""
    if isinstance(guidelines, str):
        return guidelines
    if isinstance(guidelines, list):
        parts = []
        for item in guidelines:
            title = item.get("title", "")
            pts = item.get("max_points", item.get("points", 0))
            desc = item.get("grading_scheme_desc", item.get("desc", ""))
            parts.append(f"**{title}** ({pts} pts): {desc}")
        return "\n\n".join(parts)
    return str(guidelines)


def build_proof_prompt(
    problem: str,
    student_answer: str,
    solution: str,
    guidelines,
    dataset: str = "imoproofbench",
) -> str:
    """Build the LLM grading prompt for a proof (without calling the LLM).

    dataset: 'imoproofbench' uses proofbench.txt (with reference solution),
             'proofbench' uses proofbench_proofbench.txt (no reference solution).
    """
    student_answer = _remove_self_evaluation(student_answer)
    fmt_guidelines = _format_guidelines(guidelines)
    if dataset == "proofbench":
        return _load_prompt("proofbench_proofbench.txt").format(
            problem_statement=problem,
            guidelines=fmt_guidelines,
            student_answer=student_answer,
        )
    return _load_prompt("proofbench.txt").format(
        problem_statement=problem,
        solution=solution,
        guidelines=fmt_guidelines,
        student_answer=student_answer,
    )


def parse_proof_grade(response: str) -> Optional[int]:
    """Extract score from <points> tags. Handles 'N out of 7' and bare 'N' formats."""
    matches = re.findall(r"<points>\s*(\d+)\s*(?:out\s+of\s+7\s*)?</points>", response)
    if not matches:
        return None
    score = int(matches[0])
    return max(0, min(7, score))


_parse_proof_grade = parse_proof_grade  # back-compat alias


def grade_proof(
    problem: str,
    student_answer: str,
    solution: str,
    guidelines,
    model: str = "openrouter/openai/gpt-oss-20b",
    dataset: str = "imoproofbench",
) -> Optional[int]:
    """Grade a proof on 0-7 scale."""
    try:
        from llm_provider import LLM

        prompt = build_proof_prompt(
            problem, student_answer, solution, guidelines, dataset
        )
        with LLM(model) as llm:
            resp = llm.chat([{"role": "user", "content": prompt}])
        return parse_proof_grade(resp)
    except Exception:
        return None


def grade_proofs(
    items: list[dict],
    model: str = "openrouter/openai/gpt-oss-20b",
    max_concurrent: int = 64,
    dataset: str = "imoproofbench",
) -> list[Optional[int]]:
    """Batch-grade proofs on 0-7 scale.

    Each item needs: problem, student_answer, solution, grading_guidelines.
    Returns list of scores (same order as items), None on parse failure.
    """
    from llm_provider import LLM

    prompts = [
        build_proof_prompt(
            problem=item["problem"],
            student_answer=item["student_answer"],
            solution=item.get("solution", ""),
            guidelines=item["grading_guidelines"],
            dataset=dataset,
        )
        for item in items
    ]
    with LLM(model=model, max_concurrent=max_concurrent) as llm:
        results = llm.generate(prompts, temperature=0, max_tokens=4096)
    return [
        parse_proof_grade(r[0] if isinstance(r, list) and r else "") for r in results
    ]


def load_imoproofbench(split: str = "train") -> list[dict]:
    """Load lm-provers/IMOProofBench (60 problems)."""
    from datasets import load_dataset

    return [dict(row) for row in load_dataset("lm-provers/IMOProofBench", split=split)]


def load_proofbench(split: str = "24_25") -> list[dict]:
    """Load lm-provers/ProofBench. Splits: 24_25 (70), other (75), train (145)."""
    from datasets import load_dataset

    ds = load_dataset("lm-provers/ProofBench", split=split)
    return [{**dict(row), "grading_guidelines": row["grading_scheme"]} for row in ds]


# Top-level API


def is_correct(gold: str, pred: str) -> bool:
    """Check equivalence via cascade: string -> math-verify -> sympy -> numeric."""
    if not isinstance(gold, str) or not isinstance(pred, str):
        return False
    gold_candidates = [gold]
    pred_candidates = [pred]
    # Short-answer models often box a labelled equality (for example
    # ``YZ=7/2``) while the benchmark stores only ``7/2``. Compare the RHS as
    # an additional candidate without discarding equation-valued answers.
    if gold.count("=") == 1:
        gold_candidates.append(gold.split("=", 1)[1])
    if pred.count("=") == 1:
        pred_candidates.append(pred.split("=", 1)[1])
    for gold_candidate in gold_candidates:
        for pred_candidate in pred_candidates:
            candidate_pairs = [
                (gold_candidate, pred_candidate),
                (normalize(gold_candidate), normalize(pred_candidate)),
            ]
            for gold_value, pred_value in candidate_pairs:
                for check in [_string_check, _math_verify_check, _sympy_check, _numeric_check]:
                    if check(gold_value, pred_value) is True:
                        return True
    return False


def verify(
    response: str,
    ground_truth: str,
    use_llm_judge: bool = False,
    llm_model: str = "openrouter/openai/gpt-oss-20b",
) -> float:
    """Extract answer from response, check against ground_truth. Returns 1.0/0.0."""
    if response is None or ground_truth is None:
        return 0.0
    if not isinstance(response, str):
        response = str(response)
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    pred = get_answer_expr(response)
    if is_correct(ground_truth, pred):
        return 1.0
    if use_llm_judge:
        return _llm_judge_answer(ground_truth, pred, model=llm_model)
    return 0.0
