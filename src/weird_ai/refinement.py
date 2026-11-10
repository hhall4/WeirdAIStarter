"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    try:
        return float(evaluation.get(score_key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    score = get_score(evaluation, score_key)

    details = "\n".join(
        f"- {key}: {value}"
        for key, value in evaluation.items()
        if key != score_key
    )

    focus_text = "\n".join(f"- {item}" for item in focus or [])
    preserve_text = "\n".join(f"- {item}" for item in preserve or [])

    return f"""Improve the following lyrics based on the evaluation.

Original prompt or topic:
{original_prompt}

Current lyrics:
{current_text}

Current overall score: {score}

Evaluation details:
{details or "No additional evaluation details available."}

Areas to improve:
{focus_text or "- Improve overall quality, rhyme, rhythm, and structure."}

Elements to preserve:
{preserve_text or "- Preserve the original topic, tone, and approximate length."}

Make specific improvements to the lyrics while preserving their original intent.
Return only the revised lyrics, without explanations or additional text."""


def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    current_evaluation = evaluation_func(current_text)
    current_score = get_score(current_evaluation, score_key)

    prompt = build_refinement_prompt(
        original_prompt,
        current_text,
        current_evaluation,
        focus=focus,
        preserve=preserve,
        score_key=score_key,
    )

    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        **kwargs,
    }

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    revised_text = generation_func(prompt, **generation_kwargs)
    revised_evaluation = evaluation_func(revised_text)
    revised_score = get_score(revised_evaluation, score_key)

    _, _, accepted = choose_better_text(
        current_text,
        revised_text,
        current_evaluation,
        revised_evaluation,
        score_key=score_key,
    )

    return RefinementStep(
        iteration=1,
        prompt=prompt,
        previous_text=current_text,
        revised_text=revised_text,
        previous_score=current_score,
        revised_score=revised_score,
        accepted=accepted,
        evaluation=revised_evaluation,
    )


def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    current_score = get_score(current_evaluation, score_key)
    revised_score = get_score(revised_evaluation, score_key)

    if require_improvement:
        accepted = revised_score > current_score
    else:
        accepted = revised_score >= current_score

    if accepted:
        return revised_text, revised_evaluation, True

    return current_text, current_evaluation, False


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    if iteration >= max_iterations:
        return False

    if target_score is not None and current_score >= target_score:
        return False

    if stop_on_no_improvement and not last_improved:
        return False

    return True


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero.")

    current_text = initial_text
    initial_evaluation = evaluation_func(initial_text)
    initial_score = get_score(initial_evaluation, score_key)

    current_evaluation = initial_evaluation
    current_score = initial_score

    steps = []
    iteration = 0
    last_improved = True

    while should_continue(
        iteration=iteration,
        max_iterations=max_iterations,
        current_score=current_score,
        target_score=target_score,
        last_improved=last_improved,
        stop_on_no_improvement=stop_on_no_improvement,
    ):
        step = refine_once(
            original_prompt,
            current_text,
            generation_func,
            evaluation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            **kwargs,
        )

        iteration += 1
        step.iteration = iteration

        current_text, current_evaluation, accepted = choose_better_text(
            current_text,
            step.revised_text,
            current_evaluation,
            step.evaluation,
            score_key=score_key,
            require_improvement=require_improvement,
        )

        step.accepted = accepted
        steps.append(step)

        current_score = get_score(current_evaluation, score_key)
        last_improved = accepted

    if target_score is not None and current_score >= target_score:
        stopped_reason = "target_score"
    elif stop_on_no_improvement and not last_improved:
        stopped_reason = "no_improvement"
    elif iteration >= max_iterations:
        stopped_reason = "max_iterations"
    else:
        stopped_reason = "completed"

    return {
        "original_prompt": original_prompt,
        "initial_text": initial_text,
        "initial_evaluation": initial_evaluation,
        "initial_score": initial_score,
        "final_text": current_text,
        "final_evaluation": current_evaluation,
        "final_score": current_score,
        "iterations": steps,
        "iteration_count": iteration,
        "stopped_reason": stopped_reason,
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    lines = [
        "Refinement Summary",
        f"Initial score: {result.get('initial_score', 0.0):.2f}",
        f"Final score: {result.get('final_score', 0.0):.2f}",
        f"Iterations: {len(result.get('iterations', []))}",
        "",
        "Iteration results:",
    ]

    for step in result.get("iterations", []):
        status = "Accepted" if step.accepted else "Rejected"
        lines.append(
            f"Iteration {step.iteration}: {status} "
            f"(Score: {step.previous_score:.2f} -> {step.revised_score:.2f})"
        )

    lines.extend([
        "",
        f"Stopped reason: {result.get('stopped_reason', 'Unknown')}",
    ])

    return "\n".join(lines)