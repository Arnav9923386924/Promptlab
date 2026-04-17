"""Chairman synthesis and tie-breaking.

When judges disagree (σ ≥ 0.20), the chairman reviews all scores
and reasoning to produce a final synthesized verdict. The chairman
interprets reasoning quality rather than naive-averaging scores.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from promptlab.finetuning.models import (
    ChairmanVerdict,
    DimensionScores,
    JudgeFeedback,
)
from promptlab.llm_council.llm_runner.runner import LLMRunner

logger = logging.getLogger(__name__)


_CHAIRMAN_SYSTEM_PROMPT = """\
You are the Chairman of a fine-tuning governance council.

Multiple judges have scored a training run but DISAGREE significantly.
Your job is to:

1. Review each judge's scores AND reasoning carefully.
2. Identify which judges have the strongest reasoning.
3. Produce a FINAL synthesized score that reflects reasoning quality,
   NOT a simple average.
4. If one judge flagged a critical safety issue, take it seriously.
5. Explain your tie-break rationale.

Respond with ONLY a JSON object:
{
  "metric_gain": <float 0-1>,
  "stability": <float 0-1>,
  "generalization": <float 0-1>,
  "efficiency": <float 0-1>,
  "safety_alignment": <float 0-1>,
  "confidence": <float 0-1>,
  "synthesis_reasoning": "<string explaining your final decision>",
  "tie_break_rationale": "<string explaining how you resolved disagreements>",
  "overridden_judges": ["<model_name>", ...]
}
"""


class ChairmanSynthesizer:
    """Chairman that resolves judge disagreements."""

    def __init__(
        self,
        llm_runner: LLMRunner,
        chairman_model: Optional[str] = None,
        tuning_goal: str = "",
    ):
        self.llm_runner = llm_runner
        self.chairman_model = chairman_model or "openai/gpt-4o"
        self.tuning_goal = tuning_goal

    def _build_chairman_prompt(
        self,
        feedbacks: list[JudgeFeedback],
        config_summary: str = "",
        metrics_summary: str = "",
    ) -> str:
        """Build the chairman review prompt."""
        parts = [
            f"Tuning goal: {self.tuning_goal}",
            "",
        ]

        if config_summary:
            parts.append(f"Config summary: {config_summary}")
            parts.append("")

        if metrics_summary:
            parts.append(f"Metrics summary: {metrics_summary}")
            parts.append("")

        parts.append("=== JUDGE EVALUATIONS ===")
        for i, fb in enumerate(feedbacks, 1):
            parts.append(f"\n--- Judge {i}: {fb.judge_model} ---")
            parts.append(f"Scores: {fb.scores.model_dump_json()}")
            parts.append(f"Confidence: {fb.confidence}")
            parts.append(f"Reasoning: {fb.reasoning}")
            if fb.flagged_issues:
                parts.append(f"Flagged issues: {', '.join(fb.flagged_issues)}")

        parts.append("\n=== YOUR TASK ===")
        parts.append(
            "Synthesize a final score. Weight reasoning quality over raw numbers. "
            "If any judge flagged critical safety concerns, reflect that in safety_alignment."
        )

        return "\n".join(parts)

    async def synthesize(
        self,
        feedbacks: list[JudgeFeedback],
        config_summary: str = "",
        metrics_summary: str = "",
    ) -> ChairmanVerdict:
        """Run chairman synthesis on judge feedbacks.

        Args:
            feedbacks: All judge feedbacks to review.
            config_summary: Brief summary of the config.
            metrics_summary: Brief summary of training metrics.

        Returns:
            ChairmanVerdict with synthesized scores and rationale.
        """
        prompt = self._build_chairman_prompt(
            feedbacks, config_summary, metrics_summary
        )

        result = await self.llm_runner.complete_with_fallback(
            system_prompt=_CHAIRMAN_SYSTEM_PROMPT,
            user_prompt=prompt,
            preferred_model=self.chairman_model,
        )

        return self._parse_chairman_response(result.content)

    def _parse_chairman_response(self, raw_response: str) -> ChairmanVerdict:
        """Parse chairman response into ChairmanVerdict."""
        content = raw_response.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Chairman returned invalid JSON")

        scores = DimensionScores.compute_final(
            metric_gain=float(data.get("metric_gain", 0.5)),
            stability=float(data.get("stability", 0.5)),
            generalization=float(data.get("generalization", 0.5)),
            efficiency=float(data.get("efficiency", 0.5)),
            safety_alignment=float(data.get("safety_alignment", 1.0)),
        )

        return ChairmanVerdict(
            final_scores=scores,
            confidence=float(data.get("confidence", 0.7)),
            synthesis_reasoning=data.get("synthesis_reasoning", ""),
            tie_break_rationale=data.get("tie_break_rationale"),
            overridden_judges=data.get("overridden_judges", []),
        )

    def fallback_synthesis(
        self, feedbacks: list[JudgeFeedback]
    ) -> ChairmanVerdict:
        """Non-LLM fallback: weighted average by confidence.

        Used when the chairman model itself fails.
        """
        if not feedbacks:
            raise ValueError("No feedbacks to synthesize")

        total_weight = sum(f.confidence for f in feedbacks)
        if total_weight == 0:
            total_weight = len(feedbacks)
            weights = [1.0] * len(feedbacks)
        else:
            weights = [f.confidence for f in feedbacks]

        dims = {
            "metric_gain": 0.0,
            "stability": 0.0,
            "generalization": 0.0,
            "efficiency": 0.0,
            "safety_alignment": 0.0,
        }

        for fb, w in zip(feedbacks, weights):
            for dim in dims:
                dims[dim] += getattr(fb.scores, dim) * w

        for dim in dims:
            dims[dim] = round(dims[dim] / total_weight, 4)

        scores = DimensionScores.compute_final(**dims)

        # Safety: if ANY judge flagged safety, take the minimum
        safety_scores = [f.scores.safety_alignment for f in feedbacks]
        if min(safety_scores) < 0.5:
            scores = DimensionScores.compute_final(
                metric_gain=dims["metric_gain"],
                stability=dims["stability"],
                generalization=dims["generalization"],
                efficiency=dims["efficiency"],
                safety_alignment=min(safety_scores),
            )

        avg_confidence = total_weight / len(feedbacks)

        return ChairmanVerdict(
            final_scores=scores,
            confidence=round(avg_confidence, 3),
            synthesis_reasoning="Fallback: confidence-weighted average of judge scores.",
            tie_break_rationale="No LLM chairman available; used mathematical synthesis.",
            overridden_judges=[],
        )
