"""Multi-judge evaluation of fine-tuning run artifacts.

Orchestrates multiple judge models to score a run across
governance dimensions, with judge-count enforcement,
fallback logic, and chunking for large artifact sets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Optional

from promptlab.finetuning.models import (
    CandidateConfig,
    DimensionScores,
    DisagreementLevel,
    JudgeFeedback,
    RunArtifact,
    TrainingMetrics,
)
from promptlab.llm_council.llm_runner.runner import LLMRunner

logger = logging.getLogger(__name__)


_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating a fine-tuning training run.

Score the run on these dimensions (0.0 to 1.0):

1. **metric_gain**: How much did the target metric improve vs. baseline / previous best?
2. **stability**: Was training stable? (monotonic loss decrease, no spikes, no NaN)
3. **generalization**: Is the eval-train gap healthy? (low overfitting signal)
4. **efficiency**: Was the run resource-efficient? (time, VRAM, steps)
5. **safety_alignment**: Any safety/alignment red flags? (1.0 = fully safe)

Respond with ONLY a JSON object:
{
  "metric_gain": <float>,
  "stability": <float>,
  "generalization": <float>,
  "efficiency": <float>,
  "safety_alignment": <float>,
  "reasoning": "<string>",
  "confidence": <float 0-1>,
  "flagged_issues": ["<issue1>", ...]
}
"""


class RunEvaluator:
    """Multi-judge evaluator for fine-tuning run artifacts."""

    def __init__(
        self,
        llm_runner: LLMRunner,
        judge_models: list[str],
        min_required_judges: int = 3,
        use_fixed_judges: bool = False,
        max_chunk_tokens: int = 4000,
        tuning_goal: str = "",
        base_model: str = "",
    ):
        self.llm_runner = llm_runner
        self.judge_models = judge_models
        self.min_required_judges = min_required_judges
        self.use_fixed_judges = use_fixed_judges
        self.max_chunk_tokens = max_chunk_tokens
        self.tuning_goal = tuning_goal
        self.base_model = base_model

    def _format_run_artifacts(
        self,
        artifact: RunArtifact,
        previous_best: Optional[RunArtifact] = None,
    ) -> str:
        """Format run artifacts into a judge-readable prompt."""
        parts = [
            f"Tuning goal: {self.tuning_goal}",
            f"Base model: {self.base_model}",
            "",
            "=== CONFIGURATION ===",
            artifact.config.model_dump_json(indent=2),
            "",
            "=== TRAINING METRICS ===",
            artifact.metrics.model_dump_json(indent=2),
            "",
        ]

        if previous_best:
            parts.append("=== PREVIOUS BEST RUN ===")
            parts.append(f"Best eval loss: {previous_best.metrics.best_eval_loss}")
            parts.append(f"Config: {previous_best.config.model_dump_json(indent=2)}")
            parts.append("")

        # Loss curve summary (keep compact)
        if artifact.metrics.loss_curve:
            parts.append("=== LOSS CURVE (sampled) ===")
            curve = artifact.metrics.loss_curve
            sample = self._sample_curve(curve, max_points=20)
            for pt in sample:
                parts.append(json.dumps(pt))
            parts.append("")

        return "\n".join(parts)

    def _sample_curve(
        self, curve: list[dict], max_points: int = 20
    ) -> list[dict]:
        """Downsample loss curve to fit context window."""
        if len(curve) <= max_points:
            return curve
        step = max(1, len(curve) // max_points)
        sampled = [curve[i] for i in range(0, len(curve), step)]
        # Always include the last point
        if sampled[-1] != curve[-1]:
            sampled.append(curve[-1])
        return sampled

    def _needs_chunking(self, prompt: str) -> bool:
        """Check if prompt exceeds chunk token limit (rough estimate)."""
        estimated_tokens = len(prompt) // 4
        return estimated_tokens > self.max_chunk_tokens

    async def evaluate(
        self,
        artifact: RunArtifact,
        previous_best: Optional[RunArtifact] = None,
    ) -> tuple[list[JudgeFeedback], dict]:
        """Evaluate a run using multiple judges.

        Returns:
            Tuple of (judge_feedbacks, metadata_dict).
        """
        prompt = self._format_run_artifacts(artifact, previous_best)
        chunking_used = self._needs_chunking(prompt)

        start_time = time.time()
        feedbacks: list[JudgeFeedback] = []
        judges_attempted = 0
        fallback_used = 0

        for model in self.judge_models:
            if len(feedbacks) >= self.min_required_judges:
                # Check for early agreement
                if self._has_early_agreement(feedbacks):
                    logger.info(f"Early agreement after {len(feedbacks)} judges")
                    break

            judges_attempted += 1
            try:
                feedback = await self._get_judge_feedback(model, prompt)
                feedbacks.append(feedback)
            except Exception as e:
                logger.warning(f"Judge {model} failed: {e}")
                continue

        # Enforce minimum judge count
        if len(feedbacks) < self.min_required_judges:
            if self.use_fixed_judges:
                raise RuntimeError(
                    f"Insufficient judges: got {len(feedbacks)}, "
                    f"required {self.min_required_judges}. "
                    f"Set use_fixed_judges=false to enable fallback."
                )
            # Fallback: try additional models from the runner's pool
            fallback_models = self._get_fallback_models(
                [f.judge_model for f in feedbacks]
            )
            for model in fallback_models:
                if len(feedbacks) >= self.min_required_judges:
                    break
                judges_attempted += 1
                try:
                    feedback = await self._get_judge_feedback(model, prompt)
                    feedbacks.append(feedback)
                    fallback_used += 1
                except Exception as e:
                    logger.warning(f"Fallback judge {model} failed: {e}")
                    continue

        if len(feedbacks) < self.min_required_judges:
            raise RuntimeError(
                f"Insufficient judges even with fallback: "
                f"got {len(feedbacks)}, required {self.min_required_judges}."
            )

        # Compute disagreement
        sigma = self._compute_disagreement(feedbacks)
        level = self._classify_disagreement(sigma)

        elapsed = time.time() - start_time

        metadata = {
            "run_id": artifact.run_id,
            "config_id": artifact.config.config_id,
            "judges_used": [f.judge_model for f in feedbacks],
            "judges_attempted": judges_attempted,
            "judges_succeeded": len(feedbacks),
            "fallback_used": fallback_used,
            "disagreement_level": level,
            "disagreement_sigma": round(sigma, 4),
            "chunking_used": chunking_used,
            "evaluation_time_seconds": round(elapsed, 2),
        }

        return feedbacks, metadata

    async def _get_judge_feedback(
        self, model: str, prompt: str
    ) -> JudgeFeedback:
        """Get evaluation feedback from a single judge model."""
        result = await self.llm_runner.complete_with_fallback(
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            user_prompt=prompt,
            preferred_model=model,
        )

        return self._parse_judge_response(model, result.content)

    def _parse_judge_response(
        self, model: str, raw_response: str
    ) -> JudgeFeedback:
        """Parse judge response into JudgeFeedback."""
        content = raw_response.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Judge {model} returned invalid JSON")

        scores = DimensionScores.compute_final(
            metric_gain=float(data.get("metric_gain", 0.5)),
            stability=float(data.get("stability", 0.5)),
            generalization=float(data.get("generalization", 0.5)),
            efficiency=float(data.get("efficiency", 0.5)),
            safety_alignment=float(data.get("safety_alignment", 1.0)),
        )

        return JudgeFeedback(
            judge_model=model,
            scores=scores,
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.8)),
            flagged_issues=data.get("flagged_issues", []),
        )

    def _compute_disagreement(self, feedbacks: list[JudgeFeedback]) -> float:
        """Compute standard deviation of final scores across judges."""
        if len(feedbacks) < 2:
            return 0.0
        scores = [f.scores.final_score for f in feedbacks]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return math.sqrt(variance)

    def _classify_disagreement(self, sigma: float) -> DisagreementLevel:
        """Classify disagreement level from sigma."""
        if sigma < 0.10:
            return DisagreementLevel.LOW
        elif sigma < 0.20:
            return DisagreementLevel.MODERATE
        else:
            return DisagreementLevel.HIGH

    def _has_early_agreement(self, feedbacks: list[JudgeFeedback]) -> bool:
        """Check if judges already agree enough to stop early."""
        if len(feedbacks) < 2:
            return False
        sigma = self._compute_disagreement(feedbacks)
        return sigma < 0.06

    def _get_fallback_models(self, already_used: list[str]) -> list[str]:
        """Get additional models to try as fallback judges."""
        # Common capable models for evaluation
        fallback_pool = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-1.5-flash",
            "google/gemini-1.5-pro",
        ]
        return [m for m in fallback_pool if m not in already_used]
