"""Hyperparameter Governance Council — Main Orchestrator.

Implements the 8-step governance process:
1. Propose candidate config
2. (External) Execute fine-tuning
3. Multi-judge evaluation
4. Judge count enforcement
5. Agreement check
6. Chairman synthesis (on disagreement)
7. Chunking for large artifacts
8. Decision update & recommendation

This module does NOT execute training — it receives run artifacts
and governs the hyperparameter selection loop.
"""

from __future__ import annotations

import logging
from typing import Optional

from promptlab.finetuning.chairman import ChairmanSynthesizer
from promptlab.finetuning.evaluator import RunEvaluator
from promptlab.finetuning.models import (
    CandidateConfig,
    ChairmanVerdict,
    DatasetProfile,
    Decision,
    DecisionVerdict,
    DimensionScores,
    DisagreementLevel,
    EvaluationMetadata,
    GovernanceOutput,
    HardwareProfile,
    NextStep,
    NextStepAction,
    PrioritizedRecommendation,
    RunArtifact,
    RunHistoryEntry,
    SearchSpace,
)
from promptlab.finetuning.proposer import HyperparamProposer
from promptlab.llm_council.llm_runner.runner import LLMRunner

logger = logging.getLogger(__name__)


class HyperparamGovernanceCouncil:
    """Main orchestrator for the fine-tuning governance loop.

    Usage:
        council = HyperparamGovernanceCouncil(...)

        # Step 1: Get a proposed config
        config = await council.propose_config()

        # Step 2: Train externally, collect artifacts
        artifact = RunArtifact(config=config, metrics=..., ...)

        # Step 3: Evaluate and get governance decision
        output = await council.evaluate_run(artifact)
        print(output.to_json())
    """

    # ── Governance policy thresholds ──
    PASS_THRESHOLD: float = 0.65
    CONDITIONAL_PASS_THRESHOLD: float = 0.50
    SAFETY_HARD_FAIL_THRESHOLD: float = 0.30
    DISAGREEMENT_CHAIRMAN_THRESHOLD: float = 0.20  # σ trigger

    def __init__(
        self,
        llm_runner: LLMRunner,
        judge_models: list[str],
        search_space: SearchSpace | None = None,
        hardware_profile: HardwareProfile | None = None,
        dataset_profile: DatasetProfile | None = None,
        tuning_goal: str = "Minimize eval loss while maintaining generalization",
        task_domain: str = "general",
        base_model: str = "unsloth/Qwen2.5-1.5B",
        chairman_model: str | None = None,
        min_required_judges: int = 3,
        use_fixed_judges: bool = False,
        run_history: list[RunHistoryEntry] | None = None,
    ):
        self.llm_runner = llm_runner
        self.tuning_goal = tuning_goal
        self.task_domain = task_domain
        self.base_model = base_model
        self.run_history: list[RunHistoryEntry] = run_history or []

        self.search_space = search_space or SearchSpace()
        self.hardware = hardware_profile or HardwareProfile()
        self.dataset = dataset_profile or DatasetProfile(
            name="default", num_samples=0
        )

        # Sub-components
        self.proposer = HyperparamProposer(
            llm_runner=llm_runner,
            search_space=self.search_space,
            hardware_profile=self.hardware,
            tuning_goal=tuning_goal,
            task_domain=task_domain,
            base_model=base_model,
        )

        self.evaluator = RunEvaluator(
            llm_runner=llm_runner,
            judge_models=judge_models,
            min_required_judges=min_required_judges,
            use_fixed_judges=use_fixed_judges,
            tuning_goal=tuning_goal,
            base_model=base_model,
        )

        self.chairman = ChairmanSynthesizer(
            llm_runner=llm_runner,
            chairman_model=chairman_model,
            tuning_goal=tuning_goal,
        )

    # ── Step 1: Propose ──────────────────────────────────────────────────

    async def propose_config(
        self, model: Optional[str] = None
    ) -> CandidateConfig:
        """Propose the next hyperparameter config."""
        return await self.proposer.propose(
            run_history=self.run_history,
            model=model,
        )

    # ── Steps 3–8: Evaluate, Judge, Chairman, Decide ─────────────────────

    async def evaluate_run(
        self,
        artifact: RunArtifact,
        previous_best: Optional[RunArtifact] = None,
    ) -> GovernanceOutput:
        """Full governance evaluation of a completed training run.

        This implements steps 3-8 of the governance pipeline:
        3. Multi-judge evaluation
        4. Judge count enforcement (inside evaluator)
        5. Agreement check
        6. Chairman synthesis (if needed)
        7. Chunking (inside evaluator)
        8. Decision + recommendation

        Args:
            artifact: The completed run's artifacts.
            previous_best: The best previous run for comparison.

        Returns:
            GovernanceOutput with the complete JSON decision.
        """
        # ── Step 3+4: Multi-judge evaluation with enforcement ──
        feedbacks, eval_meta = await self.evaluator.evaluate(
            artifact, previous_best
        )

        # ── Step 5: Agreement check ──
        chairman_verdict: Optional[ChairmanVerdict] = None
        disagreement = DisagreementLevel(eval_meta["disagreement_level"])

        if disagreement == DisagreementLevel.HIGH:
            # ── Step 6: Chairman synthesis ──
            logger.info(
                f"High disagreement (σ={eval_meta['disagreement_sigma']:.3f})"
                " — invoking chairman"
            )
            try:
                chairman_verdict = await self.chairman.synthesize(
                    feedbacks,
                    config_summary=artifact.config.model_dump_json(),
                    metrics_summary=artifact.metrics.model_dump_json(),
                )
                eval_meta["chairman_invoked"] = True
            except Exception as e:
                logger.warning(f"Chairman LLM failed, using fallback: {e}")
                chairman_verdict = self.chairman.fallback_synthesis(feedbacks)
                eval_meta["chairman_invoked"] = True

        # ── Step 8: Decision ──
        final_scores = self._resolve_final_scores(feedbacks, chairman_verdict)
        decision = self._make_decision(final_scores, feedbacks)
        next_step = self._make_next_step(
            decision, artifact, final_scores, feedbacks
        )

        # Update run history
        self.run_history.append(
            RunHistoryEntry(
                run_id=artifact.run_id,
                config_id=artifact.config.config_id,
                final_eval_loss=artifact.metrics.final_eval_loss,
                best_eval_loss=artifact.metrics.best_eval_loss,
                verdict=decision.verdict.value,
                timestamp=artifact.timestamp,
            )
        )

        # ── Build output ──
        evaluation = EvaluationMetadata(**eval_meta)

        return GovernanceOutput(
            tuning_goal=self.tuning_goal,
            task_domain=self.task_domain,
            base_model=self.base_model,
            evaluated_config=artifact.config,
            evaluation=evaluation,
            judge_feedback=feedbacks,
            chairman_verdict=chairman_verdict,
            decision=decision,
            next_step=next_step,
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _resolve_final_scores(
        self,
        feedbacks: list,
        chairman_verdict: Optional[ChairmanVerdict],
    ) -> DimensionScores:
        """Resolve final scores: chairman verdict wins if present."""
        if chairman_verdict:
            return chairman_verdict.final_scores

        # No chairman — confidence-weighted average
        if not feedbacks:
            return DimensionScores.compute_final(0.5, 0.5, 0.5, 0.5, 1.0)

        total_w = sum(f.confidence for f in feedbacks) or 1.0
        dims = {}
        for dim in ["metric_gain", "stability", "generalization", "efficiency", "safety_alignment"]:
            dims[dim] = sum(
                getattr(f.scores, dim) * f.confidence for f in feedbacks
            ) / total_w

        return DimensionScores.compute_final(**dims)

    def _make_decision(
        self,
        scores: DimensionScores,
        feedbacks: list,
    ) -> Decision:
        """Make pass/fail decision based on scores and policy."""
        # Safety hard-fail
        if scores.safety_alignment < self.SAFETY_HARD_FAIL_THRESHOLD:
            return Decision(
                verdict=DecisionVerdict.FAIL,
                confidence=0.95,
                scores=scores,
                summary=(
                    f"SAFETY HARD-FAIL: safety_alignment={scores.safety_alignment:.3f} "
                    f"is below threshold {self.SAFETY_HARD_FAIL_THRESHOLD}."
                ),
                safety_hard_fail=True,
            )

        # Pass
        if scores.final_score >= self.PASS_THRESHOLD:
            confidence = min(
                0.95,
                0.6 + (scores.final_score - self.PASS_THRESHOLD) * 2,
            )
            return Decision(
                verdict=DecisionVerdict.PASS,
                confidence=round(confidence, 3),
                scores=scores,
                summary=(
                    f"PASS: final_score={scores.final_score:.3f} "
                    f"≥ threshold {self.PASS_THRESHOLD}. "
                    f"Generalization={scores.generalization:.3f}, "
                    f"Stability={scores.stability:.3f}."
                ),
            )

        # Conditional pass
        if scores.final_score >= self.CONDITIONAL_PASS_THRESHOLD:
            return Decision(
                verdict=DecisionVerdict.CONDITIONAL_PASS,
                confidence=round(0.4 + scores.final_score * 0.3, 3),
                scores=scores,
                summary=(
                    f"CONDITIONAL PASS: final_score={scores.final_score:.3f} "
                    f"in range [{self.CONDITIONAL_PASS_THRESHOLD}, "
                    f"{self.PASS_THRESHOLD}). Review recommended."
                ),
            )

        # Fail
        return Decision(
            verdict=DecisionVerdict.FAIL,
            confidence=round(0.5 + (self.CONDITIONAL_PASS_THRESHOLD - scores.final_score), 3),
            scores=scores,
            summary=(
                f"FAIL: final_score={scores.final_score:.3f} "
                f"< threshold {self.CONDITIONAL_PASS_THRESHOLD}. "
                f"Weakest dimensions: "
                + self._weakest_dims(scores)
            ),
        )

    def _weakest_dims(self, scores: DimensionScores) -> str:
        """Identify the weakest scoring dimensions."""
        dims = {
            "metric_gain": scores.metric_gain,
            "stability": scores.stability,
            "generalization": scores.generalization,
            "efficiency": scores.efficiency,
            "safety_alignment": scores.safety_alignment,
        }
        sorted_dims = sorted(dims.items(), key=lambda x: x[1])
        return ", ".join(f"{k}={v:.3f}" for k, v in sorted_dims[:2])

    def _make_next_step(
        self,
        decision: Decision,
        artifact: RunArtifact,
        scores: DimensionScores,
        feedbacks: list,
    ) -> NextStep:
        """Generate the next step recommendation."""
        recommendations: list[PrioritizedRecommendation] = []

        if decision.safety_hard_fail:
            return NextStep(
                action=NextStepAction.STOP,
                rationale=(
                    "Safety alignment critically low. "
                    "Review training data and model behavior before continuing."
                ),
            )

        if decision.verdict == DecisionVerdict.FAIL:
            # Build recommendations from weakness analysis
            if scores.stability < 0.5:
                recommendations.append(
                    PrioritizedRecommendation(
                        priority=1,
                        parameter="learning_rate",
                        direction="decrease",
                        target_value=artifact.config.learning_rate * 0.5,
                        rationale="Low stability suggests learning rate is too high.",
                    )
                )

            if scores.generalization < 0.5:
                recommendations.append(
                    PrioritizedRecommendation(
                        priority=2,
                        parameter="num_epochs",
                        direction="decrease",
                        target_value=max(1, artifact.config.num_epochs - 1),
                        rationale="Low generalization suggests overfitting.",
                    )
                )

            if scores.metric_gain < 0.4:
                recommendations.append(
                    PrioritizedRecommendation(
                        priority=3,
                        parameter="lora_rank",
                        direction="increase",
                        target_value=min(64, artifact.config.lora_rank * 2),
                        rationale="Low metric gain — model may need more capacity.",
                    )
                )

            return NextStep(
                action=NextStepAction.RETRY_WITH_MODIFICATION,
                recommendations=recommendations,
                rationale=f"Run failed with score {scores.final_score:.3f}. Apply recommended modifications.",
            )

        if decision.verdict == DecisionVerdict.CONDITIONAL_PASS:
            return NextStep(
                action=NextStepAction.RETRY_WITH_MODIFICATION,
                recommendations=recommendations,
                rationale=(
                    f"Conditional pass ({scores.final_score:.3f}). "
                    "Consider minor adjustments to push above threshold."
                ),
            )

        # PASS
        return NextStep(
            action=NextStepAction.ADOPT_AND_CONTINUE,
            rationale=(
                f"Config passed with score {scores.final_score:.3f}. "
                "Adopt this configuration and continue optimization."
            ),
        )
