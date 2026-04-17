"""Fine-tuning hyperparameter governance models.

Pydantic models for the entire governance JSON schema:
candidate configs, evaluation results, dimension scores,
judge feedback, and decision summaries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────

class DecisionVerdict(str, Enum):
    """Final verdict for a governance decision."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL_PASS = "conditional_pass"


class DisagreementLevel(str, Enum):
    """Level of disagreement between judges."""
    LOW = "low"          # σ < 0.10
    MODERATE = "moderate"  # 0.10 ≤ σ < 0.20
    HIGH = "high"        # σ ≥ 0.20


class NextStepAction(str, Enum):
    """Recommended next action after a governance decision."""
    ADOPT_AND_CONTINUE = "adopt_and_continue"
    RETRY_WITH_MODIFICATION = "retry_with_modification"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"
    STOP = "stop"


# ─── Input Models ───────────────────────────────────────────────────────────

class DatasetProfile(BaseModel):
    """Profile of the fine-tuning dataset."""
    name: str
    num_samples: int
    avg_input_length: int = 0
    avg_output_length: int = 0
    task_types: list[str] = Field(default_factory=list)
    quality_score: Optional[float] = None
    split_ratio: dict[str, float] = Field(default_factory=lambda: {"train": 0.8, "eval": 0.2})


class HardwareProfile(BaseModel):
    """Hardware constraints for training."""
    gpu_type: str = "T4"
    gpu_count: int = 1
    vram_gb: float = 16.0
    max_batch_size: Optional[int] = None
    max_training_hours: float = 4.0


class SearchSpace(BaseModel):
    """Allowed search space for hyperparameters."""
    learning_rate: dict[str, float] = Field(
        default_factory=lambda: {"min": 1e-6, "max": 5e-4}
    )
    lora_rank: dict[str, int] = Field(
        default_factory=lambda: {"min": 4, "max": 64, "step": 4}
    )
    lora_alpha: dict[str, int] = Field(
        default_factory=lambda: {"min": 8, "max": 128, "step": 8}
    )
    num_epochs: dict[str, int] = Field(
        default_factory=lambda: {"min": 1, "max": 10}
    )
    warmup_ratio: dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 0.2}
    )
    weight_decay: dict[str, float] = Field(
        default_factory=lambda: {"min": 0.0, "max": 0.1}
    )
    scheduler_types: list[str] = Field(
        default_factory=lambda: ["cosine", "linear", "constant_with_warmup"]
    )
    batch_sizes: list[int] = Field(
        default_factory=lambda: [2, 4, 8, 16]
    )
    gradient_accumulation_steps: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8]
    )


class CandidateConfig(BaseModel):
    """A proposed hyperparameter configuration."""
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    extra_params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# ─── Run Artifacts ──────────────────────────────────────────────────────────

class TrainingMetrics(BaseModel):
    """Metrics from a completed training run."""
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float
    best_eval_step: int = 0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    loss_curve: list[dict[str, float]] = Field(default_factory=list)
    eval_scores: dict[str, float] = Field(default_factory=dict)


class RunArtifact(BaseModel):
    """Artifacts from a completed fine-tuning run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: CandidateConfig
    metrics: TrainingMetrics
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "completed"
    error: Optional[str] = None


class RunHistoryEntry(BaseModel):
    """Summary entry for run history."""
    run_id: str
    config_id: str
    final_eval_loss: float
    best_eval_loss: float
    verdict: Optional[str] = None
    timestamp: str = ""


# ─── Evaluation & Scoring ──────────────────────────────────────────────────

class DimensionScores(BaseModel):
    """Scores across the governance evaluation dimensions."""
    metric_gain: float = Field(ge=0.0, le=1.0, description="Improvement in target metric")
    stability: float = Field(ge=0.0, le=1.0, description="Training stability / loss monotonicity")
    generalization: float = Field(ge=0.0, le=1.0, description="Eval vs train gap quality")
    efficiency: float = Field(ge=0.0, le=1.0, description="Resource / time efficiency")
    safety_alignment: float = Field(ge=0.0, le=1.0, description="Safety and alignment checks")
    final_score: float = Field(ge=0.0, le=1.0, description="Weighted aggregate score")

    @classmethod
    def compute_final(
        cls,
        metric_gain: float,
        stability: float,
        generalization: float,
        efficiency: float,
        safety_alignment: float,
        weights: Optional[dict[str, float]] = None,
    ) -> "DimensionScores":
        """Compute final score with weighted dimensions."""
        w = weights or {
            "metric_gain": 0.30,
            "stability": 0.20,
            "generalization": 0.25,
            "efficiency": 0.10,
            "safety_alignment": 0.15,
        }
        final = (
            metric_gain * w["metric_gain"]
            + stability * w["stability"]
            + generalization * w["generalization"]
            + efficiency * w["efficiency"]
            + safety_alignment * w["safety_alignment"]
        )
        return cls(
            metric_gain=metric_gain,
            stability=stability,
            generalization=generalization,
            efficiency=efficiency,
            safety_alignment=safety_alignment,
            final_score=round(min(max(final, 0.0), 1.0), 4),
        )


class JudgeFeedback(BaseModel):
    """Feedback from a single judge."""
    judge_model: str
    scores: DimensionScores
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    flagged_issues: list[str] = Field(default_factory=list)


class ChairmanVerdict(BaseModel):
    """Chairman's synthesis verdict."""
    final_scores: DimensionScores
    confidence: float = Field(ge=0.0, le=1.0)
    synthesis_reasoning: str = ""
    tie_break_rationale: Optional[str] = None
    overridden_judges: list[str] = Field(default_factory=list)


# ─── Decision & Output ─────────────────────────────────────────────────────

class PrioritizedRecommendation(BaseModel):
    """A ranked recommendation for the next config."""
    priority: int
    parameter: str
    direction: str  # e.g., "increase", "decrease", "change_to"
    target_value: Optional[Any] = None
    rationale: str = ""


class NextStep(BaseModel):
    """Recommended next action."""
    action: NextStepAction
    suggested_config: Optional[CandidateConfig] = None
    recommendations: list[PrioritizedRecommendation] = Field(default_factory=list)
    rationale: str = ""


class EvaluationMetadata(BaseModel):
    """Metadata about the evaluation process."""
    run_id: str
    config_id: str
    judges_used: list[str] = Field(default_factory=list)
    judges_attempted: int = 0
    judges_succeeded: int = 0
    fallback_used: int = 0
    disagreement_level: DisagreementLevel = DisagreementLevel.LOW
    disagreement_sigma: float = 0.0
    chairman_invoked: bool = False
    chunking_used: bool = False
    evaluation_time_seconds: float = 0.0


class Decision(BaseModel):
    """Final governance decision for a run."""
    verdict: DecisionVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    scores: DimensionScores
    summary: str = ""
    safety_hard_fail: bool = False


class GovernanceOutput(BaseModel):
    """Top-level governance output — the complete JSON response.

    This is the strict schema that the governance council produces.
    """
    governance_version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    tuning_goal: str = ""
    task_domain: str = ""
    base_model: str = ""

    # The config that was evaluated
    evaluated_config: CandidateConfig

    # Evaluation details
    evaluation: EvaluationMetadata
    judge_feedback: list[JudgeFeedback] = Field(default_factory=list)
    chairman_verdict: Optional[ChairmanVerdict] = None

    # Decision
    decision: Decision

    # Next steps
    next_step: NextStep

    def to_json(self) -> str:
        """Serialize to strict JSON (no markdown, no prose)."""
        return self.model_dump_json(indent=2, exclude_none=True)
