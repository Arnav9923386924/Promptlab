"""Fine-tuning hyperparameter governance module.

Provides the LLM Council-based governance system for
proposing, evaluating, and selecting hyperparameter
configurations for fine-tuning pipelines.

This module is fully independent of the BSP workflow.
"""

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
    JudgeFeedback,
    NextStep,
    NextStepAction,
    PrioritizedRecommendation,
    RunArtifact,
    RunHistoryEntry,
    SearchSpace,
    TrainingMetrics,
)
from promptlab.finetuning.governance import HyperparamGovernanceCouncil
from promptlab.finetuning.proposer import HyperparamProposer
from promptlab.finetuning.evaluator import RunEvaluator
from promptlab.finetuning.chairman import ChairmanSynthesizer

__all__ = [
    # Core orchestrator
    "HyperparamGovernanceCouncil",
    # Sub-components
    "HyperparamProposer",
    "RunEvaluator",
    "ChairmanSynthesizer",
    # Models
    "CandidateConfig",
    "ChairmanVerdict",
    "DatasetProfile",
    "Decision",
    "DecisionVerdict",
    "DimensionScores",
    "DisagreementLevel",
    "EvaluationMetadata",
    "GovernanceOutput",
    "HardwareProfile",
    "JudgeFeedback",
    "NextStep",
    "NextStepAction",
    "PrioritizedRecommendation",
    "RunArtifact",
    "RunHistoryEntry",
    "SearchSpace",
    "TrainingMetrics",
]
