"""Hyperparameter config proposal engine.

Uses LLM reasoning (via LLMRunner) to propose the next candidate
configuration within the search space, informed by run history.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from promptlab.finetuning.models import (
    CandidateConfig,
    HardwareProfile,
    RunHistoryEntry,
    SearchSpace,
)
from promptlab.llm_council.llm_runner.runner import LLMRunner

logger = logging.getLogger(__name__)


_PROPOSAL_SYSTEM_PROMPT = """\
You are a hyperparameter optimization expert for LLM fine-tuning.

Given:
- A search space with allowed ranges for each hyperparameter
- History of previous runs (configs + results)
- Hardware constraints
- The tuning goal

Propose the SINGLE BEST next hyperparameter configuration to try.

Rules:
1. Stay within the search space bounds.
2. Do NOT repeat a config that already failed with the same parameters.
3. Prioritize generalization over metric spikes.
4. Respect hardware constraints (VRAM, max batch size).
5. Provide a brief rationale for each choice.

Respond with ONLY a JSON object matching this schema:
{
  "learning_rate": <float>,
  "lora_rank": <int>,
  "lora_alpha": <int>,
  "num_epochs": <int>,
  "batch_size": <int>,
  "gradient_accumulation_steps": <int>,
  "warmup_ratio": <float>,
  "weight_decay": <float>,
  "scheduler_type": "<string>",
  "max_seq_length": <int>,
  "rationale": "<string explaining your choices>"
}
"""


class HyperparamProposer:
    """Proposes next hyperparameter configs using LLM reasoning."""

    def __init__(
        self,
        llm_runner: LLMRunner,
        search_space: SearchSpace,
        hardware_profile: HardwareProfile,
        tuning_goal: str = "",
        task_domain: str = "",
        base_model: str = "",
    ):
        self.llm_runner = llm_runner
        self.search_space = search_space
        self.hardware = hardware_profile
        self.tuning_goal = tuning_goal
        self.task_domain = task_domain
        self.base_model = base_model

    def _build_proposal_prompt(
        self,
        run_history: list[RunHistoryEntry],
    ) -> str:
        """Build the user prompt for config proposal."""
        parts = [
            f"Tuning goal: {self.tuning_goal}",
            f"Task domain: {self.task_domain}",
            f"Base model: {self.base_model}",
            "",
            "=== SEARCH SPACE ===",
            self.search_space.model_dump_json(indent=2),
            "",
            "=== HARDWARE CONSTRAINTS ===",
            self.hardware.model_dump_json(indent=2),
            "",
        ]

        if run_history:
            parts.append("=== RUN HISTORY (most recent first) ===")
            for entry in reversed(run_history[-10:]):  # Last 10 runs
                parts.append(json.dumps(entry.model_dump(), indent=2))
            parts.append("")
        else:
            parts.append("=== RUN HISTORY ===")
            parts.append("No previous runs. This is the first configuration.")
            parts.append("")

        parts.append("Propose the next configuration as JSON:")
        return "\n".join(parts)

    async def propose(
        self,
        run_history: list[RunHistoryEntry] | None = None,
        model: Optional[str] = None,
    ) -> CandidateConfig:
        """Propose the next hyperparameter config.

        Args:
            run_history: Previous run results to learn from.
            model: Specific model to use for proposal.

        Returns:
            CandidateConfig with proposed parameters.

        Raises:
            ValueError: If the LLM returns unparseable JSON or invalid values.
        """
        history = run_history or []
        prompt = self._build_proposal_prompt(history)

        # Get LLM completion
        result = await self.llm_runner.complete_with_fallback(
            system_prompt=_PROPOSAL_SYSTEM_PROMPT,
            user_prompt=prompt,
            preferred_model=model,
        )

        # Parse and validate
        return self._parse_proposal(result.content)

    def _parse_proposal(self, raw_response: str) -> CandidateConfig:
        """Parse LLM response into a validated CandidateConfig."""
        # Strip markdown fences if present
        content = raw_response.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON for config proposal: {e}")

        # Clamp values to search space
        ss = self.search_space
        data["learning_rate"] = max(
            ss.learning_rate["min"],
            min(ss.learning_rate["max"], float(data.get("learning_rate", 2e-4))),
        )
        data["lora_rank"] = max(
            ss.lora_rank["min"],
            min(ss.lora_rank["max"], int(data.get("lora_rank", 16))),
        )
        data["lora_alpha"] = max(
            ss.lora_alpha["min"],
            min(ss.lora_alpha["max"], int(data.get("lora_alpha", 32))),
        )
        data["num_epochs"] = max(
            ss.num_epochs["min"],
            min(ss.num_epochs["max"], int(data.get("num_epochs", 3))),
        )

        if data.get("scheduler_type") not in ss.scheduler_types:
            data["scheduler_type"] = ss.scheduler_types[0]

        if data.get("batch_size") not in ss.batch_sizes:
            # Pick the closest allowed batch size
            data["batch_size"] = min(
                ss.batch_sizes, key=lambda b: abs(b - int(data.get("batch_size", 4)))
            )

        if data.get("gradient_accumulation_steps") not in ss.gradient_accumulation_steps:
            data["gradient_accumulation_steps"] = min(
                ss.gradient_accumulation_steps,
                key=lambda g: abs(g - int(data.get("gradient_accumulation_steps", 1))),
            )

        # Respect hardware VRAM constraint
        hw = self.hardware
        if hw.max_batch_size and data["batch_size"] > hw.max_batch_size:
            data["batch_size"] = hw.max_batch_size

        return CandidateConfig(**data)
