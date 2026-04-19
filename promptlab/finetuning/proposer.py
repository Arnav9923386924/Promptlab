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


_RETRY_SUFFIX = """

IMPORTANT OUTPUT RULES:
- Return ONLY a single valid JSON object.
- Do not use markdown fences.
- Ensure all braces are closed.
- Keep rationale to <= 2 short sentences.
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
            prompt=prompt,
            fallback_models=[],
            model=model,
            system_prompt=_PROPOSAL_SYSTEM_PROMPT,
            max_tokens=1600,
        )

        # Parse and validate; retry once if model output is truncated/malformed.
        try:
            return self._parse_proposal(result.text)
        except ValueError as first_error:
            logger.info("First FT proposal parse failed, retrying once: %s", first_error)
            retry_prompt = prompt + _RETRY_SUFFIX
            retry_result = await self.llm_runner.complete_with_fallback(
                prompt=retry_prompt,
                fallback_models=[],
                model=model,
                system_prompt=_PROPOSAL_SYSTEM_PROMPT,
                max_tokens=2600,
            )
            return self._parse_proposal(retry_result.text)

    def _extract_json_object(self, raw_response: str) -> str:
        """Extract a best-effort JSON object from model output.

        Handles markdown fences, leading prose, and truncated closing braces.
        """
        content = raw_response.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        # Prefer extracting from the first "{" to the matching closing brace.
        start = content.find("{")
        if start == -1:
            return content

        chunk = content[start:]
        brace_depth = 0
        end_index = None
        for i, ch in enumerate(chunk):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    end_index = i
                    break

        # Fully balanced JSON object found.
        if end_index is not None:
            return chunk[: end_index + 1]

        # Truncated output: close any missing braces.
        if brace_depth > 0:
            return chunk + ("}" * brace_depth)

        return chunk

    def _parse_json_loose(self, content: str) -> dict:
        """Parse JSON with minor repair for common LLM formatting issues."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Remove trailing commas before object/array close: ",}" -> "}" and ",]" -> "]"
            repaired = content.replace(",}", "}").replace(",]", "]")
            return json.loads(repaired)

    def _parse_proposal(self, raw_response: str) -> CandidateConfig:
        """Parse LLM response into a validated CandidateConfig."""
        content = self._extract_json_object(raw_response)

        try:
            data = self._parse_json_loose(content)
        except json.JSONDecodeError as e:
            logger.debug("Raw response from LLM: %s", raw_response)
            logger.debug("Processed content for parse: %s", content)
            raise ValueError(f"LLM returned invalid JSON for config proposal: {e}")

        required_fields = [
            "learning_rate",
            "lora_rank",
            "lora_alpha",
            "num_epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "warmup_ratio",
            "weight_decay",
            "scheduler_type",
            "max_seq_length",
            "rationale",
        ]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(
                "LLM proposal missing required fields: " + ", ".join(missing_fields)
            )

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
