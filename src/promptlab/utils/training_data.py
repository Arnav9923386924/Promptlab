"""Training Data Generator - Convert BSP + evaluation outputs to fine-tuning data.

Generates JSONL training data from:
- BSP (system prompt)
- Test prompts (user messages)
- High-scoring LLM outputs (assistant messages)

Filters by quality score threshold and exports in OpenAI fine-tuning format.
This is primarily local processing — zero API calls needed.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingExample:
    """A single training example in OpenAI format."""
    system: str
    user: str
    assistant: str
    quality_score: float = 0.0
    source_test_id: str = ""
    metadata: dict = field(default_factory=dict)

    def to_openai_format(self) -> dict:
        """Convert to OpenAI fine-tuning JSONL format."""
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ]
        }

    def to_alpaca_format(self) -> dict:
        """Convert to Alpaca fine-tuning format."""
        return {
            "instruction": self.user,
            "input": "",
            "output": self.assistant,
            "system": self.system,
        }

    def to_sharegpt_format(self) -> dict:
        """Convert to ShareGPT format (used by many fine-tuning tools)."""
        return {
            "conversations": [
                {"from": "system", "value": self.system},
                {"from": "human", "value": self.user},
                {"from": "gpt", "value": self.assistant},
            ]
        }


@dataclass
class TrainingDataset:
    """A complete training dataset."""
    examples: list[TrainingExample] = field(default_factory=list)
    bsp_version: str = ""
    bsp_hash: str = ""
    created_at: str = ""
    min_quality_score: float = 0.0
    format: str = "openai"  # openai, alpaca, sharegpt

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def total_examples(self) -> int:
        return len(self.examples)

    @property
    def average_quality(self) -> float:
        if not self.examples:
            return 0.0
        return sum(e.quality_score for e in self.examples) / len(self.examples)

    @property
    def total_tokens_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        total_chars = sum(
            len(e.system) + len(e.user) + len(e.assistant)
            for e in self.examples
        )
        return total_chars // 4


class TrainingDataGenerator:
    """Generates fine-tuning datasets from BSP and evaluation outputs.
    
    Zero API calls — purely local processing.
    
    Workflow:
    1. Load BSP and test outputs (from evaluation runs)
    2. Filter by quality score threshold
    3. Export to JSONL in chosen format
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        min_quality_score: float = 0.7,
        output_format: str = "openai",
    ):
        """Initialize training data generator.
        
        Args:
            project_root: Project root directory
            min_quality_score: Minimum quality score to include (0.0-1.0)
            output_format: Export format: "openai", "alpaca", "sharegpt"
        """
        self.project_root = project_root or Path.cwd()
        self.min_quality_score = min_quality_score
        self.output_format = output_format
        self.output_dir = self.project_root / ".promptlab" / "training_data"

    def generate_from_outputs(
        self,
        bsp: str,
        outputs: list[dict],
        overall_score: float = 0.0,
        bsp_version: str = "1.0.0",
    ) -> TrainingDataset:
        """Generate training data from evaluation outputs.
        
        Args:
            bsp: The BSP text (becomes system prompt)
            outputs: List of test outputs [{prompt, response, test_id}, ...]
            overall_score: Overall evaluation score (used as quality proxy)
            bsp_version: BSP version string
            
        Returns:
            TrainingDataset with filtered examples
        """
        bsp_hash = hashlib.md5(bsp.encode()).hexdigest()[:12]
        examples = []

        for out in outputs:
            prompt = out.get("prompt", "").strip()
            response = out.get("response", "").strip()
            test_id = out.get("test_id", "")

            # Skip empty or error responses
            if not prompt or not response:
                continue
            if response.startswith("ERROR:") or response.startswith("[Error"):
                continue

            # Use per-test score if available, otherwise use overall score
            score = out.get("score", overall_score)

            # Filter by quality
            if score < self.min_quality_score:
                continue

            examples.append(TrainingExample(
                system=bsp,
                user=prompt,
                assistant=response,
                quality_score=score,
                source_test_id=test_id,
                metadata={
                    "bsp_version": bsp_version,
                    "bsp_hash": bsp_hash,
                },
            ))

        return TrainingDataset(
            examples=examples,
            bsp_version=bsp_version,
            bsp_hash=bsp_hash,
            min_quality_score=self.min_quality_score,
            format=self.output_format,
        )

    def generate_from_conversations(
        self,
        bsp: str,
        conversations: list[list[dict]],
        min_score: float = 0.7,
    ) -> TrainingDataset:
        """Generate training data from multi-turn conversations.
        
        Each conversation becomes multiple training examples
        (each user-assistant pair is one example).
        
        Args:
            bsp: BSP text
            conversations: List of conversations, each a list of {role, content} dicts
            min_score: Minimum score threshold
            
        Returns:
            TrainingDataset
        """
        examples = []
        bsp_hash = hashlib.md5(bsp.encode()).hexdigest()[:12]

        for conv_idx, conversation in enumerate(conversations):
            # Build context for multi-turn
            context = []
            for msg in conversation:
                if msg["role"] == "user":
                    context.append(msg["content"])
                elif msg["role"] == "assistant":
                    # Create a training example for each assistant response
                    # User input is the accumulated context
                    user_input = "\n\n".join(context) if len(context) > 1 else context[-1] if context else ""

                    examples.append(TrainingExample(
                        system=bsp,
                        user=user_input,
                        assistant=msg["content"],
                        quality_score=min_score,
                        source_test_id=f"conv_{conv_idx}",
                    ))
                    context.append(msg["content"])

        return TrainingDataset(
            examples=examples,
            bsp_hash=bsp_hash,
            min_quality_score=min_score,
            format=self.output_format,
        )

    def export_jsonl(
        self,
        dataset: TrainingDataset,
        filename: Optional[str] = None,
    ) -> Path:
        """Export dataset to JSONL file.
        
        Args:
            dataset: Training dataset to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the exported file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_{dataset.format}_{timestamp}.jsonl"

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            for example in dataset.examples:
                if dataset.format == "openai":
                    data = example.to_openai_format()
                elif dataset.format == "alpaca":
                    data = example.to_alpaca_format()
                elif dataset.format == "sharegpt":
                    data = example.to_sharegpt_format()
                else:
                    data = example.to_openai_format()

                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return output_path

    def export_json(
        self,
        dataset: TrainingDataset,
        filename: Optional[str] = None,
    ) -> Path:
        """Export dataset to JSON file (for preview/debugging).
        
        Args:
            dataset: Training dataset to export
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_preview_{timestamp}.json"

        output_path = self.output_dir / filename

        data = {
            "metadata": {
                "total_examples": dataset.total_examples,
                "average_quality": round(dataset.average_quality, 4),
                "estimated_tokens": dataset.total_tokens_estimate,
                "bsp_version": dataset.bsp_version,
                "bsp_hash": dataset.bsp_hash,
                "min_quality_score": dataset.min_quality_score,
                "format": dataset.format,
                "created_at": dataset.created_at,
            },
            "examples": [
                {
                    "system": e.system[:100] + "..." if len(e.system) > 100 else e.system,
                    "user": e.user,
                    "assistant": e.assistant,
                    "quality_score": e.quality_score,
                    "source_test_id": e.source_test_id,
                }
                for e in dataset.examples
            ],
        }

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return output_path

    def validate_dataset(self, dataset: TrainingDataset) -> dict:
        """Validate a training dataset for common issues.
        
        Returns:
            Dict with validation results and warnings
        """
        warnings = []
        errors = []

        if dataset.total_examples == 0:
            errors.append("Dataset is empty — no examples passed quality filter")

        if dataset.total_examples < 10:
            warnings.append(f"Very small dataset ({dataset.total_examples} examples). Recommended: 50+ for fine-tuning")

        if dataset.total_examples > 0:
            # Check for duplicates
            seen = set()
            duplicates = 0
            for e in dataset.examples:
                key = (e.user.strip()[:100], e.assistant.strip()[:100])
                if key in seen:
                    duplicates += 1
                seen.add(key)

            if duplicates > 0:
                warnings.append(f"{duplicates} duplicate examples found")

            # Check for very short responses
            short_responses = sum(1 for e in dataset.examples if len(e.assistant) < 20)
            if short_responses > dataset.total_examples * 0.3:
                warnings.append(f"{short_responses} examples have very short responses (<20 chars)")

            # Check for very long system prompts
            if dataset.examples and len(dataset.examples[0].system) > 2000:
                warnings.append("System prompt is very long (>2000 chars). This may increase fine-tuning cost")

            # Estimate cost
            tokens = dataset.total_tokens_estimate
            estimated_cost_gpt35 = (tokens / 1000) * 0.008  # GPT-3.5 fine-tuning rate
            if estimated_cost_gpt35 > 10:
                warnings.append(f"Estimated fine-tuning cost (GPT-3.5): ${estimated_cost_gpt35:.2f}")

        return {
            "valid": len(errors) == 0,
            "total_examples": dataset.total_examples,
            "average_quality": round(dataset.average_quality, 4),
            "estimated_tokens": dataset.total_tokens_estimate,
            "errors": errors,
            "warnings": warnings,
        }

    def format_report(self, dataset: TrainingDataset) -> str:
        """Format dataset info as readable report."""
        validation = self.validate_dataset(dataset)

        lines = []
        lines.append("Training Data Report")
        lines.append("=" * 40)
        lines.append(f"Total Examples: {dataset.total_examples}")
        lines.append(f"Average Quality: {dataset.average_quality:.2f}")
        lines.append(f"Estimated Tokens: {dataset.total_tokens_estimate:,}")
        lines.append(f"Format: {dataset.format}")
        lines.append(f"BSP Version: {dataset.bsp_version}")
        lines.append(f"Min Quality Score: {dataset.min_quality_score}")
        lines.append("")

        if validation["errors"]:
            lines.append("ERRORS:")
            for err in validation["errors"]:
                lines.append(f"  ✗ {err}")

        if validation["warnings"]:
            lines.append("WARNINGS:")
            for warn in validation["warnings"]:
                lines.append(f"  ⚠ {warn}")

        if not validation["errors"] and not validation["warnings"]:
            lines.append("✓ Dataset looks good!")

        return "\n".join(lines)
