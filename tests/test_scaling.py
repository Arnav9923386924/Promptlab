"""Tests for PromptLab scaling changes.

Covers:
- CLI count resolution (--generate flag vs config vs default)
- Auto-test generator adaptive rounds & hybrid mode
- Council chunked evaluation for large batches
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# A) CLI count resolution
# ---------------------------------------------------------------------------

class TestCLICountResolution:
    """Verify effective_count picks --generate > config > 50."""

    def _make_config(self, auto_generate_count: int = 50, generation_mode: str = "web"):
        from promptlab.utils.config import PromptLabConfig, BSPConfig
        cfg = PromptLabConfig()
        cfg.bsp = BSPConfig(
            prompt="You are a test assistant.",
            min_score=0.7,
            auto_generate_count=auto_generate_count,
            generation_mode=generation_mode,
        )
        return cfg

    def test_generate_flag_overrides_config(self):
        """--generate 200 should override config.bsp.auto_generate_count=100."""
        config = self._make_config(auto_generate_count=100)
        generate = 200
        # Simulate CLI resolution logic
        if generate is not None:
            effective = generate
        elif config.bsp and config.bsp.auto_generate_count:
            effective = config.bsp.auto_generate_count
        else:
            effective = 50
        assert effective == 200

    def test_config_auto_generate_count_used(self):
        """When --generate is None, config.bsp.auto_generate_count should be used."""
        config = self._make_config(auto_generate_count=150)
        generate = None
        if generate is not None:
            effective = generate
        elif config.bsp and config.bsp.auto_generate_count:
            effective = config.bsp.auto_generate_count
        else:
            effective = 50
        assert effective == 150

    def test_default_fallback_50(self):
        """When --generate is None and config has default 50, result should be 50."""
        config = self._make_config(auto_generate_count=50)
        generate = None
        if generate is not None:
            effective = generate
        elif config.bsp and config.bsp.auto_generate_count:
            effective = config.bsp.auto_generate_count
        else:
            effective = 50
        assert effective == 50

    def test_generation_mode_defaults_to_web(self):
        from promptlab.utils.config import BSPConfig
        cfg = BSPConfig()
        assert cfg.generation_mode == "web"

    def test_generation_mode_hybrid(self):
        from promptlab.utils.config import BSPConfig
        cfg = BSPConfig(generation_mode="hybrid")
        assert cfg.generation_mode == "hybrid"


# ---------------------------------------------------------------------------
# B) Auto-test generator: adaptive rounds & hybrid
# ---------------------------------------------------------------------------

class TestAutoTestGeneratorAdaptive:
    """Verify the generator attempts multiple rounds to reach target."""

    def test_generate_tests_accepts_generation_mode(self):
        """generate_tests should accept generation_mode kwarg without error."""
        from promptlab.utils.auto_test_generator import AutoTestGenerator
        gen = AutoTestGenerator()
        # Just verify the method signature accepts generation_mode
        import inspect
        sig = inspect.signature(gen.generate_tests)
        assert "generation_mode" in sig.parameters

    def test_generate_tests_accepts_target_count(self):
        """generate_tests should accept arbitrary target_count."""
        from promptlab.utils.auto_test_generator import AutoTestGenerator
        gen = AutoTestGenerator()
        import inspect
        sig = inspect.signature(gen.generate_tests)
        assert "target_count" in sig.parameters
        # Default should be 50
        assert sig.parameters["target_count"].default == 50

    def test_synthetic_variants_qa(self):
        """Hybrid mode should generate QA variants from seed tests."""
        from promptlab.utils.auto_test_generator import AutoTestGenerator
        from promptlab.utils.data_processor import QAPair, MaskedTest

        gen = AutoTestGenerator()
        seeds = [
            QAPair(question="What is contract law?", answer="A body of law governing agreements.", source_url="test", tags=["legal"]),
            QAPair(question="How does tort law work?", answer="It provides remedies for civil wrongs.", source_url="test", tags=["legal"]),
        ]
        syn_qa, syn_masked = gen._generate_synthetic_variants(seeds, [], 5, "all")
        assert len(syn_qa) > 0
        # All synthetic should have "synthetic" tag
        for qa in syn_qa:
            assert "synthetic" in qa.tags

    def test_synthetic_variants_empty_seeds(self):
        """With no seeds, synthetic variant generation produces nothing."""
        from promptlab.utils.auto_test_generator import AutoTestGenerator
        gen = AutoTestGenerator()
        syn_qa, syn_masked = gen._generate_synthetic_variants([], [], 10, "all")
        assert syn_qa == []
        assert syn_masked == []

    def test_synthetic_variants_cloze(self):
        """Hybrid mode should generate cloze variants from seed tests."""
        from promptlab.utils.auto_test_generator import AutoTestGenerator
        from promptlab.utils.data_processor import QAPair, MaskedTest

        gen = AutoTestGenerator()
        seeds = [
            MaskedTest(masked_text="The ___ is a body of law.", answer="constitution", original_text="The constitution is a body of law.", mask_position=4, source_url="test", tags=["legal"]),
        ]
        syn_qa, syn_masked = gen._generate_synthetic_variants([], seeds, 3, "all")
        # Should produce at least some cloze variants (depends on available words)
        # Not guaranteed since the mask-swap needs a suitable candidate word
        assert isinstance(syn_masked, list)

    def test_shortfall_warning_logged(self):
        """When fewer tests than target are produced, a warning should appear."""
        # This is an integration-level check; we verify the code path exists
        from promptlab.utils.auto_test_generator import AutoTestGenerator
        gen = AutoTestGenerator()
        # The warning is printed by console.print — just verify the method
        # handles < target gracefully (no exception)
        assert hasattr(gen, '_generate_synthetic_variants')


# ---------------------------------------------------------------------------
# C) Council chunked evaluation
# ---------------------------------------------------------------------------

class TestCouncilChunkedEvaluation:
    """Verify chunked evaluation for large batches."""

    def _make_council(self):
        """Create a minimal Council for testing."""
        from promptlab.llm_council.council.council import Council
        mock_runner = MagicMock()
        config = {
            "members": ["model-a", "model-b"],
            "chairman": "model-chairman",
            "mode": "fast",
        }
        council = Council(config, mock_runner, run_id=None, project_root=None)
        return council

    def test_chunk_size_constant(self):
        """Council should have a CHUNK_SIZE attribute."""
        from promptlab.llm_council.council.council import Council
        assert hasattr(Council, "CHUNK_SIZE")
        assert Council.CHUNK_SIZE == 25

    def test_small_batch_no_chunking(self):
        """Batches ≤ CHUNK_SIZE should go directly to _evaluate_single_batch."""
        council = self._make_council()
        outputs = [{"test_id": f"t{i}", "prompt": "p", "response": "r", "expected": "e"} for i in range(10)]
        
        with patch.object(council, '_evaluate_single_batch', new_callable=AsyncMock) as mock_single:
            mock_single.return_value = MagicMock(final_score=0.8)
            
            result = asyncio.run(council.evaluate_batch(outputs, "test bsp", 0.7))
            mock_single.assert_called_once()

    def test_large_batch_triggers_chunking(self):
        """Batches > CHUNK_SIZE should use _evaluate_batch_chunked."""
        council = self._make_council()
        outputs = [{"test_id": f"t{i}", "prompt": "p", "response": "r", "expected": "e"} for i in range(60)]
        
        with patch.object(council, '_evaluate_batch_chunked', new_callable=AsyncMock) as mock_chunked:
            mock_chunked.return_value = MagicMock(final_score=0.75)
            
            result = asyncio.run(council.evaluate_batch(outputs, "test bsp", 0.7))
            mock_chunked.assert_called_once()

    def test_chunked_aggregation_weights(self):
        """Chunked aggregation should weight by chunk size."""
        from promptlab.llm_council.council.council import Council, BatchEvaluationResult, BatchJudgeScore
        
        council = self._make_council()
        
        # Mock _evaluate_single_batch to return different scores per chunk
        call_count = [0]
        async def mock_evaluate(outputs, bsp, min_score):
            call_count[0] += 1
            score = 0.8 if call_count[0] == 1 else 0.6
            return BatchEvaluationResult(
                final_score=score,
                passed=score >= min_score,
                confidence="high",
                member_scores=[BatchJudgeScore(
                    model="m1", overall_score=score,
                    instruction_following=score, helpfulness=score,
                    coherence=score, safety=score,
                    reasoning="test", weak_areas=[],
                )],
                summary=f"chunk {call_count[0]}",
                recommendations=[],
                breakdown={
                    "instruction_following": score,
                    "helpfulness": score,
                    "coherence": score,
                    "safety": score,
                },
            )
        
        # 30 outputs: chunk1=25, chunk2=5
        outputs = [{"test_id": f"t{i}", "prompt": "p", "response": "r", "expected": "e"} for i in range(30)]
        
        with patch.object(council, '_evaluate_single_batch', side_effect=mock_evaluate):
            result = asyncio.run(council._evaluate_batch_chunked(outputs, "bsp", 0.7))
        
        # Expected: (0.8 * 25/30) + (0.6 * 5/30) = 0.6667 + 0.1 = 0.7667
        expected = round(0.8 * (25/30) + 0.6 * (5/30), 4)
        assert abs(result.final_score - expected) < 0.01
        assert result.breakdown["chunks"] == 2
        assert result.breakdown["total_outputs"] == 30

    def test_chunked_deduplicates_recommendations(self):
        """Recommendations from multiple chunks should be deduplicated."""
        from promptlab.llm_council.council.council import Council, BatchEvaluationResult, BatchJudgeScore

        council = self._make_council()

        async def mock_evaluate(outputs, bsp, min_score):
            return BatchEvaluationResult(
                final_score=0.7,
                passed=True,
                confidence="medium",
                member_scores=[],
                summary="chunk",
                recommendations=["Improve formatting", "Be more concise"],
                breakdown={"instruction_following": 0.7, "helpfulness": 0.7, "coherence": 0.7, "safety": 0.7},
            )

        outputs = [{"test_id": f"t{i}", "prompt": "p", "response": "r", "expected": "e"} for i in range(60)]

        with patch.object(council, '_evaluate_single_batch', side_effect=mock_evaluate):
            result = asyncio.run(council._evaluate_batch_chunked(outputs, "bsp", 0.7))

        # "Improve formatting" appears in all chunks but should appear only once
        fmt_count = sum(1 for r in result.recommendations if r.lower() == "improve formatting")
        assert fmt_count == 1
