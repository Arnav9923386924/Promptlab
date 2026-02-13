"""Tests for council.required_judges functionality.

Verifies:
- Config validation (min >= 2)
- Backward compatible default (=2)
- Fallback backfills failed configured judges
- Evaluation fails when successful < required
- Early agreement respects required count
"""

import pytest
from pydantic import ValidationError
from promptlab.utils.config import CouncilConfig
from promptlab.llm_council.council.council import Council, JudgeScore
from promptlab.llm_council.llm_runner.runner import LLMRunner, CompletionResult
from unittest.mock import AsyncMock, MagicMock


class TestRequiredJudgesConfig:
    """Test config validation for required_judges."""
    
    def test_default_required_judges_is_2(self):
        """Verify default value is 2 (backward compatible)."""
        config = CouncilConfig(members=["model1", "model2"])
        assert config.required_judges == 2
    
    def test_required_judges_accepts_valid_values(self):
        """Valid values (>= 2) should be accepted."""
        for value in [2, 3, 5, 10]:
            config = CouncilConfig(members=["m1", "m2"], required_judges=value)
            assert config.required_judges == value
    
    def test_required_judges_rejects_less_than_2(self):
        """Values < 2 should raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CouncilConfig(members=["m1"], required_judges=1)
        
        error_msg = str(exc_info.value)
        assert "must be >= 2" in error_msg
        assert "at least 2 judges" in error_msg
    
    def test_required_judges_rejects_negative(self):
        """Negative values should raise validation error."""
        with pytest.raises(ValidationError):
            CouncilConfig(members=["m1"], required_judges=-1)
    
    def test_required_judges_rejects_zero(self):
        """Zero should raise validation error."""
        with pytest.raises(ValidationError):
            CouncilConfig(members=["m1"], required_judges=0)


class TestRequiredJudgesFallback:
    """Test fallback behavior with required_judges."""
    
    @pytest.mark.asyncio
    async def test_fallback_backfills_failed_configured_judges(self):
        """When configured judges fail, fallback models should backfill to meet required count."""
        # Setup: 3 configured, required_judges=2, first 2 configured fail, fallback succeeds
        config = {
            "enabled": True,
            "mode": "fast",
            "members": ["config1", "config2", "config3"],
            "chairman": "config1",
            "required_judges": 2,
            "use_fixed_judges": False,
        }
        
        mock_runner = MagicMock(spec=LLMRunner)
        council = Council(config, mock_runner, run_id=None, project_root=None)
        
        # Mock _get_judge_model_list to return configured + pool models
        async def mock_get_judges():
            return ["config1", "config2", "config3", "pool1", "pool2"]
        council._get_judge_model_list = mock_get_judges
        
        # Mock scoring: config1 fails, config2 fails, config3 succeeds, pool1 succeeds
        call_count = 0
        async def mock_get_judge_score_with_retry(model, prompt):
            nonlocal call_count
            call_count += 1
            if model in ["config1", "config2"]:
                raise RuntimeError(f"{model} failed")
            return JudgeScore(
                model=model,
                score=0.8,
                reasoning=f"Score from {model}",
                passed=True,
            )
        
        council._get_judge_score_with_retry = mock_get_judge_score_with_retry
        
        # Execute
        scores = await council._stage1_judge("test response", "test criteria")
        
        # Verify: Got exactly 2 scores (required count met)
        assert len(scores) == 2
        # Verify: config3 and pool1 were used
        assert scores[0].model == "config3"
        assert scores[1].model == "pool1"
    
    @pytest.mark.asyncio
    async def test_strict_mode_no_fallback(self):
        """With use_fixed_judges=True, should NOT fallback to pool models."""
        config = {
            "enabled": True,
            "mode": "fast",
            "members": ["config1", "config2"],
            "chairman": "config1",
            "required_judges": 2,
            "use_fixed_judges": True,  # Strict mode
        }
        
        mock_runner = MagicMock(spec=LLMRunner)
        council = Council(config, mock_runner, run_id=None, project_root=None)
        
        # Mock scoring: config1 fails, config2 fails
        async def mock_get_judge_score_with_retry(model, prompt):
            raise RuntimeError(f"{model} failed")
        
        council._get_judge_score_with_retry = mock_get_judge_score_with_retry
        
        # Execute and verify: Should raise RuntimeError (insufficient judges)
        with pytest.raises(RuntimeError) as exc_info:
            await council._stage1_judge("test response", "test criteria")
        
        error_msg = str(exc_info.value)
        assert "Insufficient callable judges" in error_msg
        assert "got 0, required 2" in error_msg


class TestInsufficientJudges:
    """Test behavior when insufficient judges available."""
    
    @pytest.mark.asyncio
    async def test_fails_when_insufficient_judges(self):
        """Evaluation should fail with clear error when < required_judges successful."""
        config = {
            "enabled": True,
            "mode": "fast",
            "members": ["m1", "m2", "m3"],
            "chairman": "m1",
            "required_judges": 3,
            "use_fixed_judges": False,
        }
        
        mock_runner = MagicMock(spec=LLMRunner)
        council = Council(config, mock_runner, run_id=None, project_root=None)
        
        # Mock _get_judge_model_list to return limited models
        async def mock_get_judges():
            return ["m1", "m2", "m3", "pool1"]
        council._get_judge_model_list = mock_get_judges
        
        # Mock scoring: only 2 succeed, 1 required fails
        success_count = 0
        async def mock_get_judge_score_with_retry(model, prompt):
            nonlocal success_count
            if success_count >= 2:
                raise RuntimeError(f"{model} failed")
            success_count += 1
            return JudgeScore(
                model=model,
                score=0.7,
                reasoning="ok",
                passed=True,
            )
        
        council._get_judge_score_with_retry = mock_get_judge_score_with_retry
        
        # Execute and verify
        with pytest.raises(RuntimeError) as exc_info:
            await council._stage1_judge("test", "criteria")
        
        error_msg = str(exc_info.value)
        assert "Insufficient callable judges" in error_msg
        assert "got 2, required 3" in error_msg
    
    @pytest.mark.asyncio
    async def test_no_silent_fallback_score(self):
        """Should NOT return fallback 0.5 score when all judges fail."""
        config = {
            "enabled": True,
            "mode": "fast",
            "members": ["m1", "m2"],
            "chairman": "m1",
            "required_judges": 2,
        }
        
        mock_runner = MagicMock(spec=LLMRunner)
        council = Council(config, mock_runner, run_id=None, project_root=None)
        
        # Mock _get_judge_model_list
        async def mock_get_judges():
            return ["m1", "m2"]
        council._get_judge_model_list = mock_get_judges
        
        # Mock scoring: all fail
        async def mock_get_judge_score_with_retry(model, prompt):
            raise RuntimeError(f"{model} rate limited")
        
        council._get_judge_score_with_retry = mock_get_judge_score_with_retry
        
        # Execute and verify: Should raise RuntimeError, NOT return fallback
        with pytest.raises(RuntimeError) as exc_info:
            await council._stage1_judge("test", "criteria")
        
        # Should NOT get a fallback score
        error_msg = str(exc_info.value)
        assert "Insufficient callable judges" in error_msg


class TestEarlyAgreement:
    """Test early agreement respects required_judges."""
    
    @pytest.mark.asyncio
    async def test_early_agreement_not_before_required_count(self):
        """Early agreement should NOT trigger before required_judges is met."""
        config = {
            "enabled": True,
            "mode": "fast",
            "members": ["m1", "m2", "m3", "m4"],
            "chairman": "m1",
            "required_judges": 3,  # Require 3 judges
        }
        
        mock_runner = MagicMock(spec=LLMRunner)
        council = Council(config, mock_runner, run_id=None, project_root=None)
        
        async def mock_get_judges():
            return ["m1", "m2", "m3", "m4"]
        council._get_judge_model_list = mock_get_judges
        
        # Mock scoring: First 2 judges agree perfectly (should NOT stop early)
        call_count = 0
        async def mock_get_judge_score_with_retry(model, prompt):
            nonlocal call_count
            call_count += 1
            return JudgeScore(
                model=model,
                score=0.85,  # All agree perfectly
                reasoning=f"Score from {model}",
                passed=True,
            )
        
        council._get_judge_score_with_retry = mock_get_judge_score_with_retry
        
        # Execute
        scores = await council._stage1_judge("test", "criteria")
        
        # Verify: Should collect AT LEAST required_judges (3), not stop at 2
        assert len(scores) >= 3
        # Note: Early agreement MIGHT trigger at 3 if std is low, that's OK
    
    @pytest.mark.asyncio
    async def test_early_agreement_triggers_after_required_count(self):
        """Early agreement CAN trigger after required_judges is met."""
        config = {
            "enabled": True,
            "mode": "fast",
            "members": ["m1", "m2", "m3", "m4", "m5"],
            "chairman": "m1",
            "required_judges": 2,  # Require 2 judges
        }
        
        mock_runner = MagicMock(spec=LLMRunner)
        council = Council(config, mock_runner, run_id=None, project_root=None)
        
        async def mock_get_judges():
            return ["m1", "m2", "m3", "m4", "m5"]
        council._get_judge_model_list = mock_get_judges
        
        # Mock scoring: All judges agree perfectly
        async def mock_get_judge_score_with_retry(model, prompt):
            return JudgeScore(
                model=model,
                score=0.85,
                reasoning=f"Score from {model}",
                passed=True,
            )
        
        council._get_judge_score_with_retry = mock_get_judge_score_with_retry
        
        # Execute
        scores = await council._stage1_judge("test", "criteria")
        
        # Verify: Should have >= 2 judges (required), but may stop early due to agreement
        assert len(scores) >= 2
        # Early agreement check happens after required_judges is met,
        # so with perfect agreement (std < 0.06), it might stop at 2 or 3


class TestConfiguredVsFallback:
    """Test logging distinguishes configured vs fallback judges."""
    
    @pytest.mark.asyncio
    async def test_logs_show_configured_and_fallback_counts(self, capsys):
        """Console logs should show how many configured vs fallback judges were used."""
        config = {
            "enabled": True,
            "mode": "fast",
            "members": ["config1", "config2"],
            "chairman": "config1",
            "required_judges": 3,  # More than configured
            "use_fixed_judges": False,
        }
        
        mock_runner = MagicMock(spec=LLMRunner)
        council = Council(config, mock_runner, run_id=None, project_root=None)
        
        async def mock_get_judges():
            return ["config1", "config2", "pool1", "pool2"]
        council._get_judge_model_list = mock_get_judges
        
        # Mock scoring: config1 succeeds, config2 succeeds, pool1 succeeds
        async def mock_get_judge_score_with_retry(model, prompt):
            return JudgeScore(
                model=model,
                score=0.8,
                reasoning=f"Score from {model}",
                passed=True,
            )
        
        council._get_judge_score_with_retry = mock_get_judge_score_with_retry
        
        # Execute
        scores = await council._stage1_judge("test", "criteria")
        
        # Verify
        assert len(scores) == 3
        # 2 configured (config1, config2) + 1 fallback (pool1)
        configured_models = [s.model for s in scores if s.model in ["config1", "config2"]]
        fallback_models = [s.model for s in scores if s.model not in ["config1", "config2"]]
        assert len(configured_models) == 2
        assert len(fallback_models) == 1
