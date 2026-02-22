"""Unit tests for PromptLab production-hardening changes.

Covers:
- env var config loading  (A)
- retry / backoff behaviour (B)
- council score parsing edge cases (C)
- evaluation history write consistency (C/D)
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# A) Config + env var expansion
# ---------------------------------------------------------------------------

class TestEnvVarExpansion:
    """Verify ${VAR} expansion works in config loading."""

    def test_expand_single_var(self):
        from promptlab.utils.config import _expand_env_vars
        os.environ["TEST_KEY_123"] = "secret-abc"
        assert _expand_env_vars("${TEST_KEY_123}") == "secret-abc"
        del os.environ["TEST_KEY_123"]

    def test_expand_missing_var_unchanged(self):
        from promptlab.utils.config import _expand_env_vars
        os.environ.pop("MISSING_KEY_XYZ", None)
        assert _expand_env_vars("${MISSING_KEY_XYZ}") == "${MISSING_KEY_XYZ}"

    def test_expand_inline(self):
        from promptlab.utils.config import _expand_env_vars
        os.environ["INLINE_VAR"] = "hello"
        result = _expand_env_vars("prefix-${INLINE_VAR}-suffix")
        assert result == "prefix-hello-suffix"
        del os.environ["INLINE_VAR"]

    def test_deep_expand(self):
        from promptlab.utils.config import _deep_expand
        os.environ["DK"] = "deep_val"
        tree = {"a": {"b": "${DK}"}, "c": ["${DK}", "plain"]}
        expanded = _deep_expand(tree)
        assert expanded["a"]["b"] == "deep_val"
        assert expanded["c"][0] == "deep_val"
        assert expanded["c"][1] == "plain"
        del os.environ["DK"]

    def test_load_config_loads_dotenv(self, tmp_path: Path):
        """Config loader should pick up .env next to promptlab.yaml."""
        from promptlab.utils.config import load_config

        (tmp_path / ".env").write_text("MY_TEST_API_KEY=from-dotenv\n", encoding="utf-8")
        (tmp_path / "promptlab.yaml").write_text(
            "version: 1\nmodels:\n  providers:\n    openrouter:\n      api_key: ${MY_TEST_API_KEY}\n",
            encoding="utf-8",
        )
        cfg = load_config(tmp_path / "promptlab.yaml")
        assert cfg.models.providers["openrouter"].api_key == "from-dotenv"
        os.environ.pop("MY_TEST_API_KEY", None)


# ---------------------------------------------------------------------------
# B) Retry / backoff behaviour
# ---------------------------------------------------------------------------

class TestCooldownTracker:
    """Test per-model cooldown tracker."""

    def test_initially_available(self):
        from promptlab.orchestrators.bsp_validator import _CooldownTracker
        ct = _CooldownTracker()
        assert ct.is_available("model-a") is True

    def test_mark_blocks_temporarily(self):
        from promptlab.orchestrators.bsp_validator import _CooldownTracker
        ct = _CooldownTracker()
        ct.mark("model-a", base_delay=0.05, jitter=0.0)
        assert ct.is_available("model-a") is False
        import time; time.sleep(0.1)
        assert ct.is_available("model-a") is True

    def test_other_models_unaffected(self):
        from promptlab.orchestrators.bsp_validator import _CooldownTracker
        ct = _CooldownTracker()
        ct.mark("model-a", base_delay=60.0, jitter=0.0)
        assert ct.is_available("model-b") is True


class TestRunTelemetry:
    """Test telemetry accounting."""

    def test_record_request(self):
        from promptlab.orchestrators.bsp_validator import RunTelemetry
        t = RunTelemetry(run_id="t1", start_time=1.0)
        t.record_request("m1", success=True, latency_ms=100)
        t.record_request("m1", success=False, is_429=True)
        assert t.total_requests == 2
        assert t.rate_limit_429s == 1
        assert t.per_model["m1"]["ok"] == 1
        assert t.per_model["m1"]["fail"] == 1

    def test_avg_latency(self):
        from promptlab.orchestrators.bsp_validator import RunTelemetry
        t = RunTelemetry(run_id="t2", start_time=1.0)
        t.record_request("m1", success=True, latency_ms=100)
        t.record_request("m1", success=True, latency_ms=200)
        assert t.avg_latency_ms == 150.0

    def test_save_creates_file(self, tmp_path: Path):
        from promptlab.orchestrators.bsp_validator import RunTelemetry
        import time
        t = RunTelemetry(run_id="test_run_1", start_time=time.time())
        t.record_request("m1", success=True, latency_ms=50)
        t.end_time = time.time()
        path = t.save(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "test_run_1"
        assert data["total_requests"] == 1


# ---------------------------------------------------------------------------
# C) Council score parsing edge cases
# ---------------------------------------------------------------------------

class TestScoreParsing:
    """Test _parse_batch_scores for various edge cases."""

    def _make_validator(self):
        """Create a minimal BSPValidator for testing parse methods."""
        from promptlab.utils.config import PromptLabConfig
        from promptlab.orchestrators.bsp_validator import BSPValidator
        cfg = PromptLabConfig()
        cfg.council.enabled = False  # skip council init
        with patch("promptlab.orchestrators.bsp_validator.load_bsp", return_value="test"):
            return BSPValidator(cfg)

    def test_perfect_format(self):
        v = self._make_validator()
        text = (
            "ROLE_ADHERENCE: 0.9\n"
            "RESPONSE_QUALITY: 0.85\n"
            "CONSISTENCY: 0.8\n"
            "APPROPRIATENESS: 0.75\n"
            "FINAL_SCORE: 0.82\n"
            "CONFIDENCE: high\n"
            "SUMMARY: Good\n"
            "RECOMMENDATIONS: none\n"
        )
        s = v._parse_batch_scores(text)
        assert s["final_score"] == 0.82
        assert s["role_adherence"] == 0.9
        assert s["parse_error"] is False

    def test_overall_only_backfills_dimensions(self):
        v = self._make_validator()
        text = "FINAL_SCORE: 0.7\nCONFIDENCE: medium\n"
        s = v._parse_batch_scores(text)
        assert s["final_score"] == 0.7
        assert s["role_adherence"] == 0.7  # backfilled
        assert s["response_quality"] == 0.7

    def test_malformed_flags_parse_error(self):
        v = self._make_validator()
        text = "This is just some random text with no scores."
        s = v._parse_batch_scores(text)
        assert s["parse_error"] is True
        assert s["final_score"] == 0.5  # fallback

    def test_scale_10_normalised(self):
        v = self._make_validator()
        text = "OVERALL_SCORE: 8.5\n"
        s = v._parse_batch_scores(text)
        assert s["final_score"] == 0.85

    def test_scale_100_normalised(self):
        v = self._make_validator()
        text = "FINAL_SCORE: 75\n"
        s = v._parse_batch_scores(text)
        assert s["final_score"] == 0.75


# ---------------------------------------------------------------------------
# C2) Council-level score parsing: markdown / format robustness
# ---------------------------------------------------------------------------

class TestCouncilScoreParsingMarkdown:
    """Test Council._parse_score_fields with markdown-wrapped responses.
    
    This covers the exact issue that step-3.5-flash:free (and other free models)
    trigger: they wrap output in markdown code fences, bold markers, etc.
    """

    def _make_council(self):
        """Create a minimal Council for testing parse methods."""
        from promptlab.llm_council.council.council import Council
        from promptlab.llm_council.llm_runner.runner import LLMRunner
        runner = LLMRunner({"default": "ollama/test", "providers": {}})
        return Council(
            {"members": [], "chairman": None, "mode": "fast", "required_judges": 2},
            runner,
        )

    def test_sanitize_strips_code_fences(self):
        c = self._make_council()
        raw = "```json\nOVERALL_SCORE: 0.8\nROLE_ADHERENCE: 0.7\n```"
        cleaned = c._sanitize_llm_response(raw)
        assert "```" not in cleaned
        assert "OVERALL_SCORE: 0.8" in cleaned

    def test_sanitize_strips_bold_markers(self):
        c = self._make_council()
        raw = "**OVERALL_SCORE:** 0.75\n**ROLE_ADHERENCE:** 0.8"
        cleaned = c._sanitize_llm_response(raw)
        assert "**" not in cleaned
        assert "OVERALL_SCORE:" in cleaned

    def test_sanitize_strips_list_prefixes(self):
        c = self._make_council()
        raw = "- OVERALL_SCORE: 0.6\n* ROLE_ADHERENCE: 0.7\n1. CONSISTENCY: 0.8"
        cleaned = c._sanitize_llm_response(raw)
        for line in cleaned.split("\n"):
            if line.strip():
                assert not line.strip().startswith("-")
                assert not line.strip().startswith("*")
                assert not line.strip().startswith("1.")

    def test_sanitize_strips_html_tags(self):
        c = self._make_council()
        raw = "<b>OVERALL_SCORE:</b> 0.9<br>ROLE_ADHERENCE: 0.85"
        cleaned = c._sanitize_llm_response(raw)
        assert "<b>" not in cleaned
        assert "<br>" not in cleaned

    def test_parse_markdown_code_fenced(self):
        """step-3.5-flash wraps in code fences — should parse correctly."""
        c = self._make_council()
        text = (
            "Here is my evaluation:\n"
            "```\n"
            "OVERALL_SCORE: 0.72\n"
            "ROLE_ADHERENCE: 0.8\n"
            "RESPONSE_QUALITY: 0.65\n"
            "CONSISTENCY: 0.75\n"
            "CONSTRAINT_COMPLIANCE: 0.68\n"
            "REASONING: Good overall adherence\n"
            "WEAK_AREAS: constraint handling\n"
            "```\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.72
        assert s["role_adherence"] == 0.8
        assert s["response_quality"] == 0.65
        assert s["consistency"] == 0.75
        assert s["constraint_compliance"] == 0.68

    def test_parse_bold_wrapped_fields(self):
        """Some models bold the field names — should still parse."""
        c = self._make_council()
        text = (
            "**OVERALL_SCORE:** 0.80\n"
            "**ROLE_ADHERENCE:** 0.85\n"
            "**RESPONSE_QUALITY:** 0.75\n"
            "**CONSISTENCY:** 0.80\n"
            "**CONSTRAINT_COMPLIANCE:** 0.78\n"
            "**REASONING:** Well structured responses\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.80
        assert s["role_adherence"] == 0.85

    def test_parse_list_prefixed_fields(self):
        """Models sometimes prefix with list markers."""
        c = self._make_council()
        text = (
            "- OVERALL_SCORE: 0.65\n"
            "- ROLE_ADHERENCE: 0.7\n"
            "- RESPONSE_QUALITY: 0.6\n"
            "- CONSISTENCY: 0.65\n"
            "- CONSTRAINT_COMPLIANCE: 0.6\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.65
        assert s["role_adherence"] == 0.7

    def test_parse_mixed_markdown(self):
        """Combination of code fence + bold + list prefix."""
        c = self._make_council()
        text = (
            "My evaluation:\n"
            "```\n"
            "- **OVERALL_SCORE:** 0.88\n"
            "- **ROLE_ADHERENCE:** 0.9\n"
            "```\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.88
        assert s["role_adherence"] == 0.9

    def test_parse_plain_text_still_works(self):
        """Regression: normal well-formatted output must still work."""
        c = self._make_council()
        text = (
            "OVERALL_SCORE: 0.59\n"
            "ROLE_ADHERENCE: 0.75\n"
            "RESPONSE_QUALITY: 0.55\n"
            "CONSISTENCY: 0.7\n"
            "CONSTRAINT_COMPLIANCE: 0.45\n"
            "REASONING: The LLM shows moderate adherence\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.59
        assert s["role_adherence"] == 0.75
        assert s["constraint_compliance"] == 0.45

    def test_parse_json_object_scores(self):
        """Many models return JSON instead of key-value lines."""
        c = self._make_council()
        text = (
            "{\n"
            '  "overall_score": 0.74,\n'
            '  "role_adherence": 0.8,\n'
            '  "response_quality": 0.7,\n'
            '  "consistency": 0.75,\n'
            '  "constraint_compliance": 0.68,\n'
            '  "reasoning": "Mostly compliant with minor issues",\n'
            '  "weak_areas": ["constraint handling"]\n'
            "}"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.74
        assert s["role_adherence"] == 0.8
        assert s["response_quality"] == 0.7
        assert s["consistency"] == 0.75
        assert s["constraint_compliance"] == 0.68
        assert "Mostly compliant" in s["reasoning"]
        assert s["weak_areas"] == ["constraint handling"]

    def test_parse_compact_shorthand_scores(self):
        """Compact score formats like R/Q/C/K/O should parse."""
        c = self._make_council()
        text = "R:0.8 Q:0.6 C:0.8 K:0.4 O:0.65"
        s = c._parse_score_fields(text)
        assert s["role_adherence"] == 0.8
        assert s["response_quality"] == 0.6
        assert s["consistency"] == 0.8
        assert s["constraint_compliance"] == 0.4
        assert s["overall_score"] == 0.65

    def test_parse_markdown_table_with_percentages(self):
        """Markdown-table score outputs with percentages should parse."""
        c = self._make_council()
        text = (
            "| field | score |\n"
            "|---|---|\n"
            "| Overall Score | 69% |\n"
            "| Role Adherence | 80% |\n"
            "| Response Quality | 60% |\n"
            "| Consistency | 80% |\n"
            "| Constraint Compliance | 40% |\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.69
        assert s["role_adherence"] == 0.8
        assert s["response_quality"] == 0.6
        assert s["consistency"] == 0.8
        assert s["constraint_compliance"] == 0.4

    def test_parse_arrow_separator(self):
        """Arrow separators (→, ->, =>) should be normalised and parsed."""
        c = self._make_council()
        text = (
            "OVERALL_SCORE → 0.77\n"
            "ROLE_ADHERENCE -> 0.85\n"
            "RESPONSE_QUALITY => 0.70\n"
            "CONSISTENCY: 0.80\n"
            "CONSTRAINT_COMPLIANCE: 0.65\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.77
        assert s["role_adherence"] == 0.85
        assert s["response_quality"] == 0.70

    def test_parse_natural_language_scores(self):
        """Scores embedded in natural language should be caught by regex fallback."""
        c = self._make_council()
        text = (
            "After careful evaluation, I would rate the overall score as 0.72.\n"
            "The role adherence is 0.80, response quality is 0.65, "
            "consistency is 0.75, and constraint compliance is 0.68.\n"
        )
        s = c._parse_score_fields(text)
        # The regex fallback should catch these
        assert s["overall_score"] is not None
        assert s["role_adherence"] is not None

    def test_dimension_regex_fallback(self):
        """Dimension-level regex fallback should find scores in prose."""
        c = self._make_council()
        text = (
            "The response shows good role adherence = 0.85 and "
            "decent consistency = 0.70. Overall score: 0.78"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.78
        assert s["role_adherence"] == 0.85
        assert s["consistency"] == 0.70

    def test_parse_scores_out_of_ten(self):
        """Scores given as X/10 should normalise to 0..1."""
        c = self._make_council()
        text = (
            "OVERALL_SCORE: 7.5/10\n"
            "ROLE_ADHERENCE: 8/10\n"
            "RESPONSE_QUALITY: 6/10\n"
            "CONSISTENCY: 7/10\n"
            "CONSTRAINT_COMPLIANCE: 5/10\n"
        )
        s = c._parse_score_fields(text)
        assert s["overall_score"] == 0.75
        assert s["role_adherence"] == 0.8
        assert s["constraint_compliance"] == 0.5


# ---------------------------------------------------------------------------
# D) Evaluation history write consistency
# ---------------------------------------------------------------------------

class TestHistoryConsistency:
    """Ensure evaluation_history never stores inconsistent tuples."""

    def test_zeroed_dimensions_are_backfilled(self, tmp_path: Path):
        from promptlab.utils.history import EvaluationHistory
        h = EvaluationHistory(project_root=tmp_path)
        h.record(overall_score=0.8, role_adherence=0.0, response_quality=0.0, consistency=0.0)
        entry = h._history[-1]
        # Should backfill dimensions from overall
        assert entry.role_adherence == 0.8
        assert entry.response_quality == 0.8
        assert entry.consistency == 0.8
        assert entry.parse_error is True

    def test_normal_record_no_parse_error(self, tmp_path: Path):
        from promptlab.utils.history import EvaluationHistory
        h = EvaluationHistory(project_root=tmp_path)
        h.record(overall_score=0.75, role_adherence=0.8, response_quality=0.7, consistency=0.75)
        entry = h._history[-1]
        assert entry.parse_error is False
        assert entry.role_adherence == 0.8

    def test_explicit_parse_error_flag(self, tmp_path: Path):
        from promptlab.utils.history import EvaluationHistory
        h = EvaluationHistory(project_root=tmp_path)
        h.record(overall_score=0.5, parse_error=True)
        entry = h._history[-1]
        assert entry.parse_error is True

    def test_history_persists_to_disk(self, tmp_path: Path):
        from promptlab.utils.history import EvaluationHistory
        h = EvaluationHistory(project_root=tmp_path)
        h.record(overall_score=0.9, bsp_version="1.0", model="test/model")
        
        data = json.loads(h.history_file.read_text())
        assert data["total_evaluations"] == 1
        assert data["evaluations"][0]["overall_score"] == 0.9
