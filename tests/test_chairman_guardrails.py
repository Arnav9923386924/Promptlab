"""Unit tests for chairman BSP guardrails.

These tests lock in behavior for the chairman update safety layer so
append-style BSP outputs are rejected before file writes, but CHANGES:
markers are cleaned automatically (not rejected).
"""

import json
import tempfile
from pathlib import Path

from promptlab.utils.chairman_guardrails import (
    is_append_only_update,
    strip_changes_block,
    strip_evaluation_context,
    validate_chairman_bsp_candidate,
)


class TestAppendDetection:
    """Deterministic checks for append-only BSP detection."""

    def test_detects_exact_prefix_append(self):
        current = "You are assistant.\nRule A\nRule B"
        improved = current + "\n\nAdditional notes at bottom"
        assert is_append_only_update(current, improved) is True

    def test_not_append_when_content_is_rewritten(self):
        current = "You are assistant.\nRule A\nRule B"
        improved = "You are assistant.\nRule A updated\nRule B\nRule C"
        assert is_append_only_update(current, improved) is False


class TestStripEvaluationContext:
    """Tests for the strip_evaluation_context() cleaning function."""

    def test_strips_evaluation_results(self):
        text = (
            "You are a helpful assistant.\n"
            "Rule 1: be concise.\n"
            "## EVALUATION RESULTS:\n"
            "- Overall Score: 0.75"
        )
        result = strip_evaluation_context(text)
        assert result == "You are a helpful assistant.\nRule 1: be concise."

    def test_strips_judge_feedback(self):
        text = (
            "You are a helpful assistant.\n"
            "## JUDGE FEEDBACK\n"
            "The model was not concise."
        )
        result = strip_evaluation_context(text)
        assert result == "You are a helpful assistant."

    def test_preserves_text_without_context(self):
        text = "You are a helpful assistant.\nRule 1: be concise."
        result = strip_evaluation_context(text)
        assert result == text


class TestStripChangesBlock:
    """Tests for the strip_changes_block() cleaning function."""

    def test_strips_leading_changes_block(self):
        text = (
            "CHANGES:\n"
            "- Added rule A\n"
            "- Improved section B\n"
            "\n"
            "You are a helpful assistant.\n"
            "Rule 1: be concise."
        )
        result = strip_changes_block(text)
        assert result == "You are a helpful assistant.\nRule 1: be concise."

    def test_strips_suggested_changes_block(self):
        text = (
            "SUGGESTED_CHANGES:\n"
            "1. Added rule A\n"
            "2. Improved section B\n"
            "\n"
            "You are a helpful assistant."
        )
        result = strip_changes_block(text)
        assert result == "You are a helpful assistant."

    def test_preserves_text_without_changes(self):
        text = "You are a helpful assistant.\nRule 1: be concise."
        result = strip_changes_block(text)
        assert result == text

    def test_handles_empty_string(self):
        assert strip_changes_block("") == ""

    def test_strips_numbered_items(self):
        text = (
            "CHANGES:\n"
            "1. First change\n"
            "2) Second change\n"
            "\n"
            "You are an AI."
        )
        result = strip_changes_block(text)
        assert result == "You are an AI."


class TestChairmanGuardrailValidation:
    """Guardrails-AI-backed checks for chairman BSP candidates."""

    def test_accepts_valid_replacement_candidate(self):
        current = (
            "You are LegalAI.\n"
            "Always explain legal concepts clearly.\n"
            "Include educational disclaimer."
        )
        candidate = (
            "You are LegalAI, a legal education assistant focused on contracts.\n"
            "Always explain legal principles in plain language with concise examples.\n"
            "Include a clear educational disclaimer and avoid legal representation."
        )

        outcome = validate_chairman_bsp_candidate(current, candidate)
        assert outcome.passed is True
        assert outcome.cleaned_bsp == candidate
        assert outcome.error is None

    def test_cleans_changes_block_from_candidate(self):
        """CHANGES: block should be stripped, not cause rejection."""
        current = (
            "You are LegalAI.\n"
            "Always explain legal concepts clearly.\n"
            "Include educational disclaimer."
        )
        candidate = (
            "CHANGES:\n"
            "- Added one more line\n"
            "\n"
            "You are LegalAI, an improved legal assistant.\n"
            "Always explain legal concepts clearly and concisely.\n"
            "Include educational disclaimer and safety notice."
        )

        outcome = validate_chairman_bsp_candidate(current, candidate)
        assert outcome.passed is True
        assert "CHANGES:" not in outcome.cleaned_bsp
        assert "You are LegalAI, an improved legal assistant." in outcome.cleaned_bsp

    def test_rejects_true_append_only_candidate(self):
        """True append-only (old BSP + extra content) should still be rejected."""
        current = (
            "You are LegalAI.\n"
            "Always explain legal concepts clearly.\n"
            "Include educational disclaimer."
        )
        # This is pure append: old content + new content
        candidate = (
            "You are LegalAI.\n"
            "Always explain legal concepts clearly.\n"
            "Include educational disclaimer.\n"
            "\n"
            "Additional safety rules:\n"
            "- Never provide specific legal advice"
        )

        outcome = validate_chairman_bsp_candidate(current, candidate)
        assert outcome.passed is False
        assert outcome.error is not None
        assert "append" in outcome.error.lower()

    def test_rejects_control_marker_leak(self):
        current = (
            "You are LegalAI.\n"
            "Always explain legal concepts clearly.\n"
            "Include educational disclaimer."
        )
        candidate = (
            "IMPROVED_BSP_START\n"
            "You are LegalAI and must respond safely.\n"
            "IMPROVED_BSP_END"
        )

        outcome = validate_chairman_bsp_candidate(current, candidate)
        assert outcome.passed is False
        assert outcome.error is not None
        assert "marker" in outcome.error.lower()


class TestLoadBSPFromDisk:
    """Tests that load_bsp() re-reads from disk when prompt_file is set."""

    def test_load_bsp_reads_from_file_not_cache(self):
        """load_bsp() should return file contents even if config.bsp.prompt is set."""
        from promptlab.utils.config import PromptLabConfig, BSPConfig, load_bsp

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            bsp_file = tmpdir / "bsp.txt"
            bsp_file.write_text("Version 1 BSP", encoding="utf-8")

            config = PromptLabConfig(
                bsp=BSPConfig(
                    prompt="stale cached content",
                    prompt_file="bsp.txt",
                )
            )

            result = load_bsp(config, tmpdir)
            assert result == "Version 1 BSP"

            # Update file on disk
            bsp_file.write_text("Version 2 BSP", encoding="utf-8")
            result = load_bsp(config, tmpdir)
            assert result == "Version 2 BSP"

    def test_load_bsp_falls_back_to_inline(self):
        """When prompt_file doesn't exist, fall back to inline prompt."""
        from promptlab.utils.config import PromptLabConfig, BSPConfig, load_bsp

        config = PromptLabConfig(
            bsp=BSPConfig(
                prompt="inline BSP content",
                prompt_file=None,
            )
        )

        result = load_bsp(config)
        assert result == "inline BSP content"
