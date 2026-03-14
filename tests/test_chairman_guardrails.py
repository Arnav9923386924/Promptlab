"""Unit tests for chairman BSP guardrails.

These tests lock in behavior for the chairman update safety layer so
append-style BSP outputs are rejected before file writes.
"""

from promptlab.utils.chairman_guardrails import (
    is_append_only_update,
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

    def test_rejects_append_style_candidate(self):
        current = (
            "You are LegalAI.\n"
            "Always explain legal concepts clearly.\n"
            "Include educational disclaimer."
        )
        candidate = current + "\n\nCHANGES:\n- Added one more line"

        outcome = validate_chairman_bsp_candidate(current, candidate)
        assert outcome.passed is False
        assert outcome.error is not None
        assert "append" in outcome.error.lower() or "marker" in outcome.error.lower()

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
