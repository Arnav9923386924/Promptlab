"""Guardrails for chairman-generated BSP updates.

These checks ensure the chairman returns a clean replacement BSP,
not an append-only addendum of the previous BSP.

The CHANGES: marker is treated as a *cleanable artefact* (stripped
automatically), NOT a hard failure.  Only structural markers like
IMPROVED_BSP_START / IMPROVED_BSP_END are treated as hard failures,
since their presence means the extraction step itself failed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_GUARDRAILS_AVAILABLE = False
_GUARDRAILS_IMPORT_ERROR: Optional[str] = None

try:
    from guardrails import Guard
    from guardrails.classes.validation.validation_result import FailResult, PassResult
    from guardrails.validator_base import Validator, register_validator
    _GUARDRAILS_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - environment-dependent
    Guard = None  # type: ignore[assignment]
    FailResult = None  # type: ignore[assignment]
    PassResult = None  # type: ignore[assignment]
    Validator = object  # type: ignore[assignment]

    def register_validator(*args, **kwargs):  # type: ignore[override]
        def _decorator(cls):
            return cls

        return _decorator

    _GUARDRAILS_IMPORT_ERROR = str(_exc)


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


# ---------------------------------------------------------------------------
# CHANGES: block stripping
# ---------------------------------------------------------------------------

def strip_changes_block(text: str) -> str:
    """Remove a leading CHANGES: block (and its bullet items) from BSP text.

    Models frequently leak the CHANGES: list into the BSP body.  This
    function strips it so the remaining text is a clean BSP.

    Handles patterns like:
        CHANGES:
        - change 1
        - change 2

        [actual BSP starts here]

    Also handles inline ``CHANGES: ...`` on a single line.
    """
    if not text:
        return text

    lines = text.splitlines()
    cleaned_lines: list[str] = []
    in_changes_block = False
    changes_block_ended = False

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if not changes_block_ended:
            # Detect start of a CHANGES: block
            if upper.startswith("CHANGES:") or upper.startswith("SUGGESTED_CHANGES:"):
                in_changes_block = True
                # If there's BSP content on the same line after "CHANGES: ...",
                # that's just the header — skip it.
                continue

            if in_changes_block:
                # Inside a changes block: skip bullet items and numbered items
                if stripped.startswith("- ") or stripped.startswith("* ") or re.match(r"^\d+[\.\)]\s", stripped):
                    continue
                # Empty line might separate changes from BSP
                if not stripped:
                    continue
                # Non-bullet, non-empty line → changes block is over
                in_changes_block = False
                changes_block_ended = True

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result if result else text.strip()


def strip_evaluation_context(text: str) -> str:
    """Remove leaked evaluation context from the end of a BSP candidate.

    Models occasionally regurgitate the evaluation context (results, feedback,
    sample outputs) provided in the prompt inside the improved BSP block.
    This function truncates the text at the first sign of these sections.
    """
    if not text:
        return text

    # Common headers that indicate the start of leaked context
    context_headers = [
        "## EVALUATION RESULTS",
        "EVALUATION RESULTS:",
        "## JUDGE FEEDBACK",
        "JUDGE FEEDBACK:",
        "## WEAK AREAS IDENTIFIED",
        "WEAK AREAS IDENTIFIED:",
        "## SAMPLE OUTPUTS",
        "SAMPLE OUTPUTS:"
    ]

    lines = text.splitlines()
    for i, line in enumerate(lines):
        upper_line = line.strip().upper()
        for header in context_headers:
            if upper_line.startswith(header):
                # Truncate at this line
                return "\n".join(lines[:i]).strip()

    return text.strip()


def is_append_only_update(current_bsp: str, improved_bsp: str) -> bool:
    """Detect when improved BSP is mostly old BSP with extra trailing content."""
    current = (current_bsp or "").strip()
    improved = (improved_bsp or "").strip()

    if not current or not improved:
        return False
    if len(improved) <= len(current):
        return False

    # Exact-prefix append is the most common failure mode.
    if improved.startswith(current):
        return True

    # Near-prefix append with minor formatting differences.
    current_norm = _normalize_text(current)
    improved_norm = _normalize_text(improved)
    if not current_norm or not improved_norm:
        return False

    idx = improved_norm.find(current_norm)
    if idx != -1 and idx <= 30 and len(improved_norm) > int(len(current_norm) * 1.05):
        return True

    return False


if _GUARDRAILS_AVAILABLE:

    @register_validator(name="chairman_non_empty_bsp", data_type="string")
    class ChairmanNonEmptyBSPValidator(Validator):
        """Ensure the proposed BSP is non-empty and reasonably sized."""

        def _validate(self, value, metadata):
            text = (value or "").strip()
            if len(text) < 20:
                return FailResult(outcome="fail", error_message="Improved BSP is too short to be a complete replacement.")
            return PassResult()


    @register_validator(name="chairman_no_marker_leak", data_type="string")
    class ChairmanNoMarkerLeakValidator(Validator):
        """Ensure output doesn't keep internal prompt markers.

        Only IMPROVED_BSP_START / IMPROVED_BSP_END are hard failures.
        CHANGES: is handled by the cleaning step, not by rejection.
        """

        def _validate(self, value, metadata):
            text_upper = (value or "").upper()
            # Only structural extraction markers are hard failures.
            leaked_tokens = [
                "IMPROVED_BSP_START",
                "IMPROVED_BSP_END",
            ]
            for token in leaked_tokens:
                if token in text_upper:
                    return FailResult(
                        outcome="fail",
                        error_message=f"Improved BSP leaked control marker: {token}",
                    )
            return PassResult()


    @register_validator(name="chairman_no_append_only", data_type="string")
    class ChairmanNoAppendOnlyValidator(Validator):
        """Block append-only outputs based on old BSP context."""

        def _validate(self, value, metadata):
            current_bsp = (metadata or {}).get("current_bsp", "")
            improved_bsp = (value or "")

            if is_append_only_update(current_bsp, improved_bsp):
                return FailResult(
                    outcome="fail",
                    error_message="Improved BSP appears to append to the old BSP instead of replacing it.",
                )
            return PassResult()


@dataclass
class ChairmanGuardrailOutcome:
    """Result of chairman BSP validation."""

    passed: bool
    cleaned_bsp: str
    error: Optional[str] = None


def _deterministic_checks(current_bsp: str, candidate: str) -> Optional[str]:
    """Return first deterministic failure reason, or None when valid.

    Note: CHANGES: is NOT a hard failure — it's cleaned automatically
    before this check runs.  Only IMPROVED_BSP_START / IMPROVED_BSP_END
    are structural marker leaks that indicate extraction failure.
    """
    if len(candidate) < 20:
        return "Improved BSP is too short to be a complete replacement."

    # Only structural markers are hard failures
    hard_fail_tokens = ["IMPROVED_BSP_START", "IMPROVED_BSP_END"]
    text_upper = candidate.upper()
    for token in hard_fail_tokens:
        if token in text_upper:
            return f"Improved BSP leaked control marker: {token}"

    if is_append_only_update(current_bsp, candidate):
        return "Improved BSP appears to append to the old BSP instead of replacing it."

    return None


if _GUARDRAILS_AVAILABLE:
    _CHAIRMAN_GUARD = Guard.for_string(
        validators=[
            ChairmanNonEmptyBSPValidator(on_fail="noop"),
            ChairmanNoMarkerLeakValidator(on_fail="noop"),
            ChairmanNoAppendOnlyValidator(on_fail="noop"),
        ],
        string_description="Chairman-generated replacement BSP",
    )
else:
    _CHAIRMAN_GUARD = None


def validate_chairman_bsp_candidate(current_bsp: str, candidate_bsp: str) -> ChairmanGuardrailOutcome:
    """Validate chairman BSP candidate with Guardrails AI + deterministic checks.

    Performs automatic cleaning (strip CHANGES: blocks) before validation,
    so that common model formatting artefacts don't cause hard failures.
    """
    candidate = (candidate_bsp or "").strip()
    if not candidate:
        return ChairmanGuardrailOutcome(passed=False, cleaned_bsp="", error="Improved BSP candidate is empty.")

    # ---- Step 1: Clean known artefacts ----
    candidate = strip_evaluation_context(candidate)
    candidate = strip_changes_block(candidate)

    # ---- Step 2: Deterministic checks on cleaned candidate ----
    deterministic_error = _deterministic_checks(current_bsp or "", candidate)
    if deterministic_error:
        return ChairmanGuardrailOutcome(passed=False, cleaned_bsp=candidate, error=deterministic_error)

    if _CHAIRMAN_GUARD is None:
        # Guardrails optional: deterministic checks are still enforced.
        return ChairmanGuardrailOutcome(passed=True, cleaned_bsp=candidate, error=None)

    try:
        outcome = _CHAIRMAN_GUARD.validate(
            llm_output=candidate,
            metadata={"current_bsp": current_bsp or ""},
        )
        passed = bool(getattr(outcome, "validation_passed", False))
        validated_output = getattr(outcome, "validated_output", None)
        cleaned = (validated_output if isinstance(validated_output, str) else candidate).strip()
        error = None
        if not passed:
            summaries = getattr(outcome, "validation_summaries", None) or []
            messages = []
            for summary in summaries:
                reason = getattr(summary, "failure_reason", None)
                if reason:
                    messages.append(str(reason))
            if messages:
                error = "; ".join(messages)
            else:
                base_error = getattr(outcome, "error", None)
                error = str(base_error) if base_error else "Chairman BSP failed guardrails validation."
        return ChairmanGuardrailOutcome(passed=passed, cleaned_bsp=cleaned, error=error)
    except Exception as exc:
        # Fail closed for chairman update safety.
        return ChairmanGuardrailOutcome(
            passed=False,
            cleaned_bsp=candidate,
            error=f"Guardrails validation error: {exc}; import_error={_GUARDRAILS_IMPORT_ERROR}",
        )
