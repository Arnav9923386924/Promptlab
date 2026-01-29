"""Assertion engine for validating LLM responses."""

import re
import json
from typing import Optional

from promptlab.core.models import Assertion, AssertionType, AssertionResult


def run_assertion(assertion: Assertion, response: str) -> AssertionResult:
    """Run a single assertion against a response.
    
    Args:
        assertion: The assertion to run
        response: The LLM response to validate
        
    Returns:
        AssertionResult with pass/fail status and details
    """
    try:
        if assertion.type == AssertionType.CONTAINS:
            return _check_contains(assertion, response)
        elif assertion.type == AssertionType.CONTAINS_NUMBER:
            return _check_contains_number(assertion, response)
        elif assertion.type == AssertionType.EXACT_MATCH:
            return _check_exact_match(assertion, response)
        elif assertion.type == AssertionType.REGEX:
            return _check_regex(assertion, response)
        elif assertion.type == AssertionType.ONE_OF:
            return _check_one_of(assertion, response)
        elif assertion.type == AssertionType.MAX_LENGTH:
            return _check_max_length(assertion, response)
        elif assertion.type == AssertionType.MIN_LENGTH:
            return _check_min_length(assertion, response)
        elif assertion.type == AssertionType.JSON_VALID:
            return _check_json_valid(assertion, response)
        elif assertion.type == AssertionType.COUNCIL_JUDGE:
            # Council judge is handled separately by the council module
            return AssertionResult(
                type=assertion.type,
                passed=True,
                message="Council evaluation pending",
            )
        else:
            return AssertionResult(
                type=assertion.type,
                passed=False,
                message=f"Unknown assertion type: {assertion.type}",
            )
    except Exception as e:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=f"Assertion error: {str(e)}",
        )


def _check_contains(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response contains the expected value."""
    if assertion.value is None:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message="No value specified for contains assertion",
        )
    
    check_response = response if assertion.case_sensitive else response.lower()
    check_value = assertion.value if assertion.case_sensitive else assertion.value.lower()
    
    passed = check_value in check_response
    
    return AssertionResult(
        type=assertion.type,
        passed=passed,
        expected=assertion.value,
        actual=response[:200] + "..." if len(response) > 200 else response,
        message=None if passed else f"Response does not contain '{assertion.value}'",
    )


def _normalize_number(s: str) -> str:
    """Normalize a number string by removing formatting characters.
    
    Handles: commas (70,000 -> 70000), currency symbols ($70000 -> 70000),
    trailing zeros after decimal (.00 -> ""), etc.
    """
    # Remove common currency symbols and whitespace
    normalized = re.sub(r'[$€£¥₹\s]', '', s)
    # Remove thousand separators (commas)
    normalized = re.sub(r',', '', normalized)
    # Remove trailing .00 or .0
    normalized = re.sub(r'\.0+$', '', normalized)
    return normalized


def _extract_numbers(text: str) -> list[str]:
    """Extract all numbers from text, normalizing each one.
    
    Matches integers and decimals, including those with commas and currency symbols.
    """
    # Match numbers that may have currency symbols, commas, and decimals
    # This regex captures numbers like: 70000, 70,000, $70,000, 70000.00, $70,000.00
    pattern = r'[$€£¥₹]?[\d,]+\.?\d*'
    matches = re.findall(pattern, text)
    # Normalize each match
    return [_normalize_number(m) for m in matches if _normalize_number(m)]


def _check_contains_number(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response contains the expected number, with format normalization.
    
    This handles cases where the LLM might format numbers differently:
    - 70000 vs 70,000
    - 70000 vs $70,000
    - 45 vs 45.0
    """
    if assertion.value is None:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message="No value specified for contains_number assertion",
        )
    
    expected_normalized = _normalize_number(assertion.value)
    response_numbers = _extract_numbers(response)
    
    passed = expected_normalized in response_numbers
    
    return AssertionResult(
        type=assertion.type,
        passed=passed,
        expected=assertion.value,
        actual=f"Found numbers: {response_numbers[:10]}" if response_numbers else "No numbers found",
        message=None if passed else f"Response does not contain number '{assertion.value}' (normalized: {expected_normalized})",
    )


def _check_exact_match(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response exactly matches the expected value."""
    if assertion.value is None:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message="No value specified for exact_match assertion",
        )
    
    check_response = response.strip() if assertion.case_sensitive else response.strip().lower()
    check_value = assertion.value.strip() if assertion.case_sensitive else assertion.value.strip().lower()
    
    passed = check_response == check_value
    
    return AssertionResult(
        type=assertion.type,
        passed=passed,
        expected=assertion.value,
        actual=response,
        message=None if passed else "Response does not match exactly",
    )


def _check_regex(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response matches the regex pattern."""
    if assertion.pattern is None:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message="No pattern specified for regex assertion",
        )
    
    flags = 0 if assertion.case_sensitive else re.IGNORECASE
    
    try:
        match = re.search(assertion.pattern, response, flags)
        passed = match is not None
    except re.error as e:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=f"Invalid regex pattern: {e}",
        )
    
    return AssertionResult(
        type=assertion.type,
        passed=passed,
        expected=assertion.pattern,
        actual=response[:200] + "..." if len(response) > 200 else response,
        message=None if passed else f"Response does not match pattern '{assertion.pattern}'",
    )


def _check_one_of(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response contains one of the expected values."""
    if not assertion.values:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message="No values specified for one_of assertion",
        )
    
    check_response = response if assertion.case_sensitive else response.lower()
    
    for value in assertion.values:
        check_value = value if assertion.case_sensitive else value.lower()
        if check_value in check_response:
            return AssertionResult(
                type=assertion.type,
                passed=True,
                expected=f"one of: {assertion.values}",
                actual=response[:200] + "..." if len(response) > 200 else response,
            )
    
    return AssertionResult(
        type=assertion.type,
        passed=False,
        expected=f"one of: {assertion.values}",
        actual=response[:200] + "..." if len(response) > 200 else response,
        message=f"Response does not contain any of: {assertion.values}",
    )


def _check_max_length(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response is within max length."""
    max_chars = assertion.max_chars
    if max_chars is None:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message="No chars specified for max_length assertion",
        )
    
    actual_length = len(response)
    passed = actual_length <= max_chars
    
    return AssertionResult(
        type=assertion.type,
        passed=passed,
        expected=f"<= {max_chars} chars",
        actual=f"{actual_length} chars",
        message=None if passed else f"Response too long: {actual_length} > {max_chars}",
    )


def _check_min_length(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response meets minimum length."""
    min_chars = assertion.max_chars  # Reusing field
    if min_chars is None:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message="No chars specified for min_length assertion",
        )
    
    actual_length = len(response)
    passed = actual_length >= min_chars
    
    return AssertionResult(
        type=assertion.type,
        passed=passed,
        expected=f">= {min_chars} chars",
        actual=f"{actual_length} chars",
        message=None if passed else f"Response too short: {actual_length} < {min_chars}",
    )


def _check_json_valid(assertion: Assertion, response: str) -> AssertionResult:
    """Check if response is valid JSON."""
    try:
        json.loads(response)
        return AssertionResult(
            type=assertion.type,
            passed=True,
        )
    except json.JSONDecodeError as e:
        return AssertionResult(
            type=assertion.type,
            passed=False,
            message=f"Invalid JSON: {e}",
        )


def run_all_assertions(assertions: list[Assertion], response: str) -> list[AssertionResult]:
    """Run all assertions against a response.
    
    Args:
        assertions: List of assertions to run
        response: The LLM response to validate
        
    Returns:
        List of AssertionResults
    """
    return [run_assertion(assertion, response) for assertion in assertions]
