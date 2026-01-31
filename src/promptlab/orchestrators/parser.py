"""YAML test file parser."""

from pathlib import Path
from typing import Union
import yaml

from promptlab.orchestrators.models import TestSuite, TestCase, Assertion, TestSuiteDefaults, TestSuiteMetadata


def parse_test_file(path: Union[str, Path]) -> TestSuite:
    """Parse a YAML test file into a TestSuite object.
    
    Args:
        path: Path to the YAML test file
        
    Returns:
        TestSuite object with all test cases
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If the test structure is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not data:
        raise ValueError(f"Empty test file: {path}")
    
    # Parse metadata
    metadata = None
    if "metadata" in data:
        metadata = TestSuiteMetadata(**data["metadata"])
    
    # Parse defaults
    defaults = None
    if "defaults" in data:
        defaults = TestSuiteDefaults(**data["defaults"])
    
    # Parse cases
    if "cases" not in data:
        raise ValueError(f"No test cases found in: {path}")
    
    cases = []
    for case_data in data["cases"]:
        # Parse assertions
        assertions = []
        for assertion_data in case_data.get("assertions", []):
            assertions.append(Assertion(**assertion_data))
        
        case_data["assertions"] = assertions
        cases.append(TestCase(**case_data))
    
    return TestSuite(
        metadata=metadata,
        defaults=defaults,
        stage=data.get("stage"),
        cases=cases,
    )


def discover_test_files(directory: Union[str, Path], pattern: str = "*.yaml") -> list[Path]:
    """Discover all test files in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for test files
        
    Returns:
        List of paths to test files
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    # Find all matching files recursively
    files = list(directory.rglob(pattern))
    
    # Also check for .yml extension
    if pattern == "*.yaml":
        files.extend(directory.rglob("*.yml"))
    
    return sorted(files)
