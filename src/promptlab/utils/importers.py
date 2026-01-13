"""CSV and data import utilities for test case generation."""

import csv
from pathlib import Path
from typing import Optional
import yaml


def import_from_csv(
    csv_path: Path,
    output_path: Path,
    input_column: str = "input",
    expected_column: Optional[str] = "expected",
    stage_column: Optional[str] = "stage",
    tags_column: Optional[str] = "tags",
) -> int:
    """Import test cases from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        output_path: Path to output YAML file
        input_column: Column name for test input/prompt
        expected_column: Column name for expected output
        stage_column: Column name for pipeline stage
        tags_column: Column name for tags (comma-separated)
        
    Returns:
        Number of test cases imported
    """
    cases = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            case = {
                "id": f"imported-{i+1}",
                "prompt": row.get(input_column, ""),
            }
            
            # Add expected if present
            if expected_column and expected_column in row:
                expected = row[expected_column]
                if expected:
                    case["assertions"] = [
                        {"type": "contains", "value": expected}
                    ]
            
            # Add tags if present
            if tags_column and tags_column in row:
                tags = row[tags_column]
                if tags:
                    case["tags"] = [t.strip() for t in tags.split(",")]
            
            cases.append(case)
    
    # Determine stage from first row or column
    stage = None
    if stage_column:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_row = next(reader, {})
            stage = first_row.get(stage_column)
    
    # Build YAML structure
    test_suite = {
        "metadata": {
            "name": f"Imported from {csv_path.name}",
        },
        "defaults": {
            "temperature": 0,
        },
        "cases": cases,
    }
    
    if stage:
        test_suite["stage"] = stage
    
    # Write YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(test_suite, f, default_flow_style=False, allow_unicode=True)
    
    return len(cases)


def import_from_jsonl(
    jsonl_path: Path,
    output_path: Path,
    prompt_key: str = "prompt",
    response_key: str = "response",
) -> int:
    """Import test cases from a JSONL log file.
    
    Args:
        jsonl_path: Path to JSONL file
        output_path: Path to output YAML file
        prompt_key: Key for prompt in JSON objects
        response_key: Key for response in JSON objects
        
    Returns:
        Number of test cases imported
    """
    import json
    
    cases = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                prompt = obj.get(prompt_key, "")
                response = obj.get(response_key, "")
                
                if not prompt:
                    continue
                
                case = {
                    "id": f"captured-{i+1}",
                    "prompt": prompt,
                }
                
                # Use captured response as expected (golden test)
                if response:
                    case["expected"] = response
                    case["assertions"] = [
                        {"type": "semantic_similarity", "expected": response, "min_score": 0.8}
                    ]
                
                cases.append(case)
                
            except json.JSONDecodeError:
                continue
    
    # Build YAML structure
    test_suite = {
        "metadata": {
            "name": f"Captured from {jsonl_path.name}",
        },
        "defaults": {
            "temperature": 0,
        },
        "cases": cases,
    }
    
    # Write YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(test_suite, f, default_flow_style=False, allow_unicode=True)
    
    return len(cases)
