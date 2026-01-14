"""Dynamic test generator - Uses LLM to generate role-specific test cases.

This module provides functionality to dynamically generate test YAML files
based on a given role/persona using an LLM. It supports multiple LLM providers
and includes fallback generation when JSON parsing fails.
"""

import json
import re
import yaml
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()

# Prompt template for the generator LLM
GENERATOR_PROMPT = '''You are a test case generator for LLM evaluation.

Given this role/persona: "{role}"

Analyze what capabilities this LLM role should have, then generate test categories and test cases.

You MUST respond with valid JSON in this exact format:
{{
  "categories": [
    {{
      "name": "category_name",
      "description": "What this category tests",
      "cases": [
        {{
          "id": "unique-test-id",
          "prompt": "The user prompt to test",
          "expected_pattern": "regex pattern that response should match",
          "description": "What this test checks"
        }}
      ]
    }}
  ]
}}

Generate 3-5 relevant test categories with 3-5 test cases each.
Make tests specific to the role. The expected_pattern should be a regex that matches correct responses.

Examples of categories based on roles:
- "Python expert" → code_generation, debugging, best_practices, explanations
- "Customer support" → empathy, problem_solving, escalation, product_knowledge
- "Math tutor" → arithmetic, algebra, explanations, step_by_step
- "General assistant" → reasoning, knowledge, helpfulness, clarity

RESPOND ONLY WITH THE JSON, NO OTHER TEXT.'''


async def generate_tests_for_role(
    role: str,
    llm_runner,
    generator_model: str,
    output_dir: Path,
) -> list[Path]:
    """Generate test YAML files for a given role.
    
    Args:
        role: The role/persona description
        llm_runner: LLMRunner instance
        generator_model: Model to use for generation
        output_dir: Directory to save YAML files
        
    Returns:
        List of paths to generated YAML files
    """
    console.print(f"[cyan]Generating tests for role: {role}[/cyan]")
    console.print(f"[dim]Using generator: {generator_model}[/dim]")
    
    # Generate test cases using the generator LLM
    prompt = GENERATOR_PROMPT.format(role=role)
    
    try:
        result = await llm_runner.complete(
            prompt=prompt,
            model=generator_model,
            temperature=0.3,  # Lower temperature for more consistent JSON
            max_tokens=4000,
        )
        
        response_text = result.text.strip()
        
        # Try multiple methods to extract JSON
        # Method 1: Find JSON block in markdown
        if "```json" in response_text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if match:
                response_text = match.group(1)
        elif "```" in response_text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', response_text)
            if match:
                response_text = match.group(1)
        
        # Method 2: Find JSON object directly
        if not response_text.strip().startswith('{'):
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                response_text = match.group(0)
        
        # Parse JSON
        data = json.loads(response_text)
        
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse generator response as JSON: {e}[/red]")
        console.print(f"[dim]Response was: {result.text[:500]}...[/dim]")
        
        # Fallback: Create simple tests if parsing fails
        console.print("[yellow]Using fallback test generation...[/yellow]")
        data = {
            "categories": [
                {
                    "name": "basic_capability",
                    "description": f"Basic tests for: {role}",
                    "cases": [
                        {
                            "id": "basic-greeting",
                            "prompt": "Hello, please introduce yourself briefly.",
                            "expected_pattern": "(?i).+",
                        },
                        {
                            "id": "basic-task",
                            "prompt": f"As a {role}, what is one thing you can help me with?",
                            "expected_pattern": "(?i).+",
                        },
                    ]
                }
            ]
        }
    except Exception as e:
        console.print(f"[red]Error generating tests: {e}[/red]")
        return []
    
    # Create YAML files for each category
    generated_files = []
    output_dir.mkdir(exist_ok=True)
    
    for category in data.get("categories", []):
        category_name = category.get("name", "unknown").lower().replace(" ", "_")
        description = category.get("description", "")
        cases = category.get("cases", [])
        
        if not cases:
            continue
        
        # Build YAML structure
        yaml_data = {
            "metadata": {
                "name": f"{category_name.replace('_', ' ').title()} Tests",
                "description": description,
                "generated_for_role": role,
            },
            "defaults": {
                "temperature": 0,
            },
            "cases": []
        }
        
        for case in cases:
            yaml_case = {
                "id": case.get("id", f"{category_name}-{len(yaml_data['cases'])+1}"),
                "prompt": case.get("prompt", ""),
                "assertions": [
                    {
                        "type": "regex",
                        "pattern": case.get("expected_pattern", "(?i).+"),
                    }
                ]
            }
            if case.get("description"):
                yaml_case["tags"] = [case.get("description")]
            
            yaml_data["cases"].append(yaml_case)
        
        # Write YAML file
        file_path = output_dir / f"generated_{category_name}.yaml"
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
        console.print(f"[green]✓[/green] Generated {file_path.name} ({len(cases)} tests)")
        generated_files.append(file_path)
    
    return generated_files


def get_generated_test_files(tests_dir: Path) -> list[Path]:
    """Get all generated test files (files starting with 'generated_')."""
    if not tests_dir.exists():
        return []
    return list(tests_dir.glob("generated_*.yaml"))


def clean_generated_tests(tests_dir: Path) -> int:
    """Remove all generated test files."""
    files = get_generated_test_files(tests_dir)
    for f in files:
        f.unlink()
    return len(files)
