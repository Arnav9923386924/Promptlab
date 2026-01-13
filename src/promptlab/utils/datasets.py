"""HuggingFace dataset importers for standard LLM benchmarks."""

import yaml
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress

console = Console()


def import_mmlu(output_dir: Path, max_samples: int = 50, subject: Optional[str] = None) -> int:
    """Import MMLU (Massive Multitask Language Understanding) dataset.
    
    Tests general knowledge across 57 subjects.
    
    Args:
        output_dir: Directory to save YAML files
        max_samples: Max samples per subject
        subject: Specific subject to import (or None for all)
        
    Returns:
        Number of test cases imported
    """
    from datasets import load_dataset
    
    console.print("[cyan]Downloading MMLU dataset...[/cyan]")
    
    # MMLU has many subjects - pick a few important ones
    subjects = [subject] if subject else [
        "abstract_algebra",
        "computer_security",
        "machine_learning",
        "high_school_mathematics",
        "logical_fallacies",
    ]
    
    total_cases = 0
    
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
        except Exception as e:
            console.print(f"[yellow]Skipping {subj}: {e}[/yellow]")
            continue
        
        cases = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            
            # Format: question + choices
            choices = item["choices"]
            question = item["question"]
            answer_idx = item["answer"]  # 0, 1, 2, or 3
            correct_answer = choices[answer_idx]
            
            prompt = f"""Answer this multiple choice question. Reply with ONLY the letter (A, B, C, or D).

Question: {question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}"""

            answer_letter = ["A", "B", "C", "D"][answer_idx]
            
            cases.append({
                "id": f"mmlu-{subj}-{i+1}",
                "prompt": prompt,
                "assertions": [
                    {"type": "regex", "pattern": f"(?i){answer_letter}"}
                ]
            })
        
        if cases:
            suite = {
                "metadata": {"name": f"MMLU - {subj.replace('_', ' ').title()}"},
                "defaults": {"temperature": 0},
                "cases": cases
            }
            
            output_path = output_dir / f"mmlu_{subj}.yaml"
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(suite, f, default_flow_style=False, allow_unicode=True)
            
            console.print(f"[green]✓[/green] {output_path.name}: {len(cases)} tests")
            total_cases += len(cases)
    
    return total_cases


def import_truthfulqa(output_dir: Path, max_samples: int = 50) -> int:
    """Import TruthfulQA dataset.
    
    Tests if model generates truthful (non-hallucinated) answers.
    
    Args:
        output_dir: Directory to save YAML file
        max_samples: Max samples to import
        
    Returns:
        Number of test cases imported
    """
    from datasets import load_dataset
    
    console.print("[cyan]Downloading TruthfulQA dataset...[/cyan]")
    
    ds = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
    
    cases = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        
        question = item["question"]
        best_answer = item["best_answer"]
        incorrect_answers = item["incorrect_answers"]
        
        # Create assertions based on correct/incorrect answers
        cases.append({
            "id": f"truthful-{i+1}",
            "prompt": f"Answer this question truthfully and concisely:\n\n{question}",
            "assertions": [
                # Should contain elements of correct answer
                {"type": "regex", "pattern": f"(?i)({'|'.join(best_answer.split()[:3])})"},
            ],
            "tags": ["truthfulness", "hallucination"]
        })
    
    suite = {
        "metadata": {"name": "TruthfulQA - Hallucination Detection"},
        "defaults": {"temperature": 0},
        "cases": cases
    }
    
    output_path = output_dir / "truthfulqa.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(suite, f, default_flow_style=False, allow_unicode=True)
    
    console.print(f"[green]✓[/green] {output_path.name}: {len(cases)} tests")
    return len(cases)


def import_gsm8k(output_dir: Path, max_samples: int = 50) -> int:
    """Import GSM8K dataset.
    
    Tests grade-school math reasoning (word problems).
    
    Args:
        output_dir: Directory to save YAML file
        max_samples: Max samples to import
        
    Returns:
        Number of test cases imported
    """
    from datasets import load_dataset
    
    console.print("[cyan]Downloading GSM8K dataset...[/cyan]")
    
    ds = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    
    cases = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        
        question = item["question"]
        answer = item["answer"]
        
        # Extract just the final number from answer
        # GSM8K format: "explanation #### final_answer"
        if "####" in answer:
            final_answer = answer.split("####")[-1].strip()
        else:
            final_answer = answer.strip()
        
        cases.append({
            "id": f"gsm8k-{i+1}",
            "prompt": f"Solve this math problem. Give the final numerical answer on its own line at the end.\n\n{question}",
            "assertions": [
                {"type": "contains", "value": final_answer}
            ],
            "tags": ["math", "reasoning"]
        })
    
    suite = {
        "metadata": {"name": "GSM8K - Math Reasoning"},
        "defaults": {"temperature": 0},
        "cases": cases
    }
    
    output_path = output_dir / "gsm8k.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(suite, f, default_flow_style=False, allow_unicode=True)
    
    console.print(f"[green]✓[/green] {output_path.name}: {len(cases)} tests")
    return len(cases)


def import_hellaswag(output_dir: Path, max_samples: int = 50) -> int:
    """Import HellaSwag dataset.
    
    Tests commonsense reasoning - completing sentences.
    
    Args:
        output_dir: Directory to save YAML file
        max_samples: Max samples to import
        
    Returns:
        Number of test cases imported
    """
    from datasets import load_dataset
    
    console.print("[cyan]Downloading HellaSwag dataset...[/cyan]")
    
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    
    cases = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        
        context = item["ctx"]
        endings = item["endings"]
        correct_idx = int(item["label"])
        
        prompt = f"""Complete this sentence. Choose A, B, C, or D. Reply with ONLY the letter.

Context: {context}

A) {endings[0]}
B) {endings[1]}
C) {endings[2]}
D) {endings[3]}"""

        answer_letter = ["A", "B", "C", "D"][correct_idx]
        
        cases.append({
            "id": f"hellaswag-{i+1}",
            "prompt": prompt,
            "assertions": [
                {"type": "regex", "pattern": f"(?i){answer_letter}"}
            ],
            "tags": ["commonsense", "reasoning"]
        })
    
    suite = {
        "metadata": {"name": "HellaSwag - Commonsense Reasoning"},
        "defaults": {"temperature": 0},
        "cases": cases
    }
    
    output_path = output_dir / "hellaswag.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(suite, f, default_flow_style=False, allow_unicode=True)
    
    console.print(f"[green]✓[/green] {output_path.name}: {len(cases)} tests")
    return len(cases)


# Map of available datasets
AVAILABLE_DATASETS = {
    "mmlu": import_mmlu,
    "truthfulqa": import_truthfulqa,
    "gsm8k": import_gsm8k,
    "hellaswag": import_hellaswag,
}
