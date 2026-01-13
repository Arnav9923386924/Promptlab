"""PromptLab CLI - Main entry point."""

import typer
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
import yaml

from promptlab import __version__

# Initialize Typer app
app = typer.Typer(
    name="promptlab",
    help="CI/CD for LLM Applications - Test prompts with LLM Council evaluation",
    add_completion=False,
)

console = Console()

# Default config template
DEFAULT_CONFIG = """# PromptLab Configuration
version: 1

# Models for running your prompts
models:
  default: ollama/llama3.1:8b
  providers:
    ollama:
      endpoint: http://localhost:11434
    # openrouter:
    #   api_key: ${OPENROUTER_API_KEY}

# LLM Council for evaluation
council:
  enabled: true
  mode: fast  # full | fast | vote
  members:
    - ollama/llama3.1:8b
    - ollama/mistral:7b
  chairman: ollama/llama3.1:8b

# Test execution settings
testing:
  parallelism: 4
  timeout_ms: 30000
  retries: 2
"""

# Example test template
EXAMPLE_TEST = """# Example test suite
metadata:
  name: Example Tests
  
defaults:
  model: ollama/llama3.1:8b
  temperature: 0

cases:
  - id: hello-world
    prompt: "Say hello in one word."
    assertions:
      - type: contains
        value: hello
        case_sensitive: false
        
  - id: math-simple
    prompt: "What is 2 + 2? Answer with just the number."
    assertions:
      - type: contains
        value: "4"
"""


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"[bold blue]promptlab[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """PromptLab - CI/CD for LLM Applications."""
    pass


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Initialize PromptLab in the current directory."""
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    tests_dir = cwd / "tests"
    example_test_path = tests_dir / "example.yaml"
    promptlab_dir = cwd / ".promptlab"
    
    # Check if already initialized
    if config_path.exists() and not force:
        console.print("[yellow]⚠️  promptlab.yaml already exists. Use --force to overwrite.[/yellow]")
        raise typer.Exit(1)
    
    # Create config file
    config_path.write_text(DEFAULT_CONFIG)
    console.print(f"[green]✓[/green] Created {config_path.name}")
    
    # Create tests directory
    tests_dir.mkdir(exist_ok=True)
    console.print(f"[green]✓[/green] Created {tests_dir.name}/ directory")
    
    # Create example test
    if not example_test_path.exists() or force:
        example_test_path.write_text(EXAMPLE_TEST)
        console.print(f"[green]✓[/green] Created {tests_dir.name}/example.yaml")
    
    # Create .promptlab directory
    promptlab_dir.mkdir(exist_ok=True)
    (promptlab_dir / "baselines").mkdir(exist_ok=True)
    (promptlab_dir / "runs").mkdir(exist_ok=True)
    console.print(f"[green]✓[/green] Created {promptlab_dir.name}/ directory")
    
    # Create .gitignore for .promptlab
    gitignore_path = promptlab_dir / ".gitignore"
    gitignore_path.write_text("*\n!.gitignore\n")
    
    console.print()
    console.print(Panel(
        "[bold green]PromptLab initialized![/bold green]\n\n"
        "Next steps:\n"
        "  1. Edit [cyan]promptlab.yaml[/cyan] to configure your models\n"
        "  2. Write tests in [cyan]tests/[/cyan] directory\n"
        "  3. Run [cyan]promptlab test[/cyan] to execute tests",
        title="🧪 PromptLab",
        border_style="green",
    ))


@app.command()
def test(
    files: list[str] = typer.Argument(None, help="Test files to run (default: all)"),
    stage: str = typer.Option(None, "--stage", "-s", help="Run tests for specific pipeline stage"),
    baseline: str = typer.Option(None, "--baseline", "-b", help="Compare against baseline"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch mode - re-run on changes"),
    ci: bool = typer.Option(False, "--ci", help="CI mode - machine-readable output"),
    no_council: bool = typer.Option(False, "--no-council", help="Skip council evaluation"),
):
    """Run test suites."""
    import asyncio
    from promptlab.utils.config import load_config
    from promptlab.core.runner import TestRunner, print_results
    
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    tests_dir = cwd / "tests"
    
    # Check if initialized
    if not config_path.exists():
        console.print("[red]✗ Not a PromptLab project. Run 'promptlab init' first.[/red]")
        raise typer.Exit(1)
    
    # Load config
    config = load_config(config_path)
    
    # Disable council if requested
    if no_council:
        config.council.enabled = False
    
    console.print(Panel(
        f"[bold]Model:[/bold] {config.models.default}\n"
        f"[bold]Council:[/bold] {'enabled' if config.council.enabled else 'disabled'}\n"
        f"[bold]Stage:[/bold] {stage or 'all'}",
        title="🧪 PromptLab Test",
        border_style="blue",
    ))
    
    # Run tests
    runner = TestRunner(config)
    
    try:
        run = asyncio.run(runner.run_all(tests_dir, files, stage))
        print_results(run)
        
        # Exit with appropriate code
        if run.summary.failed > 0 or run.summary.errors > 0:
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error running tests: {e}[/red]")
        raise typer.Exit(2)


@app.command()
def baseline(
    action: str = typer.Argument(..., help="Action: create, list, delete"),
    tag: str = typer.Option(None, "--tag", "-t", help="Baseline tag name"),
):
    """Manage test baselines."""
    from promptlab.core.baseline import BaselineManager
    
    cwd = Path.cwd()
    promptlab_dir = cwd / ".promptlab"
    
    if not promptlab_dir.exists():
        console.print("[red]✗ Not a PromptLab project. Run 'promptlab init' first.[/red]")
        raise typer.Exit(1)
    
    manager = BaselineManager(promptlab_dir)
    
    if action == "list":
        baselines = manager.list_baselines()
        if not baselines:
            console.print("[yellow]No baselines found.[/yellow]")
        else:
            console.print("[bold]Available baselines:[/bold]")
            for b in baselines:
                console.print(f"  • {b}")
    
    elif action == "create":
        if not tag:
            console.print("[red]✗ --tag is required for create action[/red]")
            raise typer.Exit(1)
        
        # Need to run tests first to get results
        console.print(f"[yellow]To create baseline, first run tests then use:[/yellow]")
        console.print(f"  promptlab test --save-baseline {tag}")
    
    elif action == "delete":
        if not tag:
            console.print("[red]✗ --tag is required for delete action[/red]")
            raise typer.Exit(1)
        
        if manager.delete_baseline(tag):
            console.print(f"[green]✓ Deleted baseline '{tag}'[/green]")
        else:
            console.print(f"[red]✗ Baseline '{tag}' not found[/red]")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: create, list, delete")
        raise typer.Exit(1)


@app.command("import")
def import_data(
    source: str = typer.Argument(..., help="Path to CSV or JSONL file"),
    output: str = typer.Option(None, "--output", "-o", help="Output YAML file name"),
    format: str = typer.Option("csv", "--format", "-f", help="Input format: csv, jsonl"),
):
    """Import test cases from CSV or JSONL files."""
    from promptlab.utils.importers import import_from_csv, import_from_jsonl
    
    cwd = Path.cwd()
    source_path = Path(source)
    
    if not source_path.exists():
        console.print(f"[red]✗ File not found: {source}[/red]")
        raise typer.Exit(1)
    
    # Determine output path
    if output:
        output_path = cwd / "tests" / output
    else:
        output_path = cwd / "tests" / f"{source_path.stem}.yaml"
    
    # Import based on format
    if format == "csv":
        count = import_from_csv(source_path, output_path)
    elif format == "jsonl":
        count = import_from_jsonl(source_path, output_path)
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓ Imported {count} test cases to {output_path}[/green]")


GITHUB_WORKFLOW = '''# PromptLab CI/CD Workflow
name: Prompt Tests

on:
  pull_request:
    paths: ['prompts/**', 'tests/**', 'promptlab.yaml']
  push:
    branches: [main]
    paths: ['prompts/**', 'tests/**', 'promptlab.yaml']

jobs:
  prompt-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install PromptLab
        run: pip install promptlab
      
      - name: Install Ollama
        run: |
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama serve &
          sleep 5
          ollama pull llama3.1:8b
      
      - name: Run Tests
        run: promptlab test --ci
'''


@app.command("ci-setup")
def ci_setup():
    """Generate GitHub Actions workflow for CI/CD."""
    cwd = Path.cwd()
    workflows_dir = cwd / ".github" / "workflows"
    workflow_path = workflows_dir / "prompt-tests.yml"
    
    # Create directories
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Write workflow
    workflow_path.write_text(GITHUB_WORKFLOW)
    
    console.print(f"[green]✓ Created {workflow_path}[/green]")
    console.print()
    console.print(Panel(
        "[bold]GitHub Actions workflow created![/bold]\n\n"
        "The workflow will:\n"
        "  • Run on PRs that modify prompts/tests\n"
        "  • Install Ollama and run tests\n"
        "  • Block PRs if tests fail\n\n"
        "Commit and push to enable CI/CD.",
        title="🚀 CI/CD Setup",
        border_style="green",
    ))


@app.command("benchmark")
def benchmark(
    dataset: str = typer.Argument(..., help="Dataset: mmlu, truthfulqa, gsm8k, hellaswag, or 'all'"),
    samples: int = typer.Option(20, "--samples", "-n", help="Max samples per dataset"),
):
    """Download standard LLM benchmarks from HuggingFace."""
    from promptlab.utils.datasets import AVAILABLE_DATASETS
    
    cwd = Path.cwd()
    tests_dir = cwd / "tests"
    tests_dir.mkdir(exist_ok=True)
    
    console.print(Panel(
        f"[bold]Dataset:[/bold] {dataset}\n"
        f"[bold]Samples:[/bold] {samples} per dataset\n"
        f"[bold]Output:[/bold] {tests_dir}/",
        title="📊 Downloading Benchmark",
        border_style="blue",
    ))
    
    total = 0
    
    if dataset == "all":
        for name, importer in AVAILABLE_DATASETS.items():
            try:
                count = importer(tests_dir, max_samples=samples)
                total += count
            except Exception as e:
                console.print(f"[red]Error importing {name}: {e}[/red]")
    elif dataset in AVAILABLE_DATASETS:
        total = AVAILABLE_DATASETS[dataset](tests_dir, max_samples=samples)
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print(f"Available: {', '.join(AVAILABLE_DATASETS.keys())}, all")
        raise typer.Exit(1)
    
    console.print()
    console.print(f"[green]✓ Imported {total} test cases total[/green]")
    console.print(f"[cyan]Run: promptlab test[/cyan]")


if __name__ == "__main__":
    app()


