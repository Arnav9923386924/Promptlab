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
    temp_dir = cwd / "temp"
    example_test_path = temp_dir / "example.yaml"
    promptlab_dir = cwd / ".promptlab"
    
    # Check if already initialized
    if config_path.exists() and not force:
        console.print("[yellow]⚠️  promptlab.yaml already exists. Use --force to overwrite.[/yellow]")
        raise typer.Exit(1)
    
    # Create config file
    config_path.write_text(DEFAULT_CONFIG)
    console.print(f"[green]✓[/green] Created {config_path.name}")
    
    # Create temp directory
    temp_dir.mkdir(exist_ok=True)
    console.print(f"[green]✓[/green] Created {temp_dir.name}/ directory")
    
    # Create example test
    if not example_test_path.exists() or force:
        example_test_path.write_text(EXAMPLE_TEST)
        console.print(f"[green]✓[/green] Created {temp_dir.name}/example.yaml")
    
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
    from promptlab.orchestrators.runner import TestRunner, print_results
    
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    temp_dir = cwd / "temp"
    
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
        run = asyncio.run(runner.run_all(temp_dir, files, stage))
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
    from promptlab.orchestrators.baseline import BaselineManager
    
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
        output_path = cwd / "temp" / output
    else:
        output_path = cwd / "temp" / f"{source_path.stem}.yaml"
    
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
    paths: ['prompts/**', 'tests/**', 'promptlab.yaml', 'bsp/**']
  push:
    branches: [main]
    paths: ['prompts/**', 'tests/**', 'promptlab.yaml', 'bsp/**']

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
      
  bsp-validation:
    runs-on: ubuntu-latest
    needs: prompt-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      
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
          ollama pull mistral:7b
      
      - name: Run BSP Validation
        run: |
          promptlab validate --ci --output validation-result.json
      
      - name: Upload Validation Results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: |
            validation-result.json
            .promptlab/runs/
      
      - name: Auto-Push on Improvement (optional)
        if: success()
        run: |
          # Uncomment to enable auto-push when score improves
          # git config user.name github-actions
          # git config user.email github-actions@github.com
          # promptlab validate --push --ci
          echo "BSP Validation passed! Enable auto-push in workflow to automatically update baselines."
'''


@app.command("ci-setup")
def ci_setup(
    enable_bsp_validation: bool = typer.Option(True, "--bsp/--no-bsp", help="Include BSP validation job"),
    enable_auto_push: bool = typer.Option(False, "--auto-push", help="Enable auto-push when BSP improves"),
):
    """Generate GitHub Actions workflow for CI/CD.
    
    Examples:
      promptlab ci-setup                    # Basic CI with BSP validation
      promptlab ci-setup --auto-push        # Enable auto-push on improvement
      promptlab ci-setup --no-bsp           # Skip BSP validation job
    """
    cwd = Path.cwd()
    workflows_dir = cwd / ".github" / "workflows"
    workflow_path = workflows_dir / "prompt-tests.yml"
    
    # Check if promptlab.yaml exists
    if not (cwd / "promptlab.yaml").exists():
        console.print("[yellow]⚠️  No promptlab.yaml found. Run 'promptlab init' first.[/yellow]")
        if not typer.confirm("Continue anyway?", default=False):
            raise typer.Exit(0)
    
    # Create directories
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Modify workflow based on options
    workflow_content = GITHUB_WORKFLOW
    
    if not enable_bsp_validation:
        # Remove BSP validation job
        lines = workflow_content.split('\n')
        bsp_start = None
        for i, line in enumerate(lines):
            if 'bsp-validation:' in line:
                bsp_start = i
                break
        if bsp_start:
            workflow_content = '\n'.join(lines[:bsp_start])
    
    if enable_auto_push and enable_bsp_validation:
        # Enable auto-push in the workflow
        workflow_content = workflow_content.replace(
            "# Uncomment to enable auto-push when score improves\n          # git config user.name github-actions\n          # git config user.email github-actions@github.com\n          # promptlab validate --push --ci\n          echo \"BSP Validation passed! Enable auto-push in workflow to automatically update baselines.\"",
            "git config user.name github-actions\n          git config user.email github-actions@github.com\n          promptlab validate --push --ci"
        )
    
    # Write workflow
    workflow_path.write_text(workflow_content)
    
    console.print(f"[green]✓ Created {workflow_path}[/green]")
    console.print()
    
    # Build feature list
    features = [
        "• Run on PRs that modify prompts/tests/BSP",
        "• Install PromptLab from pip automatically",
        "• Install Ollama and run tests",
        "• Block PRs if tests fail",
    ]
    
    if enable_bsp_validation:
        features.append("• BSP validation on main branch pushes")
        features.append("• Upload validation artifacts")
        if enable_auto_push:
            features.append("• [bold cyan]Auto-push enabled[/bold cyan] when BSP score improves")
        else:
            features.append("• Auto-push [dim](disabled - use --auto-push to enable)[/dim]")
    
    console.print(Panel(
        "[bold green]GitHub Actions workflow created![/bold green]\n\n"
        "Features enabled:\n" + "\n".join(features) + "\n\n"
        "[dim]Next steps:[/dim]\n"
        "  1. git add .github/workflows/prompt-tests.yml\n"
        "  2. git commit -m 'ci: add PromptLab workflow'\n"
        "  3. git push\n\n"
        "[cyan]Your CI/CD is ready! 🚀[/cyan]",
        title="🚀 CI/CD Setup Complete",
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
    temp_dir = cwd / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    console.print(Panel(
        f"[bold]Dataset:[/bold] {dataset}\n"
        f"[bold]Samples:[/bold] {samples} per dataset\n"
        f"[bold]Output:[/bold] {temp_dir}/",
        title="📊 Downloading Benchmark",
        border_style="blue",
    ))
    
    total = 0
    
    if dataset == "all":
        for name, importer in AVAILABLE_DATASETS.items():
            try:
                count = importer(temp_dir, max_samples=samples)
                total += count
            except Exception as e:
                console.print(f"[red]Error importing {name}: {e}[/red]")
    elif dataset in AVAILABLE_DATASETS:
        total = AVAILABLE_DATASETS[dataset](temp_dir, max_samples=samples)
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print(f"Available: {', '.join(AVAILABLE_DATASETS.keys())}, all")
        raise typer.Exit(1)
    
    console.print()
    console.print(f"[green]✓ Imported {total} test cases total[/green]")
    console.print(f"[cyan]Run: promptlab test[/cyan]")


# Quick config for API setup
QUICK_CONFIG_TEMPLATE = """# PromptLab Quick Config
version: 1

models:
  default: {provider}/{model}
  providers:
    {provider}:
      {config_key}: {config_value}

council:
  enabled: false

testing:
  parallelism: 4
  timeout_ms: 30000
"""


@app.command("setup")
def setup(
    provider: str = typer.Argument(..., help="Provider: ollama, openrouter"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key for openrouter"),
    endpoint: str = typer.Option("http://localhost:11434", "--endpoint", "-e", help="Endpoint for ollama"),
    model: str = typer.Option(None, "--model", "-m", help="Model to use"),
):
    """Quick setup - configure API keys and model in one command."""
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    temp_dir = cwd / "temp"
    
    # Determine config based on provider
    if provider == "openrouter":
        if not api_key:
            console.print("[red]✗ OpenRouter requires --api-key[/red]")
            raise typer.Exit(1)
        config_key = "api_key"
        config_value = api_key
        default_model = model or "google/gemini-2.0-flash-001"
    elif provider == "ollama":
        config_key = "endpoint"
        config_value = endpoint
        default_model = model or "llama3.1:8b"
    else:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print("Supported: ollama, openrouter")
        raise typer.Exit(1)
    
    # Create config
    config_content = QUICK_CONFIG_TEMPLATE.format(
        provider=provider,
        model=default_model,
        config_key=config_key,
        config_value=config_value,
    )
    
    config_path.write_text(config_content)
    temp_dir.mkdir(exist_ok=True)
    
    console.print(Panel(
        f"[bold]Provider:[/bold] {provider}\n"
        f"[bold]Model:[/bold] {default_model}\n"
        f"[bold]Config:[/bold] {config_path}",
        title="✓ Setup Complete",
        border_style="green",
    ))
    console.print()
    console.print("[cyan]Next: promptlab run performance[/cyan]")
    console.print("[cyan]  or: promptlab run roleplay[/cyan]")


@app.command("run")
def run_quick(
    mode: str = typer.Argument(..., help="Mode: 'roleplay' or 'performance'"),
    samples: int = typer.Option(10, "--samples", "-n", help="Number of test samples"),
    role: str = typer.Option(None, "--role", "-r", help="Role for dynamic test generation"),
):
    """Quick test - run roleplay or performance tests.
    
    For roleplay with dynamic generation:
      promptlab run roleplay --role "You are a Python expert"
    """
    import asyncio
    from promptlab.utils.config import load_config
    from promptlab.orchestrators.runner import TestRunner, print_results
    from promptlab.llm_council.llm_runner.runner import LLMRunner
    
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    temp_dir = cwd / "temp"
    
    # Check setup
    if not config_path.exists():
        console.print("[red]✗ Run 'promptlab setup' first[/red]")
        raise typer.Exit(1)
    
    config = load_config(config_path)
    temp_dir.mkdir(exist_ok=True)
    
    # If --role is provided for roleplay, generate tests dynamically
    if mode == "roleplay" and role:
        from promptlab.utils.test_generator import generate_tests_for_role, clean_generated_tests
        
        # Determine generator model
        generator_model = config.models.generator or config.models.default
        if not config.models.generator:
            console.print(f"[yellow]No generator model set. Using default: {generator_model}[/yellow]")
            console.print("[dim]Tip: Set 'generator' in models config for better results[/dim]")
        
        console.print(Panel(
            f"[bold]Role:[/bold] {role}\n"
            f"[bold]Generator:[/bold] {generator_model}\n"
            f"[bold]Target:[/bold] {config.models.default}",
            title="🎭 Dynamic Roleplay Test Generation",
            border_style="blue",
        ))
        
        # Clean old generated tests
        clean_generated_tests(temp_dir)
        
        # Create LLM runner and generate tests
        llm_runner = LLMRunner({
            "default": config.models.default,
            "providers": {
                name: {"endpoint": p.endpoint, "api_key": p.api_key}
                for name, p in config.models.providers.items()
            },
        })
        
        generated_files = asyncio.run(
            generate_tests_for_role(role, llm_runner, generator_model, temp_dir)
        )
        
        if not generated_files:
            console.print("[red]✗ Failed to generate tests[/red]")
            raise typer.Exit(1)
        
        console.print(f"\n[green]✓ Generated {len(generated_files)} test file(s)[/green]\n")
        
        # Run the generated tests
        runner = TestRunner(config)
        test_paths = [f"temp/{f.name}" for f in generated_files]
        run_result = asyncio.run(runner.run_all(temp_dir, test_paths))
        print_results(run_result)
        return
    
    if mode == "performance":
        # Download and run benchmark
        console.print(Panel(
            "[bold]Mode:[/bold] Performance Testing\n"
            "[bold]Dataset:[/bold] GSM8K (Math Reasoning)\n"
            f"[bold]Samples:[/bold] {samples}",
            title="🚀 Running Performance Test",
            border_style="blue",
        ))
        
        from promptlab.utils.datasets import import_gsm8k
        import_gsm8k(temp_dir, max_samples=samples)
        
        # Run tests
        config = load_config(config_path)
        runner = TestRunner(config)
        run_result = asyncio.run(runner.run_all(temp_dir, ["temp/gsm8k.yaml"]))
        print_results(run_result)
        
    elif mode == "roleplay":
        # Find all roleplay tests (exclude benchmark files)
        benchmark_prefixes = ["gsm8k", "mmlu", "hellaswag", "truthfulqa"]
        
        roleplay_files = []
        if temp_dir.exists():
            for f in temp_dir.glob("*.yaml"):
                # Skip benchmark files
                if not any(f.name.startswith(prefix) for prefix in benchmark_prefixes):
                    roleplay_files.append(f.name)
        
        if not roleplay_files:
            console.print("[yellow]No roleplay tests found. Creating sample tests...[/yellow]")
            # Create sample roleplay tests
            sample_tests = {
                "metadata": {"name": "Sample Roleplay Tests"},
                "defaults": {"temperature": 0},
                "cases": [
                    {
                        "id": "sentiment-positive",
                        "prompt": "Classify as POSITIVE, NEGATIVE, or NEUTRAL. Reply one word.\nText: I love this product!",
                        "assertions": [{"type": "regex", "pattern": "(?i)positive"}]
                    },
                    {
                        "id": "sentiment-negative",
                        "prompt": "Classify as POSITIVE, NEGATIVE, or NEUTRAL. Reply one word.\nText: This is terrible!",
                        "assertions": [{"type": "regex", "pattern": "(?i)negative"}]
                    },
                    {
                        "id": "support-billing",
                        "prompt": "Classify into BILLING, TECHNICAL, or SALES. Reply one word.\nMessage: I need a refund.",
                        "assertions": [{"type": "regex", "pattern": "(?i)billing"}]
                    },
                    {
                        "id": "instruction-follow",
                        "prompt": "Say only the word 'hello'. Nothing else.",
                        "assertions": [{"type": "regex", "pattern": "(?i)hello"}]
                    },
                ]
            }
            sample_path = temp_dir / "roleplay_sample.yaml"
            with open(sample_path, "w", encoding="utf-8") as f:
                yaml.dump(sample_tests, f, default_flow_style=False)
            roleplay_files = ["roleplay_sample.yaml"]
        
        console.print(Panel(
            "[bold]Mode:[/bold] Roleplay Testing\n"
            f"[bold]Test Files:[/bold] {len(roleplay_files)}\n"
            f"[bold]Files:[/bold] {', '.join(roleplay_files[:5])}{'...' if len(roleplay_files) > 5 else ''}",
            title="🎭 Running Roleplay Test",
            border_style="blue",
        ))
        
        # Run all roleplay tests
        config = load_config(config_path)
        runner = TestRunner(config)
        test_paths = [f"temp/{f}" for f in roleplay_files]
        run_result = asyncio.run(runner.run_all(temp_dir, test_paths))
        print_results(run_result)
        
    else:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        console.print("Use: performance, roleplay")
        raise typer.Exit(1)


@app.command("validate")
def validate_bsp(
    test_dir: str = typer.Option("temp", "--dir", "-d", help="Directory containing test files"),
    files: list[str] = typer.Option(None, "--file", "-f", help="Specific test files to run"),
    save_baseline: bool = typer.Option(True, "--save-baseline/--no-save-baseline", help="Save as new baseline if improved"),
    auto_push: bool = typer.Option(False, "--push", "-p", help="Auto-push to git if score improves"),
    ci: bool = typer.Option(False, "--ci", help="CI mode - exit with code based on validation result"),
    output_json: str = typer.Option(None, "--output", "-o", help="Save validation result to JSON file"),
    generate: int = typer.Option(None, "--generate", "-g", help="Auto-generate N test cases via web scraping (default: 50 if no tests exist)"),
    no_generate: bool = typer.Option(False, "--no-generate", help="Disable auto-generation even if no tests exist"),
):
    """Validate your Behavior Specification Prompt (BSP) against test cases.
    
    This command runs the complete BSP validation workflow:
    1. Load BSP from config (promptlab.yaml)
    2. Auto-generate tests if none exist (via web scraping)
    3. Run all test cases with BSP prepended
    4. Collect outputs for council review
    5. Submit to LLM Council for batch evaluation
    6. Compare score with baseline
    7. Optionally push to git if improved
    
    Examples:
      promptlab validate                        # Run validation (auto-generates if no tests)
      promptlab validate --generate 100         # Generate 100 tests via scraping
      promptlab validate --no-generate          # Skip auto-generation, require manual tests
      promptlab validate --push                 # Push to git if improved
      promptlab validate --ci                   # CI mode for GitHub Actions
      promptlab validate -f temp/my_test.yaml   # Validate specific file
    """
    import asyncio
    import json as json_module
    from promptlab.utils.config import load_config, load_bsp
    from promptlab.orchestrators.bsp_validator import BSPValidator
    from promptlab.orchestrators.baseline import BaselineManager
    from promptlab.utils.git_integration import GitIntegration
    
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    test_path = cwd / test_dir
    
    # Check if initialized
    if not config_path.exists():
        console.print("[red]✗ Not a PromptLab project. Run 'promptlab init' first.[/red]")
        raise typer.Exit(1)
    
    # Load config
    config = load_config(config_path)
    
    # Check if BSP is configured
    bsp = load_bsp(config, cwd)
    if not bsp:
        console.print("[yellow]⚠️  No BSP configured. Add 'bsp' section to promptlab.yaml[/yellow]")
        console.print("[dim]Example:[/dim]")
        console.print("[dim]  bsp:[/dim]")
        console.print("[dim]    prompt: 'You are a helpful assistant...'[/dim]")
        console.print("[dim]    min_score: 0.7[/dim]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold]BSP Version:[/bold] {config.bsp.version if config.bsp else 'default'}\n"
        f"[bold]Model:[/bold] {config.models.default}\n"
        f"[bold]Council:[/bold] {'enabled' if config.council.enabled else 'disabled'}\n"
        f"[bold]Test Dir:[/bold] {test_path}\n"
        f"[bold]Auto Generate:[/bold] {'disabled' if no_generate else f'{generate or 50} tests'}\n"
        f"[bold]Auto Push:[/bold] {'yes' if auto_push else 'no'}",
        title="🔍 BSP Validation",
        border_style="blue",
    ))
    
    async def run_validation():
        # Create validator
        validator = BSPValidator(config, cwd)
        
        # Determine auto-generation settings
        should_auto_generate = not no_generate
        target_count = generate if generate else 50
        
        # Run validation
        result = await validator.validate(
            test_dir=test_path,
            test_files=list(files) if files else None,
            auto_generate=should_auto_generate,
            generate_count=target_count,
        )
        
        return result
    
    try:
        result = asyncio.run(run_validation())
        
        # Save baseline if improved
        if save_baseline and result.should_push:
            baseline_manager = BaselineManager(cwd / ".promptlab")
            baseline_manager.save_bsp_baseline(
                run_id=result.run_id,
                score=result.council_score,
                bsp_hash=result.bsp_hash,
                bsp_version=result.bsp_version,
                model=result.model,
                total_tests=result.total_tests,
                passed=result.passed,
                confidence=result.council_result.confidence if result.council_result else "medium",
                tag=f"bsp_{result.bsp_version}_{result.bsp_hash[:8]}",
            )
            console.print("[green]✓ Saved new baseline[/green]")
        
        # Auto-push to git if enabled
        if auto_push and result.should_push:
            git = GitIntegration(cwd)
            commit_template = config.git.commit_template if config.git else "chore: BSP validation passed (score: {score:.2f})"
            git.commit_and_push(
                score=result.council_score,
                previous_score=result.baseline_score,
                commit_template=commit_template,
                branch=config.git.branch if config.git else None,
                auto_push=config.git.auto_push if config.git else True,
            )
        elif auto_push and not result.should_push:
            console.print("[yellow]Not pushing - score did not improve sufficiently[/yellow]")
        
        # Save JSON output if requested
        if output_json:
            output_path = Path(output_json)
            with open(output_path, "w", encoding="utf-8") as f:
                json_module.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            console.print(f"[green]✓ Saved result to {output_path}[/green]")
        
        # Final summary
        console.print()
        if result.passed:
            console.print(Panel(
                f"[bold green]✓ Validation PASSED[/bold green]\n\n"
                f"Score: {result.council_score:.2f}\n"
                f"{'Improvement: ' + f'+{result.improvement:.2f}' if result.improvement and result.improvement > 0 else ''}\n"
                f"Outputs: {result.outputs_file}",
                title="🎉 Success",
                border_style="green",
            ))
        else:
            console.print(Panel(
                f"[bold red]✗ Validation FAILED[/bold red]\n\n"
                f"Score: {result.council_score:.2f}\n"
                f"Min Required: {config.bsp.min_score if config.bsp else 0.7}\n"
                f"Outputs: {result.outputs_file}",
                title="❌ Failed",
                border_style="red",
            ))
        
        # Exit code for CI
        if ci:
            raise typer.Exit(0 if result.passed else 1)
            
    except Exception as e:
        console.print(f"[red]Error during validation: {e}[/red]")
        if ci:
            raise typer.Exit(2)
        raise typer.Exit(1)


@app.command("generate")
def generate_tests(
    count: int = typer.Option(50, "--count", "-n", help="Number of test cases to generate"),
    output_dir: str = typer.Option("temp", "--output", "-o", help="Output directory for generated tests"),
    max_pages: int = typer.Option(20, "--pages", "-p", help="Maximum pages to scrape"),
):
    """Auto-generate test cases from your BSP via web scraping.
    
    This command analyzes your BSP (Behavior Specification Prompt) to:
    1. Extract domain, keywords, and capabilities
    2. Generate smart search queries
    3. Scrape relevant web content
    4. Create 50-100 test cases automatically
    
    No manual test writing required!
    
    Examples:
      promptlab generate                     # Generate 50 tests from BSP
      promptlab generate --count 100         # Generate 100 tests
      promptlab generate -n 25 -p 10         # Generate 25 tests from 10 pages
    """
    import asyncio
    from promptlab.utils.config import load_config, load_bsp
    from promptlab.utils.auto_test_generator import AutoTestGenerator
    
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    out_path = cwd / output_dir
    
    # Check if initialized
    if not config_path.exists():
        console.print("[red]✗ Not a PromptLab project. Run 'promptlab init' first.[/red]")
        raise typer.Exit(1)
    
    # Load config
    config = load_config(config_path)
    
    # Load BSP
    bsp = load_bsp(config, cwd)
    if not bsp:
        console.print("[red]✗ No BSP configured. Add 'bsp' section to promptlab.yaml[/red]")
        console.print("[dim]Example:[/dim]")
        console.print("[dim]  bsp:[/dim]")
        console.print("[dim]    prompt: 'You are a helpful assistant...'[/dim]")
        console.print("[dim]    # OR[/dim]")
        console.print("[dim]    prompt_file: bsp.txt[/dim]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold]Target Tests:[/bold] {count}\n"
        f"[bold]Max Pages:[/bold] {max_pages}\n"
        f"[bold]Output:[/bold] {out_path}/\n"
        f"[bold]BSP Preview:[/bold] {bsp[:100]}...",
        title="🧪 Auto-Generate Tests from BSP",
        border_style="blue",
    ))
    
    async def run_generate():
        serpapi_key = config.scraper.serpapi_key if config.scraper else None
        
        generator = AutoTestGenerator(
            serpapi_key=serpapi_key,
            max_pages=max_pages,
        )
        
        result = await generator.generate_tests(
            bsp=bsp,
            target_count=count,
            output_dir=out_path,
        )
        
        return result
    
    try:
        result = asyncio.run(run_generate())
        
        console.print()
        console.print(Panel(
            f"[bold green]✓ Tests generated successfully![/bold green]\n\n"
            f"Domain: {result.domain}\n"
            f"Q&A Tests: {len(result.qa_pairs)}\n"
            f"Cloze Tests: {len(result.masked_tests)}\n"
            f"Total: {len(result.qa_pairs) + len(result.masked_tests)} tests\n"
            f"Sources Scraped: {result.scraped_sources}\n"
            f"Time: {result.generation_time:.1f}s\n\n"
            f"Output: [cyan]{result.output_file}[/cyan]\n\n"
            f"[dim]Next: promptlab validate[/dim]",
            title="🎉 Generation Complete",
            border_style="green",
        ))
        
    except Exception as e:
        console.print(f"[red]Error generating tests: {e}[/red]")
        raise typer.Exit(1)


@app.command("scrape")
def scrape_data(
    domain: str = typer.Argument(..., help="Domain/topic to scrape (e.g., 'legal contracts', 'medical diagnosis')"),
    urls: list[str] = typer.Option(None, "--url", "-u", help="Specific URLs to scrape (can be used multiple times)"),
    num_pages: int = typer.Option(10, "--pages", "-p", help="Number of pages to scrape"),
    output_type: str = typer.Option("all", "--output", "-o", help="Output type: benchmark, roleplay, finetune, all, masked"),
    keep_temp: bool = typer.Option(False, "--keep-temp", help="Keep temporary scraped files (for debugging)"),
):
    """Scrape web content and generate test cases for a domain.
    
    Examples:
      promptlab scrape "legal contracts"
      promptlab scrape "python programming" --pages 20
      promptlab scrape "medical diagnosis" --output benchmark
      promptlab scrape "customer support" -u https://example.com/faq
    
    Scraped content is stored temporarily and automatically cleaned up after processing.
    """
    import asyncio
    from promptlab.utils.scraper import WebScraper, ScraperConfig, scrape_for_domain, cleanup_temp_dir
    from promptlab.utils.data_processor import DataProcessor, save_outputs
    
    cwd = Path.cwd()
    temp_dir = cwd / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    console.print(Panel(
        f"[bold]Domain:[/bold] {domain}\n"
        f"[bold]Pages:[/bold] {num_pages}\n"
        f"[bold]Output:[/bold] {output_type}",
        title="🌐 Web Scraper",
        border_style="blue",
    ))
    
    async def run_scrape():
        # Step 1: Scrape content
        console.print("\n[bold cyan]Step 1: Scraping web content...[/bold cyan]")
        
        if urls:
            scraper = WebScraper(ScraperConfig(max_pages=num_pages))
            contents = await scraper.crawl(urls, max_pages=num_pages)
        else:
            contents = await scrape_for_domain(domain, num_pages=num_pages)
        
        if not contents:
            console.print("[red]✗ No content scraped. Try different URLs or domain.[/red]")
            return None
        
        console.print(f"[green]✓ Scraped {len(contents)} pages[/green]")
        
        # Step 2: Extract Q&A pairs (NO LLM needed!)
        console.print("\n[bold cyan]Step 2: Extracting Q&A pairs...[/bold cyan]")
        
        processor = DataProcessor()  # No LLM required
        results = await processor.process_content(contents, domain, output_type)
        
        # Step 3: Save outputs
        console.print("\n[bold cyan]Step 3: Saving outputs...[/bold cyan]")
        saved_files = save_outputs(results, temp_dir, domain)
        
        return results, saved_files
    
    try:
        result = asyncio.run(run_scrape())
        
        if result:
            results, saved_files = result
            console.print()
            console.print(Panel(
                f"[bold green]Scraping complete![/bold green]\n\n"
                f"Q&A pairs extracted: {len(results['qa_pairs'])}\n"
                f"Persona generated: {'Yes' if results['persona'] else 'No'}\n"
                f"Files created: {len(saved_files)}\n\n"
                f"[cyan]Run: promptlab test[/cyan]",
                title="✓ Success",
                border_style="green",
            ))
    except Exception as e:
        console.print(f"[red]Error during scraping: {e}[/red]")
        raise typer.Exit(1)
    finally:
        # Clean up temporary files unless --keep-temp is specified
        if not keep_temp:
            from promptlab.utils.scraper import cleanup_temp_dir
            cleanup_temp_dir()


if __name__ == "__main__":
    app()
