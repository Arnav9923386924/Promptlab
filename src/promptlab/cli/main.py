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
        "  2. Write tests in [cyan]temp/[/cyan] directory\n"
        "  3. Run [cyan]promptlab test[/cyan] to execute tests",
        title="PromptLab",
        border_style="green",
    ))



GITHUB_WORKFLOW = '''# PromptLab CI/CD Workflow
name: Prompt Validation

on:
  pull_request:
    paths: ['prompts/**', 'temp/**', 'promptlab.yaml', 'bsp/**']
  push:
    branches: [main]
    paths: ['prompts/**', 'temp/**', 'promptlab.yaml', 'bsp/**']

jobs:
  validate:
    runs-on: ubuntu-latest
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
        run: promptlab validate --ci --output validation-result.json
      
      - name: Upload Validation Results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: |
            validation-result.json
            .promptlab/runs/
      
      - name: Auto-Push on Improvement (optional)
        if: success() && github.ref == 'refs/heads/main'
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
        "[cyan]Your CI/CD is ready![/cyan]",
        title="CI/CD Setup Complete",
        border_style="green",
    ))



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
        title="Setup Complete",
        border_style="green",
    ))
    console.print()
    console.print("[cyan]Next: promptlab run performance[/cyan]")
    console.print("[cyan]  or: promptlab run roleplay[/cyan]")


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
        title="BSP Validation",
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
                title="Success",
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



if __name__ == "__main__":
    app()

