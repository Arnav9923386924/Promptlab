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


def _ensure_initialized(cwd: Path = None) -> Path:
    """Check if promptlab.yaml exists. If not, offer to run init.
    
    This is called by every command that needs config. If the user
    hasn't run 'promptlab init' yet, this will guide them through
    interactive setup automatically.
    
    Returns:
        Path to the config file
    """
    if cwd is None:
        cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    
    if config_path.exists():
        return config_path
    
    console.print("[yellow]⚠️  No promptlab.yaml found in this directory.[/yellow]")
    console.print()
    
    if typer.confirm("Would you like to initialize PromptLab here?", default=True):
        # Import and call init programmatically
        console.print()
        # We can't call the init command directly, so we'll do inline setup
        _run_interactive_init(cwd)
        return cwd / "promptlab.yaml"
    else:
        console.print("[dim]Run 'promptlab init' to set up your project.[/dim]")
        raise typer.Exit(1)


def _run_interactive_init(cwd: Path):
    """Run interactive initialization — called when auto-init triggers."""
    config_path = cwd / "promptlab.yaml"
    temp_dir = cwd / "temp"
    example_test_path = temp_dir / "example.yaml"
    promptlab_dir = cwd / ".promptlab"
    
    console.print(Panel(
        "[bold blue]Welcome to PromptLab![/bold blue]\n"
        "Let's set up your project configuration.",
        border_style="blue",
    ))
    
    # Ask for provider
    console.print("\n[bold]Which LLM provider do you want to use?[/bold]\n")
    console.print("  1. [bold green]Google AI Studio[/bold green] (Gemini) — FREE, 15 RPM [recommended]")
    console.print("  2. [bold]OpenRouter[/bold] — FREE tier, many models")
    console.print("  3. [bold]Ollama[/bold] — Local, FREE, no key needed")
    console.print("  4. [bold]OpenAI[/bold] — GPT-4o, paid")
    console.print("  5. [bold]Other[/bold] (anthropic, xai)")
    console.print()
    
    choice = typer.prompt("Choose provider (1-5)", default="1")
    
    provider_map = {"1": "google", "2": "openrouter", "3": "ollama", "4": "openai", "5": "ollama"}
    primary = provider_map.get(choice, "google")
    
    if choice == "5":
        primary = typer.prompt("Enter provider name (anthropic, xai)", default="anthropic")
    
    # Get API key
    api_key = None
    if primary != "ollama":
        help_urls = {
            "google": "https://aistudio.google.com/app/apikey",
            "openrouter": "https://openrouter.ai/keys",
            "openai": "https://platform.openai.com/api-keys",
            "anthropic": "https://console.anthropic.com/settings/keys",
            "xai": "https://console.x.ai/",
        }
        if primary in help_urls:
            console.print(f"\n[dim]Get your key at: {help_urls[primary]}[/dim]")
        api_key = typer.prompt(f"API key for {primary}")
        if not api_key:
            console.print("[yellow]No key provided, falling back to ollama[/yellow]")
            primary = "ollama"
    
    # Build config
    block = PROVIDER_BLOCKS.get(primary, "")
    if api_key:
        block = block.format(api_key=api_key)
    
    council_cfg = COUNCIL_CONFIGS.get(primary, COUNCIL_CONFIGS["ollama"])
    council_members = "\n".join(f"    - {m}" for m in council_cfg["members"])
    
    config_content = DEFAULT_CONFIG.format(
        default_model=DEFAULT_MODELS.get(primary, "ollama/llama3.1:8b"),
        generator_model=DEFAULT_MODELS.get(primary, "ollama/llama3.1:8b"),
        providers_block=block,
        council_enabled="true",
        council_members=council_members,
        council_chairman=council_cfg["chairman"],
    )
    
    config_path.write_text(config_content)
    console.print(f"\n[green]✓[/green] Created promptlab.yaml")
    
    # Create bsp.txt
    bsp_path = cwd / "bsp.txt"
    if not bsp_path.exists():
        bsp_path.write_text(BSP_TEMPLATE)
        console.print(f"[green]✓[/green] Created bsp.txt")
    
    # Create root .gitignore
    root_gitignore = cwd / ".gitignore"
    if not root_gitignore.exists():
        root_gitignore.write_text(GITIGNORE_TEMPLATE)
        console.print(f"[green]✓[/green] Created .gitignore")
    
    temp_dir.mkdir(exist_ok=True)
    if not example_test_path.exists():
        example_test_path.write_text(EXAMPLE_TEST)
    console.print(f"[green]✓[/green] Created temp/example.yaml")
    
    promptlab_dir.mkdir(exist_ok=True)
    (promptlab_dir / "baselines").mkdir(exist_ok=True)
    (promptlab_dir / "runs").mkdir(exist_ok=True)
    (promptlab_dir / ".gitignore").write_text("*\n!.gitignore\n")
    console.print(f"[green]✓[/green] Created .promptlab/ directory")
    
    console.print(Panel(
        f"[bold green]PromptLab initialized![/bold green]\n\n"
        f"[bold]Provider:[/bold] {primary}\n"
        f"[bold]Model:[/bold] {DEFAULT_MODELS.get(primary, 'ollama/llama3.1:8b')}\n"
        f"[bold]Council:[/bold] {len(council_cfg['members'])} judges\n\n"
        "Created: promptlab.yaml, bsp.txt, temp/example.yaml\n\n"
        "[dim]Continuing with your command...[/dim]",
        title="PromptLab",
        border_style="green",
    ))

# Default config template — comprehensive with all sections
DEFAULT_CONFIG = """# PromptLab Configuration
# Generated by: promptlab init
version: 1

# =============================================================================
# BEHAVIOR SPECIFICATION PROMPT (BSP)
# =============================================================================
bsp:
  # Load BSP from file (recommended for version control)
  prompt_file: bsp.txt
  
  # Or use inline BSP:
  # prompt: |
  #   You are a helpful assistant.
  
  min_score: 0.7
  version: "1.0.0"

# =============================================================================
# MODELS
# =============================================================================
models:
  default: {default_model}
  generator: {generator_model}
  
  providers:
{providers_block}

# =============================================================================
# LLM COUNCIL
# =============================================================================
council:
  enabled: {council_enabled}
  mode: fast  # full | fast | vote
  members:
{council_members}
  chairman: {council_chairman}

# =============================================================================
# BASELINE
# =============================================================================
baseline:
  storage_dir: .promptlab/baselines
  auto_update: true
  min_improvement: 0.01

# =============================================================================
# GIT INTEGRATION
# =============================================================================
git:
  enabled: true
  branch: null
  commit_template: "chore: BSP validation passed (score: {{score:.2f}}, improvement: +{{improvement:.2f}})"
  auto_push: false

# =============================================================================
# TESTING
# =============================================================================
testing:
  parallelism: 4
  timeout_ms: 30000
  retries: 2

# =============================================================================
# WEB SCRAPER (for auto-generating test cases)
# =============================================================================
scraper:
  serpapi_key: ""
  brave_api_key: ""
  fallback_search: searxng
  max_pages: 5
  timeout: 30
"""


# Provider config blocks for template
PROVIDER_BLOCKS = {
    "ollama": """    ollama:
      endpoint: http://localhost:11434""",
    "openrouter": """    openrouter:
      api_key: {api_key}""",
    "google": """    google:
      api_key: {api_key}""",
    "openai": """    openai:
      api_key: {api_key}""",
    "anthropic": """    anthropic:
      api_key: {api_key}""",
    "xai": """    xai:
      api_key: {api_key}""",
}

# Default models per provider
DEFAULT_MODELS = {
    "ollama": "ollama/llama3.1:8b",
    "openrouter": "openrouter/google/gemini-2.0-flash-001",
    "google": "google/gemini-2.5-flash",
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-haiku-20240307",
    "xai": "xai/grok-2",
}

# Council configs per provider
COUNCIL_CONFIGS = {
    "ollama": {
        "members": ["ollama/llama3.1:8b", "ollama/mistral:7b"],
        "chairman": "ollama/llama3.1:8b",
    },
    "openrouter": {
        "members": ["openrouter/meta-llama/llama-3.3-70b-instruct:free", "openrouter/google/gemma-3-27b-it:free"],
        "chairman": "openrouter/arcee-ai/trinity-large-preview:free",
    },
    "google": {
        "members": ["google/gemini-2.5-flash", "google/gemini-2.0-flash-lite"],
        "chairman": "google/gemini-2.5-pro",
    },
    "openai": {
        "members": ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo"],
        "chairman": "openai/gpt-4o",
    },
    "anthropic": {
        "members": ["anthropic/claude-3-haiku-20240307"],
        "chairman": "anthropic/claude-3-5-sonnet-20241022",
    },
    "xai": {
        "members": ["xai/grok-2"],
        "chairman": "xai/grok-2",
    },
}

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

# BSP template file
BSP_TEMPLATE = """# Behavior Specification Prompt (BSP)
# ============================================================================
# This file defines how your AI assistant should behave.
# PromptLab uses this to test and validate your prompts.
# ============================================================================

# CORE IDENTITY
# -------------
You are a helpful, accurate, and concise assistant.

# BEHAVIORAL GUIDELINES
# --------------------
1. Always provide factual, well-researched information
2. If you're uncertain about something, acknowledge it clearly
3. Keep responses brief and focused unless more detail is requested
4. Use clear, professional language
5. Structure complex answers with headings or bullet points

# CONSTRAINTS
# ----------
- Never fabricate information or citations
- Don't make promises about capabilities you don't have
- Avoid repetitive phrases and filler content
- Don't start responses with "I" when possible

# OUTPUT FORMAT
# -------------
- For simple questions: 1-2 sentence answers
- For explanations: structured paragraphs with clear flow
- For lists: use bullet points or numbered items
- For code: use markdown code blocks with language tags

# TONE
# ----
Professional yet approachable. Helpful without being verbose.
"""

# Root .gitignore template
GITIGNORE_TEMPLATE = """# PromptLab
.promptlab/
promptlab_cache/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
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
    provider: str = typer.Option(None, "--provider", "-p", help="Provider: ollama, openrouter, google, openai, anthropic, xai"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key for the chosen provider"),
    non_interactive: bool = typer.Option(False, "--yes", "-y", help="Non-interactive mode with defaults"),
):
    """Initialize PromptLab in the current directory.
    
    Creates:
      - promptlab.yaml — main configuration
      - bsp.txt — behavior specification template
      - temp/example.yaml — example test suite
      - .promptlab/ — baseline and run storage
      - .gitignore — ignores cache and temp files
    
    Examples:
      promptlab init                                    # Interactive setup
      promptlab init -p google -k AIza...               # Quick setup with Google
      promptlab init -p openrouter -k sk-or-...         # Quick setup with OpenRouter
      promptlab init -p ollama                          # Local Ollama (no key needed)
      promptlab init --yes                              # Non-interactive with defaults
    """
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    temp_dir = cwd / "temp"
    example_test_path = temp_dir / "example.yaml"
    promptlab_dir = cwd / ".promptlab"
    
    # Check if already initialized
    if config_path.exists() and not force:
        console.print("[yellow]⚠️  promptlab.yaml already exists. Use --force to overwrite.[/yellow]")
        raise typer.Exit(1)
    
    console.print(Panel(
        "[bold blue]Welcome to PromptLab![/bold blue]\n"
        "Let's set up your project configuration.",
        border_style="blue",
    ))
    
    # Collect provider info
    providers_selected = {}  # {provider_name: api_key_or_None}
    
    if provider:
        # Quick mode — single provider from CLI args
        if provider in ("openrouter", "google", "openai", "anthropic", "xai"):
            if not api_key:
                console.print(f"[red]✗ {provider} requires --api-key[/red]")
                raise typer.Exit(1)
            providers_selected[provider] = api_key
        elif provider == "ollama":
            providers_selected["ollama"] = None
        else:
            console.print(f"[red]✗ Unknown provider: {provider}[/red]")
            console.print("[dim]Supported: ollama, openrouter, google, openai, anthropic, xai[/dim]")
            raise typer.Exit(1)
    elif non_interactive:
        # Default to ollama
        providers_selected["ollama"] = None
    else:
        # Interactive mode
        console.print("\n[bold]Which LLM providers do you want to use?[/bold]")
        console.print("[dim]You can configure multiple providers. PromptLab will use them[/dim]")
        console.print("[dim]for model fallback if one gets rate-limited.[/dim]\n")
        
        provider_choices = [
            ("google", "Google AI Studio (Gemini) — FREE, 15 RPM, recommended", "Get key: https://aistudio.google.com/app/apikey"),
            ("openrouter", "OpenRouter — FREE tier models, many models", "Get key: https://openrouter.ai/keys"),
            ("ollama", "Ollama — Local, FREE, no key needed", "Install: https://ollama.ai"),
            ("openai", "OpenAI — GPT-4o, paid", ""),
            ("anthropic", "Anthropic — Claude, paid", ""),
            ("xai", "xAI — Grok, paid", ""),
        ]
        
        for name, desc, help_url in provider_choices:
            prompt_text = f"  Use {desc}?"
            use_it = typer.confirm(prompt_text, default=(name in ("google", "openrouter")))
            
            if use_it:
                if name == "ollama":
                    providers_selected[name] = None
                else:
                    if help_url:
                        console.print(f"    [dim]{help_url}[/dim]")
                    key = typer.prompt(f"    API key for {name}", default="", hide_input=False)
                    if key:
                        providers_selected[name] = key
                    else:
                        console.print(f"    [yellow]Skipping {name} (no key provided)[/yellow]")
    
    if not providers_selected:
        console.print("[yellow]No providers selected. Defaulting to ollama (local).[/yellow]")
        providers_selected["ollama"] = None
    
    # Determine primary provider (first selected)
    primary_provider = list(providers_selected.keys())[0]
    
    # Build providers block
    providers_lines = []
    for prov, key in providers_selected.items():
        block = PROVIDER_BLOCKS.get(prov, "")
        if key:
            block = block.format(api_key=key)
        providers_lines.append(block)
    providers_block = "\n".join(providers_lines)
    
    # Pick best council provider (prefer google > openrouter > others for free evaluation)
    council_provider = primary_provider
    for pref in ["google", "openrouter"]:
        if pref in providers_selected:
            council_provider = pref
            break
    
    council_cfg = COUNCIL_CONFIGS.get(council_provider, COUNCIL_CONFIGS["ollama"])
    council_enabled = "true"
    council_members = "\n".join(f"    - {m}" for m in council_cfg["members"])
    council_chairman = council_cfg["chairman"]
    
    # Build config
    config_content = DEFAULT_CONFIG.format(
        default_model=DEFAULT_MODELS.get(primary_provider, "ollama/llama3.1:8b"),
        generator_model=DEFAULT_MODELS.get(primary_provider, "ollama/llama3.1:8b"),
        providers_block=providers_block,
        council_enabled=council_enabled,
        council_members=council_members,
        council_chairman=council_chairman,
    )
    
    # Write config
    config_path.write_text(config_content)
    console.print(f"\n[green]✓[/green] Created {config_path.name}")
    
    # Create bsp.txt
    bsp_path = cwd / "bsp.txt"
    if not bsp_path.exists() or force:
        bsp_path.write_text(BSP_TEMPLATE)
        console.print(f"[green]✓[/green] Created bsp.txt")
    
    # Create root .gitignore
    root_gitignore = cwd / ".gitignore"
    if not root_gitignore.exists():
        root_gitignore.write_text(GITIGNORE_TEMPLATE)
        console.print(f"[green]✓[/green] Created .gitignore")
    
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
    
    # Summary
    providers_summary = ", ".join(providers_selected.keys())
    console.print()
    console.print(Panel(
        f"[bold green]PromptLab initialized![/bold green]\n\n"
        f"[bold]Providers:[/bold] {providers_summary}\n"
        f"[bold]Default model:[/bold] {DEFAULT_MODELS.get(primary_provider, 'ollama/llama3.1:8b')}\n"
        f"[bold]Council:[/bold] {council_provider} ({len(council_cfg['members'])} judges)\n\n"
        "Created files:\n"
        "  • [cyan]promptlab.yaml[/cyan] — main configuration\n"
        "  • [cyan]bsp.txt[/cyan] — your behavior specification\n"
        "  • [cyan]temp/example.yaml[/cyan] — example test suite\n"
        "  • [cyan].promptlab/[/cyan] — baseline storage\n\n"
        "Next steps:\n"
        "  1. Edit [cyan]bsp.txt[/cyan] to define your AI's behavior\n"
        "  2. Run [cyan]promptlab lint-bsp[/cyan] to check BSP quality\n"
        "  3. Run [cyan]promptlab validate[/cyan] to test your prompts",
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
    
    # Ensure project is initialized
    _ensure_initialized(cwd)
    
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



@app.command("setup")
def setup(
    provider: str = typer.Argument(..., help="Provider: ollama, openrouter, google, openai, anthropic, xai"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key"),
    endpoint: str = typer.Option("http://localhost:11434", "--endpoint", "-e", help="Endpoint for ollama"),
    model: str = typer.Option(None, "--model", "-m", help="Model to use"),
):
    """Quick setup - configure API keys and model in one command.
    
    Examples:
      promptlab setup google -k AIza...                   # Setup Google Gemini
      promptlab setup openrouter -k sk-or-...             # Setup OpenRouter
      promptlab setup ollama                              # Setup local Ollama
      promptlab setup google -k AIza... -m gemini-2.5-pro # Use specific model
    """
    cwd = Path.cwd()
    config_path = cwd / "promptlab.yaml"
    temp_dir = cwd / "temp"
    
    # Validate provider
    supported = list(DEFAULT_MODELS.keys())
    if provider not in supported:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"Supported: {', '.join(supported)}")
        raise typer.Exit(1)
    
    # Require API key for cloud providers
    if provider != "ollama" and not api_key:
        console.print(f"[red]✗ {provider} requires --api-key (-k)[/red]")
        
        help_urls = {
            "google": "Get a free key at: https://aistudio.google.com/app/apikey",
            "openrouter": "Get a free key at: https://openrouter.ai/keys",
            "openai": "Get a key at: https://platform.openai.com/api-keys",
            "anthropic": "Get a key at: https://console.anthropic.com/settings/keys",
            "xai": "Get a key at: https://console.x.ai/",
        }
        if provider in help_urls:
            console.print(f"[dim]{help_urls[provider]}[/dim]")
        raise typer.Exit(1)
    
    # Determine default model
    if model:
        default_model = f"{provider}/{model}" if "/" not in model else model
    else:
        default_model = DEFAULT_MODELS[provider]
    
    # Build providers block
    providers_selected = {provider: api_key}
    providers_lines = []
    block = PROVIDER_BLOCKS.get(provider, "")
    if api_key:
        block = block.format(api_key=api_key)
    providers_lines.append(block)
    providers_block = "\n".join(providers_lines)
    
    # Council config
    council_cfg = COUNCIL_CONFIGS.get(provider, COUNCIL_CONFIGS["ollama"])
    council_members = "\n".join(f"    - {m}" for m in council_cfg["members"])
    
    config_content = DEFAULT_CONFIG.format(
        default_model=default_model,
        generator_model=default_model,
        providers_block=providers_block,
        council_enabled="true",
        council_members=council_members,
        council_chairman=council_cfg["chairman"],
    )
    
    config_path.write_text(config_content)
    temp_dir.mkdir(exist_ok=True)
    
    # Create .promptlab dirs
    promptlab_dir = cwd / ".promptlab"
    promptlab_dir.mkdir(exist_ok=True)
    (promptlab_dir / "baselines").mkdir(exist_ok=True)
    (promptlab_dir / "runs").mkdir(exist_ok=True)
    
    console.print(Panel(
        f"[bold]Provider:[/bold] {provider}\n"
        f"[bold]Model:[/bold] {default_model}\n"
        f"[bold]Council:[/bold] {len(council_cfg['members'])} judges + chairman\n"
        f"[bold]Config:[/bold] {config_path}",
        title="Setup Complete",
        border_style="green",
    ))
    console.print()
    console.print("[cyan]Next: promptlab validate[/cyan]")


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
    config_path = _ensure_initialized(cwd)
    test_path = cwd / test_dir
    
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



@app.command("lint-bsp")
def lint_bsp(
    bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file (overrides promptlab.yaml)"),
):
    """Lint your BSP for quality issues — completely FREE, no API calls.
    
    Checks for vague language, missing sections, contradictions,
    and structural problems. Returns a quality score and actionable fixes.
    
    Examples:
      promptlab lint-bsp                  # Lint BSP from promptlab.yaml
      promptlab lint-bsp -b my_bsp.txt    # Lint a specific BSP file
    """
    from promptlab.utils.config import load_config, load_bsp
    from promptlab.utils.bsp_linter import BSPLinter

    cwd = Path.cwd()

    # Load BSP
    if bsp_file:
        bsp_path = Path(bsp_file)
        if not bsp_path.exists():
            console.print(f"[red]✗ BSP file not found: {bsp_file}[/red]")
            raise typer.Exit(1)
        bsp = bsp_path.read_text(encoding="utf-8")
    else:
        config_path = _ensure_initialized(cwd)
        config = load_config(config_path)
        bsp = load_bsp(config, cwd)
        if not bsp:
            console.print("[red]✗ No BSP configured. Add 'bsp' section to promptlab.yaml[/red]")
            raise typer.Exit(1)

    linter = BSPLinter()
    result = linter.lint(bsp)
    report = linter.format_report(result)

    if result.ready_for_evaluation:
        console.print(Panel(report, title="BSP Lint — PASSED ✓", border_style="green"))
    else:
        console.print(Panel(report, title="BSP Lint — FIX REQUIRED ✗", border_style="red"))
        raise typer.Exit(1)


@app.command("history")
def show_history(
    last_n: int = typer.Option(10, "--last", "-n", help="Number of recent entries to show"),
    compare: bool = typer.Option(False, "--compare", "-c", help="Compare BSP versions"),
):
    """Show evaluation history and score trends — completely FREE.
    
    Examples:
      promptlab history               # Show recent evaluation history
      promptlab history --last 20     # Show last 20 entries
      promptlab history --compare     # Compare scores across BSP versions
    """
    from promptlab.utils.history import EvaluationHistory

    cwd = Path.cwd()
    tracker = EvaluationHistory(cwd)

    if not tracker.get_history():
        console.print("[yellow]No evaluation history found. Run 'promptlab validate' first.[/yellow]")
        raise typer.Exit(0)

    if compare:
        comparison = tracker.compare_bsp_versions()
        console.print(Panel.fit("[bold]BSP Version Comparison[/bold]", border_style="blue"))
        for version, data in comparison.items():
            console.print(f"  {version}: avg={data['average_score']:.2f}, best={data['best_score']:.2f}, runs={data['evaluations']}")
    else:
        report = tracker.format_report()
        trend = tracker.get_trend(last_n)
        border = "green" if trend.trend_direction == "improving" else "red" if trend.trend_direction == "declining" else "blue"
        console.print(Panel(report, title="Evaluation History", border_style=border))


@app.command("guardrail")
def run_guardrail(
    bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
    categories: list[str] = typer.Option(None, "--category", "-c", help="Attack categories to test"),
    evaluator: str = typer.Option(None, "--evaluator", "-e", help="Model for ambiguous evaluations"),
):
    """Run adversarial security tests against your BSP.
    
    Tests prompt injection, jailbreaks, system prompt extraction,
    role breaking, and data exfiltration attacks.
    
    Examples:
      promptlab guardrail                                    # Run all attack categories
      promptlab guardrail -c prompt_injection -c jailbreak   # Test specific categories
      promptlab guardrail -e openrouter/google/gemini-2.0-flash-001  # Use evaluator for ambiguous results
    """
    import asyncio as aio
    from promptlab.utils.config import load_config, load_bsp
    from promptlab.llm_council.llm_runner.runner import LLMRunner
    from promptlab.utils.guardrail import GuardrailTester

    cwd = Path.cwd()
    config_path = _ensure_initialized(cwd)

    config = load_config(config_path)

    if bsp_file:
        bsp = Path(bsp_file).read_text(encoding="utf-8")
    else:
        bsp = load_bsp(config, cwd)
        if not bsp:
            console.print("[red]✗ No BSP configured.[/red]")
            raise typer.Exit(1)

    async def _run():
        runner = LLMRunner(config.models.dict())
        tester = GuardrailTester(runner, {"rate_limit_delay": config.guardrail.rate_limit_delay})

        test_categories = list(categories) if categories else config.guardrail.categories
        eval_model = evaluator or config.guardrail.evaluator_model

        console.print(Panel(
            f"[bold]Model:[/bold] {config.models.default}\n"
            f"[bold]Categories:[/bold] {', '.join(test_categories)}\n"
            f"[bold]Evaluator:[/bold] {eval_model or 'pattern-matching only (free)'}",
            title="Guardrail Test",
            border_style="blue",
        ))

        report = await tester.run_guardrail_tests(
            bsp=bsp,
            model=config.models.default,
            categories=test_categories,
            evaluator_model=eval_model,
        )

        # Display results
        console.print(f"\n[bold]Safety Score:[/bold] {report.overall_safety_score:.2f}/1.00")
        console.print(f"[bold]Total Attacks:[/bold] {report.total_attacks}")
        console.print(f"[bold]Vulnerabilities:[/bold] {report.vulnerabilities_found}")

        if report.critical_count:
            console.print(f"[bold red]  Critical: {report.critical_count}[/bold red]")
        if report.high_count:
            console.print(f"[red]  High: {report.high_count}[/red]")
        if report.medium_count:
            console.print(f"[yellow]  Medium: {report.medium_count}[/yellow]")

        if report.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in report.recommendations:
                console.print(f"  → {rec}")

        if report.vulnerabilities_found == 0:
            console.print(Panel("[bold green]✓ No vulnerabilities found![/bold green]", border_style="green"))
        else:
            console.print(Panel(f"[bold red]✗ {report.vulnerabilities_found} vulnerabilities found[/bold red]", border_style="red"))

        return report

    try:
        report = aio.run(_run())
        if report.critical_count > 0:
            raise typer.Exit(1)
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("optimize-bsp")
def optimize_bsp(
    bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
    model: str = typer.Option(None, "--model", "-m", help="Model for optimization"),
    max_iterations: int = typer.Option(3, "--iterations", "-i", help="Max optimization iterations"),
    target_score: float = typer.Option(0.90, "--target", "-t", help="Target lint score"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save optimized BSP to file"),
):
    """Optimize your BSP iteratively using AI + linting feedback.
    
    Cost: 1-3 API calls total (one per iteration). Pre-validates with
    the free BSP linter before and after each optimization.
    
    Examples:
      promptlab optimize-bsp                       # Optimize BSP from config
      promptlab optimize-bsp -b bsp.txt -i 5       # 5 iterations on specific file
      promptlab optimize-bsp --target 0.95         # Aim for 0.95 lint score
    """
    import asyncio as aio
    from promptlab.utils.config import load_config, load_bsp
    from promptlab.llm_council.llm_runner.runner import LLMRunner
    from promptlab.utils.bsp_optimizer import BSPOptimizer

    cwd = Path.cwd()
    config_path = _ensure_initialized(cwd)

    config = load_config(config_path)

    if bsp_file:
        bsp_path = Path(bsp_file)
        bsp = bsp_path.read_text(encoding="utf-8")
    else:
        bsp = load_bsp(config, cwd)
        bsp_path = None
        if not bsp:
            console.print("[red]✗ No BSP configured.[/red]")
            raise typer.Exit(1)

    opt_model = model or config.optimizer.optimizer_model or config.models.default

    async def _run():
        runner = LLMRunner(config.models.dict())
        optimizer = BSPOptimizer(
            llm_runner=runner,
            optimizer_model=opt_model,
            max_iterations=max_iterations,
            target_lint_score=target_score,
        )

        console.print(Panel(
            f"[bold]Model:[/bold] {opt_model}\n"
            f"[bold]Max Iterations:[/bold] {max_iterations}\n"
            f"[bold]Target Score:[/bold] {target_score}",
            title="BSP Optimizer",
            border_style="blue",
        ))

        result = await optimizer.optimize(bsp)
        report = optimizer.format_report(result)
        console.print(Panel(report, title="Optimization Result", border_style="green" if result.improvement > 0 else "yellow"))

        return result

    try:
        result = aio.run(_run())

        if save and result.improvement > 0:
            output_path = bsp_path or (cwd / "bsp_optimized.txt")
            if bsp_path:
                # Backup original
                backup = bsp_path.with_suffix(".bak")
                backup.write_text(bsp, encoding="utf-8")
                console.print(f"[dim]Original backed up to {backup}[/dim]")
            output_path.write_text(result.optimized_bsp, encoding="utf-8")
            console.print(f"[green]✓ Optimized BSP saved to {output_path}[/green]")
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("evaluate-conversation")
def evaluate_conversation(
    bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
    conversation_file: str = typer.Option(None, "--conversation", "-c", help="Path to conversation JSON file"),
    model: str = typer.Option(None, "--model", "-m", help="Model to generate test conversation"),
    evaluator: str = typer.Option(None, "--evaluator", "-e", help="Model for evaluation"),
    num_turns: int = typer.Option(6, "--turns", "-n", help="Number of conversation turns to generate"),
):
    """Evaluate multi-turn conversation quality.
    
    Either provide a conversation file or auto-generate a test conversation.
    Cost: 1 API call for evaluation + N/2 calls if generating conversation.
    
    Examples:
      promptlab evaluate-conversation                              # Generate & evaluate conversation
      promptlab evaluate-conversation -c conversation.json         # Evaluate existing conversation
      promptlab evaluate-conversation --turns 10 -m ollama/llama3  # Generate 10-turn conversation
    """
    import asyncio as aio
    import json as json_module
    from promptlab.utils.config import load_config, load_bsp
    from promptlab.llm_council.llm_runner.runner import LLMRunner
    from promptlab.utils.multi_turn import MultiTurnEvaluator, ConversationTurn

    cwd = Path.cwd()
    config_path = _ensure_initialized(cwd)

    config = load_config(config_path)

    if bsp_file:
        bsp = Path(bsp_file).read_text(encoding="utf-8")
    else:
        bsp = load_bsp(config, cwd)
        if not bsp:
            console.print("[red]✗ No BSP configured.[/red]")
            raise typer.Exit(1)

    target_model = model or config.models.default
    eval_model = evaluator or config.multi_turn.evaluator_model or (config.council.members[0] if config.council.members else target_model)

    async def _run():
        runner = LLMRunner(config.models.dict())
        evaluator_instance = MultiTurnEvaluator(runner, {"rate_limit_delay": config.multi_turn.rate_limit_delay})

        if conversation_file:
            # Load existing conversation
            conv_data = json_module.loads(Path(conversation_file).read_text(encoding="utf-8"))
            turns = [
                ConversationTurn(role=t["role"], content=t["content"], turn_number=i + 1)
                for i, t in enumerate(conv_data)
            ]
            console.print(f"[green]✓ Loaded {len(turns)} turns from {conversation_file}[/green]")
        else:
            # Generate test conversation
            console.print(f"[blue]Generating {num_turns}-turn test conversation with {target_model}...[/blue]")
            turns = await evaluator_instance.generate_test_conversation(
                bsp=bsp,
                model=target_model,
                num_turns=num_turns,
            )
            console.print(f"[green]✓ Generated {len(turns)} turns[/green]")

        # Show conversation
        console.print("\n[bold]Conversation:[/bold]")
        for turn in turns:
            role_color = "cyan" if turn.role == "user" else "green"
            console.print(f"  [{role_color}][{turn.role.upper()}][/{role_color}] {turn.content[:150]}...")

        # Evaluate
        console.print(f"\n[blue]Evaluating with {eval_model}...[/blue]")
        result = await evaluator_instance.evaluate_conversation(turns, bsp, eval_model)

        # Display results
        console.print(Panel(
            f"[bold]Overall Score:[/bold] {result.overall_score:.2f}\n"
            f"[bold]Context Retention:[/bold] {result.context_retention_score:.2f}\n"
            f"[bold]Role Consistency:[/bold] {result.role_consistency_score:.2f}\n"
            f"[bold]Coherence:[/bold] {result.coherence_score:.2f}\n"
            f"[bold]Personality Drift:[/bold] {result.personality_drift_score:.2f} (1.0 = no drift)\n"
            f"\n{result.summary}",
            title="Conversation Evaluation",
            border_style="green" if result.overall_score >= 0.7 else "red",
        ))

        if result.drift_detected_at:
            console.print(f"[yellow]  ⚠ Drift detected at turns: {result.drift_detected_at}[/yellow]")

        if result.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in result.recommendations:
                console.print(f"  → {rec}")

        console.print(f"\n[dim]API calls used: {result.api_calls_used}[/dim]")

    try:
        aio.run(_run())
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("generate-training-data")
def generate_training_data(
    bsp_file: str = typer.Option(None, "--bsp-file", "-b", help="Path to BSP file"),
    outputs_dir: str = typer.Option("temp", "--outputs-dir", "-d", help="Directory with test outputs"),
    min_score: float = typer.Option(0.7, "--min-score", "-s", help="Minimum quality score filter"),
    format: str = typer.Option("openai", "--format", "-f", help="Output format: openai, alpaca, sharegpt"),
    preview: bool = typer.Option(False, "--preview", "-p", help="Preview without saving"),
):
    """Generate fine-tuning training data from BSP + evaluation outputs — completely FREE.
    
    Converts your BSP and high-scoring test outputs into JSONL format
    suitable for fine-tuning models. No API calls needed.
    
    Examples:
      promptlab generate-training-data                            # Generate from default outputs
      promptlab generate-training-data --format alpaca            # Alpaca format
      promptlab generate-training-data --min-score 0.8 --preview  # Preview high-quality examples only
    """
    import json as json_module
    from promptlab.utils.config import load_config, load_bsp
    from promptlab.utils.training_data import TrainingDataGenerator

    cwd = Path.cwd()
    config_path = _ensure_initialized(cwd)

    config = load_config(config_path)

    if bsp_file:
        bsp = Path(bsp_file).read_text(encoding="utf-8")
    else:
        bsp = load_bsp(config, cwd)
        if not bsp:
            console.print("[red]✗ No BSP configured.[/red]")
            raise typer.Exit(1)

    # Load outputs from evaluation history
    outputs_path = cwd / outputs_dir
    outputs = []

    if outputs_path.is_dir():
        for f in sorted(outputs_path.glob("*.yaml")) + sorted(outputs_path.glob("*.json")):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "cases" in data:
                    for case in data["cases"]:
                        outputs.append({
                            "prompt": case.get("prompt", ""),
                            "response": case.get("response", case.get("expected", "")),
                            "test_id": case.get("id", f.stem),
                            "score": min_score,
                        })
            except Exception:
                continue

    if not outputs:
        console.print("[yellow]No test outputs found. Run 'promptlab validate' first to generate outputs.[/yellow]")
        raise typer.Exit(0)

    generator = TrainingDataGenerator(
        project_root=cwd,
        min_quality_score=min_score,
        output_format=format,
    )

    dataset = generator.generate_from_outputs(
        bsp=bsp,
        outputs=outputs,
        overall_score=min_score,
        bsp_version=config.bsp.version,
    )

    report = generator.format_report(dataset)
    console.print(Panel(report, title="Training Data", border_style="blue"))

    if preview:
        console.print("\n[bold]Preview (first 3 examples):[/bold]")
        for ex in dataset.examples[:3]:
            console.print(f"  [cyan]User:[/cyan] {ex.user[:100]}")
            console.print(f"  [green]Assistant:[/green] {ex.assistant[:100]}")
            console.print()
    else:
        output_file = generator.export_jsonl(dataset)
        preview_file = generator.export_json(dataset)
        console.print(f"[green]✓ Training data saved to {output_file}[/green]")
        console.print(f"[green]✓ Preview saved to {preview_file}[/green]")
        console.print(f"\n[dim]Total examples: {dataset.total_examples} | Format: {format} | Estimated tokens: {dataset.total_tokens_estimate:,}[/dim]")


if __name__ == "__main__":
    app()

