"""Shared CLI helpers — used across all command modules."""

import sys
import typer
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

from promptlab.cli.templates import (
    DEFAULT_CONFIG, PROVIDER_BLOCKS, DEFAULT_MODELS,
    COUNCIL_CONFIGS, EXAMPLE_TEST, BSP_TEMPLATE, GITIGNORE_TEMPLATE,
)

console = Console()


def is_interactive() -> bool:
    """Check if we're running in an interactive terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def ensure_initialized(cwd: Path = None) -> Path:
    """Check if promptlab.yaml exists. If not, offer to run init.

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
        console.print()
        run_interactive_init(cwd)
        return cwd / "promptlab.yaml"
    else:
        console.print("[dim]Run 'promptlab init' to set up your project.[/dim]")
        raise typer.Exit(1)


def run_interactive_init(cwd: Path):
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
    console.print("  5. [bold]NVIDIA[/bold] — NVIDIA NIM models")
    console.print("  6. [bold]Other[/bold] (anthropic, xai)")
    console.print()

    choice = typer.prompt("Choose provider (1-6)", default="1")

    provider_map = {"1": "google", "2": "openrouter", "3": "ollama", "4": "openai", "5": "nvidia", "6": "ollama"}
    primary = provider_map.get(choice, "google")

    if choice == "6":
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
            "nvidia": "https://build.nvidia.com/",
        }
        if primary in help_urls:
            console.print(f"\n[dim]Get your key at: {help_urls[primary]}[/dim]")
        api_key = typer.prompt(f"API key for {primary}")
        if not api_key:
            console.print("[yellow]No key provided, falling back to ollama[/yellow]")
            primary = "ollama"

    # Build config
    block = PROVIDER_BLOCKS.get(primary, "")
    env_var_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai": "XAI_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }

    # Create .env
    env_path = cwd / ".env"
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text().splitlines()
    if api_key and primary in env_var_map:
        var_name = env_var_map[primary]
        found = False
        for i, line in enumerate(env_lines):
            if line.startswith(f"{var_name}="):
                env_lines[i] = f"{var_name}={api_key}"
                found = True
                break
        if not found:
            env_lines.append(f"{var_name}={api_key}")
        env_path.write_text("\n".join(env_lines) + "\n")
        console.print(f"[green]✓[/green] Created .env (API keys stored securely)")

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

    # Create .gitignore
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


def init_ft_files(cwd: Path, force: bool = False):
    """Create fine-tuning specific init files."""
    from promptlab.cli.templates import FT_CONFIG_TEMPLATE

    ft_config_path = cwd / "ft_config.json"
    if not ft_config_path.exists() or force:
        ft_config_path.write_text(FT_CONFIG_TEMPLATE)
        console.print(f"[green]✓[/green] Created ft_config.json")

    gov_dir = cwd / ".promptlab" / "governance"
    gov_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Created .promptlab/governance/ directory")
