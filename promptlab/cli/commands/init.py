"""CLI commands — project initialization and CI/CD setup."""

import typer
from rich.panel import Panel
from pathlib import Path

from promptlab.cli.helpers import (
    console, is_interactive, init_ft_files,
)
from promptlab.cli.templates import (
    DEFAULT_CONFIG, FT_DEFAULT_CONFIG, BOTH_DEFAULT_CONFIG,
    PROVIDER_BLOCKS, DEFAULT_MODELS,
    COUNCIL_CONFIGS, EXAMPLE_TEST, BSP_TEMPLATE,
    GITIGNORE_TEMPLATE, GITHUB_WORKFLOW,
)


def register(app: typer.Typer):
    """Register init-related commands on the root app."""

    @app.command()
    def init(
        force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
        provider: str = typer.Option(None, "--provider", "-p", help="Provider: ollama, openrouter, google, openai, anthropic, xai, nvidia"),
        api_key: str = typer.Option(None, "--api-key", "-k", help="API key for the chosen provider"),
        non_interactive: bool = typer.Option(False, "--yes", "-y", help="Non-interactive mode with defaults"),
        mode: str = typer.Option(None, "--mode", "-m", help="Init mode: 'bsp' (default), 'ft' (fine-tuning), or 'both'"),
    ):
        """Initialize PromptLab in the current directory.

        Modes:
          bsp   — BSP validation files (bsp.txt, tests, baselines)
          ft    — Fine-tuning governance files (ft_config.json)
          both  — Both BSP and fine-tuning files

        If --mode is not specified, you will be prompted interactively.

        Examples:
          promptlab init                                    # Interactive setup
          promptlab init --mode bsp                         # BSP only
          promptlab init --mode ft -p google -k AIza...     # Fine-tuning with Google
          promptlab init --mode both                        # Everything
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
        # Resolve mode
        init_mode = mode
        if not init_mode and not non_interactive and is_interactive():
            console.print("\n[bold]What would you like to set up?[/bold]\n")
            console.print("  1. [bold green]BSP[/bold green] — Behavior Specification Prompt validation")
            console.print("  2. [bold cyan]FT[/bold cyan]  — Fine-tuning hyperparameter governance")
            console.print("  3. [bold]Both[/bold] — BSP + Fine-tuning")
            console.print()
            mode_choice = typer.prompt("Choose mode (1-3)", default="1")
            init_mode = {"1": "bsp", "2": "ft", "3": "both"}.get(mode_choice, "bsp")
        elif not init_mode:
            init_mode = "bsp"

        include_bsp = init_mode in ("bsp", "both")
        include_ft = init_mode in ("ft", "both")

        # Collect provider info
        providers_selected = {}  # {provider_name: api_key_or_None}

        if provider:
            # Quick mode — single provider from CLI args
            if provider in ("openrouter", "google", "openai", "anthropic", "xai", "nvidia"):
                if not api_key:
                    console.print(f"[red]✗ {provider} requires --api-key[/red]")
                    raise typer.Exit(1)
                providers_selected[provider] = api_key
            elif provider == "ollama":
                providers_selected["ollama"] = None
            else:
                console.print(f"[red]✗ Unknown provider: {provider}[/red]")
                console.print("[dim]Supported: ollama, openrouter, google, openai, anthropic, xai, nvidia[/dim]")
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
                ("nvidia", "NVIDIA — NIM hosted models", "Get key: https://build.nvidia.com/"),
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

        # Map provider → env var name
        env_var_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "xai": "XAI_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
        }

        # Create .env file with actual API keys
        env_path = cwd / ".env"
        env_lines = []
        if env_path.exists():
            env_lines = env_path.read_text().splitlines()

        env_changed = False
        for prov, key in providers_selected.items():
            if key and prov in env_var_map:
                var_name = env_var_map[prov]
                found = False
                for i, line in enumerate(env_lines):
                    if line.startswith(f"{var_name}="):
                        env_lines[i] = f"{var_name}={key}"
                        found = True
                        break
                if not found:
                    env_lines.append(f"{var_name}={key}")
                env_changed = True

        if env_changed:
            env_path.write_text("\n".join(env_lines) + "\n")
            console.print(f"[green]✓[/green] Created .env (API keys stored securely)")

        # Build providers block — uses ${VAR} refs (no literal keys)
        providers_lines = []
        for prov in providers_selected:
            block = PROVIDER_BLOCKS.get(prov, "")
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

        # Select config template based on mode
        if init_mode == "ft":
            config_template = FT_DEFAULT_CONFIG
        elif init_mode == "both":
            config_template = BOTH_DEFAULT_CONFIG
        else:
            config_template = DEFAULT_CONFIG

        # Build config
        config_content = config_template.format(
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

        # Create BSP-specific files only if mode includes BSP
        if include_bsp:
            # Create bsp.txt
            bsp_path = cwd / "bsp.txt"
            if not bsp_path.exists() or force:
                bsp_path.write_text(BSP_TEMPLATE)
                console.print(f"[green]✓[/green] Created bsp.txt")

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

        # Create root .gitignore
        root_gitignore = cwd / ".gitignore"
        if not root_gitignore.exists():
            root_gitignore.write_text(GITIGNORE_TEMPLATE)
            console.print(f"[green]✓[/green] Created .gitignore")

        # Create FT-specific files if mode includes FT
        if include_ft:
            init_ft_files(cwd, force)

        # Summary
        providers_summary = ", ".join(providers_selected.keys())
        mode_label = init_mode.upper()

        # Build next steps based on mode
        next_steps = []
        if include_bsp:
            next_steps.extend([
                "  1. Edit [cyan]bsp.txt[/cyan] to define your AI's behavior",
                "  2. Run [cyan]promptlab bsp lint[/cyan] to check BSP quality",
                "  3. Run [cyan]promptlab bsp[/cyan] to validate your prompts",
            ])
        if include_ft:
            start = len(next_steps) + 1
            next_steps.extend([
                f"  {start}. Edit [cyan]ft_config.json[/cyan] to configure your search space",
                f"  {start+1}. Run [cyan]promptlab ft --propose[/cyan] to get first config",
            ])

        created_files = [
            "  • [cyan]promptlab.yaml[/cyan] — main configuration",
            "  • [cyan].env[/cyan] — API keys (never commit this!)",
        ]
        if include_bsp:
            created_files.extend([
                "  • [cyan]bsp.txt[/cyan] — your behavior specification",
                "  • [cyan]temp/example.yaml[/cyan] — example test suite",
            ])
        if include_ft:
            created_files.append("  • [cyan]ft_config.json[/cyan] — fine-tuning governance config")
        created_files.append("  • [cyan].promptlab/[/cyan] — storage")

        console.print()
        console.print(Panel(
            f"[bold green]PromptLab initialized! ({mode_label})[/bold green]\n\n"
            f"[bold]Providers:[/bold] {providers_summary}\n"
            f"[bold]Default model:[/bold] {DEFAULT_MODELS.get(primary_provider, 'ollama/llama3.1:8b')}\n"
            f"[bold]Council:[/bold] {council_provider} ({len(council_cfg['members'])} judges)\n\n"
            "Created files:\n" + "\n".join(created_files) + "\n\n"
            "Next steps:\n" + "\n".join(next_steps),
            title="PromptLab",
            border_style="green",
        ))

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
        from promptlab.cli.helpers import ensure_initialized

        cwd = Path.cwd()
        workflows_dir = cwd / ".github" / "workflows"
        workflow_path = workflows_dir / "prompt-tests.yml"

        # Ensure project is initialized
        ensure_initialized(cwd)

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
