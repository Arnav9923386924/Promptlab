"""Interactive model and role selector using InquirerPy.

Presents a terminal UI for users to:
1. Discover all available models from every configured provider
2. Select N models via multi-select checkboxes
3. Assign a role (Judge / Chairman) to each selected model

Falls back gracefully to config values when:
- Running in non-interactive mode (CI, piped stdin)
- User cancels / presses Ctrl+C
- InquirerPy is not installed
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()

# ── Predefined roles for council models ──
AVAILABLE_ROLES = ["Judge", "Chairman"]


def _is_interactive() -> bool:
    """Check if we're running in an interactive terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


# ─────────────────────────────────────────────────────────────────────────────
# Provider-specific model listing helpers (async)
# ─────────────────────────────────────────────────────────────────────────────

async def _list_openrouter_models(api_key: str) -> list[dict]:
    """Fetch available models from OpenRouter (includes free :free models)."""
    import httpx
    models: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/promptlab",
                "X-Title": "PromptLab",
            }
            resp = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                mid = entry.get("id", "")
                if not mid:
                    continue
                # Only include :free models (truly free on OpenRouter)
                if not mid.endswith(":free"):
                    continue
                ctx = entry.get("context_length", 0) or 0
                if ctx < 4000:
                    continue
                models.append({
                    "id": f"openrouter/{mid}",
                    "name": entry.get("name", mid)[:50],
                    "provider": "OpenRouter",
                    "context": ctx,
                })
    except Exception as e:
        console.print(f"[yellow]  Warning: OpenRouter model fetch failed: {str(e)[:80]}[/yellow]")
    return models


async def _list_google_models(api_key: str) -> list[dict]:
    """Fetch text-generation models from Google AI Studio (all free tier)."""
    import httpx
    models: list[dict] = []
    skip_kw = ["imagen", "veo", "embedding", "aqa", "tts", "audio",
               "predict", "robotics", "computer-use", "deep-research", "image"]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            for entry in resp.json().get("models", []):
                name = entry.get("name", "")
                display = entry.get("displayName", "")
                methods = entry.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue
                if any(kw in name.lower() for kw in skip_kw):
                    continue
                ctx = entry.get("inputTokenLimit", 0) or 0
                if ctx < 4000:
                    continue
                model_id = name.replace("models/", "")
                models.append({
                    "id": f"google/{model_id}",
                    "name": display or model_id,
                    "provider": "Google",
                    "context": ctx,
                })
    except Exception as e:
        console.print(f"[yellow]  Warning: Google model fetch failed: {str(e)[:80]}[/yellow]")
    return models


def _list_static_provider_models(provider_name: str) -> list[dict]:
    """Return well-known models for providers that don't have a list API.

    These are common models for OpenAI, Anthropic, xAI, Ollama.
    The user can always type a custom model ID via the free-text option.
    """
    static = {
        "openai": [
            ("openai/gpt-4o", "GPT-4o"),
            ("openai/gpt-4o-mini", "GPT-4o Mini"),
            ("openai/gpt-4-turbo", "GPT-4 Turbo"),
            ("openai/gpt-3.5-turbo", "GPT-3.5 Turbo"),
            ("openai/o1", "o1"),
            ("openai/o1-mini", "o1 Mini"),
            ("openai/o3-mini", "o3 Mini"),
        ],
        "anthropic": [
            ("anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4"),
            ("anthropic/claude-3.5-sonnet-20241022", "Claude 3.5 Sonnet"),
            ("anthropic/claude-3-haiku-20240307", "Claude 3 Haiku"),
            ("anthropic/claude-3-opus-20240229", "Claude 3 Opus"),
        ],
        "xai": [
            ("xai/grok-2", "Grok 2"),
            ("xai/grok-2-mini", "Grok 2 Mini"),
            ("xai/grok-3", "Grok 3"),
        ],
    }
    provider_lower = provider_name.lower()
    entries = static.get(provider_lower, [])
    return [
        {"id": mid, "name": name, "provider": provider_name.title(), "context": 0}
        for mid, name in entries
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main discovery (async)
# ─────────────────────────────────────────────────────────────────────────────

async def discover_models(config) -> list[dict]:
    """Discover all available models from every configured provider.

    Call inside an event loop (via new_event_loop + run_until_complete).
    Returns a flat list of model dicts sorted by provider.
    """
    providers = config.models.providers or {}

    console.print("\n[bold cyan]Model Selection[/bold cyan]")
    console.print("[dim]Discovering available models...[/dim]")

    all_models: list[dict] = []
    tasks = []

    # Dynamic fetch for providers with list APIs
    or_cfg = providers.get("openrouter")
    if or_cfg and or_cfg.api_key and not or_cfg.api_key.startswith("sk-xxx"):
        tasks.append(_list_openrouter_models(or_cfg.api_key))

    google_cfg = providers.get("google")
    if google_cfg and google_cfg.api_key and len(google_cfg.api_key) > 10:
        tasks.append(_list_google_models(google_cfg.api_key))

    # Fetch dynamic providers in parallel
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_models.extend(r)

    # Static models for providers with API keys configured
    for pname in ("openai", "anthropic", "xai"):
        pcfg = providers.get(pname)
        if pcfg and pcfg.api_key and not pcfg.api_key.endswith("-xxx"):
            all_models.extend(_list_static_provider_models(pname))

    # Also add any config-defined members/chairman that aren't already present
    seen_ids = {m["id"] for m in all_models}
    config_models = list(config.council.members or [])
    if config.council.chairman and config.council.chairman not in seen_ids:
        config_models.append(config.council.chairman)
    for mid in config_models:
        if mid not in seen_ids:
            short = mid.split("/")[-1][:40]
            prov = mid.split("/")[0].title() if "/" in mid else "Config"
            all_models.append({
                "id": mid,
                "name": short,
                "provider": prov,
                "context": 0,
            })
            seen_ids.add(mid)

    # Summary
    providers_found = {}
    for m in all_models:
        providers_found[m["provider"]] = providers_found.get(m["provider"], 0) + 1
    parts = [f"{p}: {c}" for p, c in sorted(providers_found.items())]
    total = len(all_models)

    if total:
        console.print(f"[green]  Found {total} available models ({', '.join(parts)})[/green]")
    else:
        console.print("[yellow]  No models discovered - using config defaults[/yellow]")

    return all_models


# ─────────────────────────────────────────────────────────────────────────────
# Interactive selection (sync — must run OUTSIDE any event loop)
# ─────────────────────────────────────────────────────────────────────────────

def run_model_selection(
    config,
    available: list[dict],
) -> tuple[list[str], Optional[str]]:
    """Present InquirerPy UI for model + role selection.

    Flow:
      1. Multi-select checkbox: pick N models from the discovered list
      2. For each selected model, pick a role (Judge / Chairman)
      3. If no Chairman was assigned, prompt for one
      4. Show a summary table

    MUST be called **outside** any running asyncio event loop.

    Returns:
        (judges, chairman) — lists of model ID strings
    """
    config_members = config.council.members
    config_chairman = config.council.chairman
    required_judges = config.council.required_judges

    if not available:
        return config_members, config_chairman

    try:
        from InquirerPy import inquirer
        from InquirerPy.separator import Separator
    except ImportError:
        console.print("[yellow]InquirerPy not installed - using config defaults[/yellow]")
        return config_members, config_chairman

    if not _is_interactive():
        console.print("[dim]Non-interactive mode - using config defaults[/dim]")
        return config_members, config_chairman

    # ── Group models by provider for the checkbox ──
    by_provider: dict[str, list[dict]] = {}
    for m in available:
        by_provider.setdefault(m["provider"], []).append(m)

    # Pre-select models already in config
    pre_selected = set(config_members or [])
    if config_chairman:
        pre_selected.add(config_chairman)

    choices = []
    for provider, models in sorted(by_provider.items()):
        choices.append(Separator(f"── {provider} ({len(models)} models) ──"))
        for m in models:
            choices.append({
                "name": f"{m['name']}  [{m['id']}]",
                "value": m["id"],
                "enabled": m["id"] in pre_selected,
            })

    # ── Step 1 — Select models (multi-select) ──
    console.print()
    console.print("[bold yellow]Use arrow keys to navigate, SPACE to toggle selection (green = selected), ENTER when done[/bold yellow]")
    console.print(f"[dim]You need at least {required_judges} judges + 1 chairman (total: {required_judges + 1} models)[/dim]\n")
    
    try:
        selected_ids = inquirer.checkbox(
            message="Select models for your council:",
            choices=choices,
            cycle=True,
            instruction="↑↓ navigate  |  SPACE toggle  |  ENTER confirm",
        ).execute()
    except (KeyboardInterrupt, EOFError):
        console.print("[dim]Selection cancelled - using config defaults[/dim]")
        return config_members, config_chairman

    # Debug: show what was selected
    console.print(f"\n[dim]Selected {len(selected_ids)} models: {', '.join([m.split('/')[-1] for m in selected_ids])}[/dim]")

    if not selected_ids or len(selected_ids) < required_judges + 1:
        console.print(
            f"[yellow]Need at least {required_judges} judges + 1 chairman "
            f"({required_judges + 1} models). Using config defaults.[/yellow]"
        )
        return config_members, config_chairman

    # Build lookup for display names
    id_to_name = {m["id"]: m["name"] for m in available}

    # ── Step 2 — Assign roles to each selected model ──
    judges: list[str] = []
    chairman: Optional[str] = None

    console.print()
    console.print("[bold]Assign a role to each selected model:[/bold]")

    for mid in selected_ids:
        display = id_to_name.get(mid, mid.split("/")[-1])
        # Default suggestion: first model → Chairman, rest → Judge
        default_role = "Chairman" if chairman is None and mid == selected_ids[-1] else "Judge"
        if mid == config_chairman:
            default_role = "Chairman"

        try:
            role = inquirer.select(
                message=f"  {display}:",
                choices=AVAILABLE_ROLES,
                default=default_role,
                cycle=True,
            ).execute()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Role assignment cancelled - using config defaults[/dim]")
            return config_members, config_chairman

        if role == "Chairman":
            if chairman is not None:
                # Already have a chairman → demote the previous one to judge
                console.print(f"[dim]    (Previous chairman demoted to judge)[/dim]")
                judges.append(chairman)
            chairman = mid
        else:
            judges.append(mid)

    # ── Step 3 — Ensure we have a chairman ──
    if chairman is None and judges:
        console.print("\n[yellow]No chairman assigned. Please select one:[/yellow]")
        try:
            chairman_pick = inquirer.select(
                message="Select chairman from your judges:",
                choices=[
                    {"name": id_to_name.get(j, j.split("/")[-1]), "value": j}
                    for j in judges
                ],
                cycle=True,
            ).execute()
            judges.remove(chairman_pick)
            chairman = chairman_pick
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Chairman selection cancelled - using first model[/dim]")
            chairman = judges.pop(0)

    # ── Validate minimum judges ──
    if len(judges) < required_judges:
        console.print(
            f"[yellow]Need at least {required_judges} judges. "
            f"Got {len(judges)}. Using config defaults.[/yellow]"
        )
        return config_members, config_chairman

    # ── Summary table ──
    console.print()
    table = Table(title="Council Configuration", border_style="cyan", show_lines=True)
    table.add_column("Model", style="bold white", min_width=30)
    table.add_column("Role", style="bold", min_width=12, justify="center")

    for j in judges:
        name = id_to_name.get(j, j.split("/")[-1])
        table.add_row(name, "[cyan]Judge[/cyan]")
    if chairman:
        name = id_to_name.get(chairman, chairman.split("/")[-1])
        table.add_row(name, "[yellow]Chairman[/yellow]")

    console.print(table)
    console.print()

    return judges, chairman
