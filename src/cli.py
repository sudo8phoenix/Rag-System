"""Typer-based command line interface for the RAG system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import ValidationError

if not hasattr(click.Choice, "__class_getitem__"):
    click.Choice.__class_getitem__ = classmethod(lambda cls, params: cls)

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

from src.config.loader import ConfigLoadError, ConfigValidationError, load_config
from src.config.settings import AppConfig
from src.embeddings.orchestrator import EmbeddingOrchestrator
from src.parsers.registry import ParserRegistry
from src.pipeline import PipelineOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

console = Console()
app = typer.Typer(
    add_completion=False, help="Command line interface for the RAG system."
)
config_app = typer.Typer(
    add_completion=False, help="Inspect and update YAML configuration."
)
app.add_typer(config_app, name="config")


def _load_runtime_config(config_path: Path) -> AppConfig:
    try:
        return load_config(config_path)
    except (ConfigLoadError, ConfigValidationError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


def _get_context_config_path(ctx: typer.Context) -> Path:
    state = ctx.obj if isinstance(ctx.obj, dict) else {}
    config_path = state.get("config_path", DEFAULT_CONFIG_PATH)
    return Path(config_path)


def _parse_cli_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _set_nested_value(payload: dict[str, Any], key_path: str, value: Any) -> None:
    parts = [part.strip() for part in key_path.split(".") if part.strip()]
    if not parts:
        raise typer.BadParameter("setting path cannot be empty")

    current: dict[str, Any] = payload
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            raise typer.BadParameter(f"Unknown config path: {key_path}")
        current = next_value

    leaf = parts[-1]
    if leaf not in current:
        raise typer.BadParameter(f"Unknown config key: {key_path}")

    current[leaf] = value


def _validate_config_payload(
    payload: dict[str, Any],
) -> tuple[AppConfig | None, list[str]]:
    try:
        return AppConfig.model_validate(payload), []
    except ValidationError as exc:
        errors: list[str] = []
        for issue in exc.errors():
            location = ".".join(str(part) for part in issue.get("loc", []))
            errors.append(f"{location}: {issue.get('msg', 'invalid value')}")
        return None, errors


def _collect_supported_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path]

    if not source_path.is_dir():
        raise typer.BadParameter(f"Path does not exist: {source_path}")

    supported_extensions = set(ParserRegistry().supported_extensions)
    return [
        path
        for path in sorted(source_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]


def _print_ingest_summary(rows: list[dict[str, Any]], discovered_count: int) -> None:
    table = Table(title="Ingest Summary", show_lines=False)
    table.add_column("File", overflow="fold")
    table.add_column("Status")
    table.add_column("Chunks", justify="right")
    table.add_column("Errors", justify="right")

    for row in rows:
        status = "success" if not row["errors"] else "error"
        error_text = "; ".join(row["errors"]) if row["errors"] else ""
        table.add_row(row["path"], status, str(row["chunks"]), error_text)

    console.print(table)
    console.print(f"Discovered {discovered_count} supported file(s).")


@app.callback()
def cli_root(
    ctx: typer.Context,
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to config.yaml.",
    ),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@app.command()
def ingest(
    ctx: typer.Context,
    source_path: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=True, file_okay=True
    ),
) -> None:
    """Ingest a file or directory into the configured vector store."""

    config = _load_runtime_config(_get_context_config_path(ctx))
    orchestrator = PipelineOrchestrator.from_config(config)
    files = _collect_supported_files(source_path)

    if not files:
        console.print(f"[yellow]No supported files found under {source_path}.[/yellow]")
        raise typer.Exit(code=1)

    rows: list[dict[str, Any]] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    with progress:
        task_id = progress.add_task("Ingesting files", total=len(files))
        for file_path in files:
            progress.update(task_id, description=f"Ingesting {file_path.name}")
            result = orchestrator.ingest_documents([file_path])
            rows.append(
                {
                    "path": str(file_path),
                    "chunks": len(result.chunks),
                    "errors": [error["error"] for error in result.errors],
                }
            )
            progress.advance(task_id)

    _print_ingest_summary(rows, len(files))
    if any(row["errors"] for row in rows):
        raise typer.Exit(code=1)


@app.command()
def query(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Question to ask the RAG system."),
    audio: bool = typer.Option(False, "--audio/--no-audio", help="Play TTS output."),
) -> None:
    """Query the existing index and print the answer to stdout."""

    config = _load_runtime_config(_get_context_config_path(ctx))
    runtime_config = config.model_copy(deep=True)
    runtime_config.tts.mute = not audio

    orchestrator = PipelineOrchestrator.from_config(runtime_config)
    result = orchestrator.answer(question, ingest_sources=False, block=True)

    if not result.success:
        console.print(
            f"[red]Query failed at {result.error_stage}: {result.error_message}[/red]"
        )
        raise typer.Exit(code=1)

    console.print(result.response_text)
    if result.audio_path and audio:
        console.print(f"[dim]Audio saved to {result.audio_path}[/dim]")


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Display the current configuration as YAML."""

    config = _load_runtime_config(_get_context_config_path(ctx))
    payload = yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False)
    console.print(Syntax(payload, "yaml", theme="ansi_dark", word_wrap=True))


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    setting_path: str = typer.Argument(
        ..., help="Dot-separated config path, for example llm.model."
    ),
    value: str = typer.Argument(..., help="New value to store."),
) -> None:
    """Update one configuration setting and write it back to disk."""

    config_path = _get_context_config_path(ctx)
    config = _load_runtime_config(config_path)
    payload = config.model_dump(mode="python")

    try:
        _set_nested_value(payload, setting_path, _parse_cli_value(value))
    except typer.BadParameter as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    validated_config, errors = _validate_config_payload(payload)
    if errors:
        console.print("[red]Validation failed:[/red]")
        for error in errors:
            console.print(f"[red]- {error}[/red]")
        raise typer.Exit(code=1)

    assert validated_config is not None
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(validated_config.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    console.print(f"Updated {setting_path} in {config_path}.")


@app.command("list-docs")
def list_docs(ctx: typer.Context) -> None:
    """List indexed documents with chunk counts and last updated timestamps."""

    config = _load_runtime_config(_get_context_config_path(ctx))
    orchestrator = EmbeddingOrchestrator.from_config(config)
    documents = orchestrator.list_documents()

    if not documents:
        console.print("No indexed documents found.")
        return

    table = Table(title="Indexed Documents")
    table.add_column("File", overflow="fold")
    table.add_column("Format")
    table.add_column("Chunks", justify="right")
    table.add_column("Last Updated")

    for document in documents:
        last_updated = document.get("latest_ingested_at") or "unknown"
        table.add_row(
            document.get("source_path", document.get("filename", "unknown")),
            str(document.get("source_type", "unknown")),
            str(document.get("chunk_count", 0)),
            last_updated,
        )

    console.print(table)


def main() -> None:
    """CLI entry point."""

    app()


if __name__ == "__main__":
    main()
