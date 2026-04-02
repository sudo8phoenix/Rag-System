"""Prompt construction helpers for grounded RAG generation."""

from __future__ import annotations

from typing import Mapping, Sequence


def _stringify_context_item(item: str | Mapping[str, object], index: int) -> str:
    if isinstance(item, str):
        return f"[{index}] {item.strip()}"

    text = str(item.get("text", "")).strip()
    source = str(item.get("source", "unknown"))
    if text:
        return f"[{index}] ({source}) {text}"
    return f"[{index}] ({source})"


def format_context(context_items: Sequence[str | Mapping[str, object]] | None) -> str:
    """Render context chunks into a deterministic prompt block."""

    if not context_items:
        return "No context provided."

    lines = [_stringify_context_item(item, idx) for idx, item in enumerate(context_items, start=1)]
    return "\n".join(lines)


def build_user_prompt(
    query: str,
    context_items: Sequence[str | Mapping[str, object]] | None = None,
) -> str:
    """Create the user prompt with explicit context and question sections."""

    context_block = format_context(context_items)
    return (
        "Use the context below to answer the question. "
        "If the answer is not in the context, explicitly say you do not know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query.strip()}"
    )
