from __future__ import annotations

from typing import Any, Sequence


def normalize_conversation_messages(
    messages: str | Sequence[Any],
    max_messages: int,
) -> list[dict[str, str]]:
    if isinstance(messages, str):
        content = messages.strip()
        if not content:
            return []
        return [{"role": "user", "content": content}]

    normalized: list[dict[str, str]] = []
    for message in list(messages or []):
        role: str | None = None
        content: str | None = None

        if isinstance(message, dict):
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
        else:
            role = str(getattr(message, "role", "")).strip().lower()
            content = str(getattr(message, "content", "")).strip()

        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    if max_messages <= 0:
        return normalized
    return normalized[-max_messages:]


def latest_user_query(messages: Sequence[dict[str, str]]) -> str:
    for message in reversed(list(messages)):
        if message.get("role") == "user":
            return message.get("content", "").strip()
    for message in reversed(list(messages)):
        content = message.get("content", "").strip()
        if content:
            return content
    return ""


def conversation_context_for_prompt(messages: Sequence[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "").strip().lower()
        content = message.get("content", "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines).strip()
