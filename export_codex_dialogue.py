#!/usr/bin/env python3
"""Export user-visible dialogue from a Codex rollout JSONL file.

Only user and assistant message items are exported. Developer/system messages,
reasoning, tool calls, tool outputs, and injected environment/plugin blocks are
intentionally omitted so the result is suitable for a project handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


INJECTED_PREFIXES = (
    "<recommended_plugins>",
    "<environment_context>",
    "<skills_instructions>",
    "<permissions instructions>",
    "<collaboration_mode>",
    "<apps_instructions>",
    "<plugins_instructions>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Codex rollout JSONL")
    parser.add_argument("--output", required=True, help="Markdown export")
    parser.add_argument("--title", default="Codex user-visible dialogue export")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message_text(payload: dict) -> str:
    accepted = "input_text" if payload.get("role") == "user" else "output_text"
    parts = []
    for item in payload.get("content", []):
        if not isinstance(item, dict) or item.get("type") != accepted:
            continue
        value = "\n".join(
            line.rstrip() for line in str(item.get("text", "")).splitlines()
        ).strip()
        if not value or value.startswith(INJECTED_PREFIXES):
            continue
        parts.append(value)
    return "\n\n".join(parts).strip()


def main() -> None:
    args = parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    messages: list[tuple[str, str, str, str]] = []

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload") or {}
            if record.get("type") != "response_item":
                continue
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = message_text(payload)
            if not text:
                continue
            phase = str(payload.get("phase", ""))
            messages.append((str(record.get("timestamp", "")), role, phase, text))

    if not messages:
        raise RuntimeError("no user-visible messages found")

    first = messages[0][0]
    last = messages[-1][0]
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# {args.title}",
        "",
        "> This export contains only user messages and user-visible assistant replies.",
        "> System/developer instructions, reasoning, tool calls, tool outputs, and",
        "> injected environment/plugin metadata are intentionally excluded.",
        "",
        f"- Source session: `{source.name}`",
        f"- Source SHA256: `{sha256(source)}`",
        f"- First message (UTC): `{first}`",
        f"- Last exported message (UTC): `{last}`",
        f"- Export generated: `{generated}`",
        f"- Exported messages: `{len(messages)}`",
        "",
    ]
    for index, (timestamp, role, phase, text) in enumerate(messages, start=1):
        label = "User" if role == "user" else "Assistant"
        phase_suffix = f" · {phase}" if phase else ""
        lines.extend(
            [
                f"## {index}. {label}{phase_suffix}",
                "",
                f"Time (UTC): `{timestamp}`",
                "",
                text,
                "",
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "messages": len(messages),
                "bytes": output.stat().st_size,
                "source_sha256": sha256(source),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
