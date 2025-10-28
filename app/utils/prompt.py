from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = (
    "You are a domain assistant. Use ONLY the provided glossary context to answer. "
    "If the context does not contain the information, respond with \"I don't know\". "
    "Cite sources inline using the format [#chunk_index] to reference the context chunks provided."
)


def build_chat_messages(
    question: str,
    context_chunks: Sequence[tuple[int, str]],
) -> list[dict[str, str]]:
    """Create chat completion messages using the retrieved context."""

    context_lines: list[str] = []
    for chunk_index, content in context_chunks:
        stripped = content.strip()
        if stripped:
            context_lines.append(f"[#{chunk_index}] {stripped}")
        else:
            context_lines.append(f"[#{chunk_index}] (blank chunk)")

    if context_lines:
        context_block = "\n".join(context_lines)
    else:
        context_block = "(No relevant context retrieved. Respond that you do not know the answer.)"

    user_message = (
        "Use the glossary context to answer the user's question.\n"
        "Question: "
        f"{question.strip()}\n\n"
        "Context:\n"
        f"{context_block}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
