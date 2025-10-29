from __future__ import annotations

from typing import Iterable
from uuid import uuid4


def generate_request_id() -> str:
    """Return a random request identifier suitable for trace correlation."""

    return uuid4().hex


def sanitize_terms(terms: Iterable[str]) -> list[str]:
    """Return normalized terms by stripping whitespace and discarding empties."""

    normalized: list[str] = []
    for raw in terms:
        value = (raw or "").strip()
        if not value:
            continue
        normalized.append(value)
    return normalized
