from __future__ import annotations

import asyncio
from time import monotonic

from .exceptions import RateLimitExceeded


class TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, capacity: int, refill_interval: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_interval <= 0:
            raise ValueError("refill_interval must be positive")

        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_interval = float(refill_interval)
        self.refill_rate = self.capacity / self.refill_interval
        self._updated_at = monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, amount: float = 1.0) -> None:
        async with self._lock:
            self._refill()
            if self.tokens < amount:
                deficit = amount - self.tokens
                retry_after = deficit / self.refill_rate if self.refill_rate > 0 else None
                raise RateLimitExceeded(retry_after)
            self.tokens -= amount

    def _refill(self) -> None:
        now = monotonic()
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
