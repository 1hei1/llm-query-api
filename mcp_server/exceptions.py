from __future__ import annotations


class RateLimitExceeded(Exception):
    """Raised when a tool invocation exceeds its rate limit."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after is not None:
            message = f"{message}. Retry after {retry_after:.2f} seconds."
        super().__init__(message)
