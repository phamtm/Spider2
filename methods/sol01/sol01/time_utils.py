"""Small helpers for formatting human-readable durations."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Format elapsed time with compact human-readable units."""

    total_seconds = max(0.0, seconds)
    if total_seconds < 60:
        text = f"{total_seconds:.1f}".rstrip("0").rstrip(".")
        return f"{text}s"

    rounded_seconds = int(round(total_seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"
