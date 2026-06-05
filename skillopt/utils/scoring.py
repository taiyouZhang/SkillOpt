"""Scoring and hashing utilities."""
from __future__ import annotations

import hashlib


def is_anomalous(r: object) -> bool:
    """Return True if a result should be excluded from scoring (timeout, error, etc.)."""
    if hasattr(r, "get"):
        fail_reason = r.get("fail_reason", "") or ""
        h = float(r.get("hard", 0) or 0)
        s = float(r.get("soft", 0) or 0)
    elif hasattr(r, "fail_reason"):
        fail_reason = r.fail_reason or ""
        h = float(r.hard if hasattr(r, "hard") else 0)
        s = float(r.soft if hasattr(r, "soft") else 0)
    else:
        return False
    if h < 1e-9 and s < 1e-9 and fail_reason:
        return True
    return False


def compute_score(results: list) -> tuple[float, float]:
    """Compute hard and soft accuracy from a list of episode results.

    Accepts both plain dicts and :class:    instances.  hard may be continuous (0.0-1.0) when using smoothed reward.
    Anomalous results (timeout, errors with hard=0 and soft=0) are excluded.
    """
    if not results:
        return 0.0, 0.0

    valid = [r for r in results if not is_anomalous(r)]
    if not valid:
        return 0.0, 0.0

    def _hard(r: object) -> float:
        return float(r.hard if hasattr(r, "hard") else r.get("hard", 0))

    def _soft(r: object) -> float:
        return float(r.soft if hasattr(r, "soft") else r.get("soft", 0.0))

    hard = sum(_hard(r) for r in valid) / len(valid)
    soft = sum(_soft(r) for r in valid) / len(valid)
    return hard, soft


def skill_hash(content: str) -> str:
    """Return a short deterministic hash of skill content (for caching)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
