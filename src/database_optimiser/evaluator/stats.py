"""Statistical helpers for reward evaluation."""

from __future__ import annotations

import random
import statistics
from datetime import datetime
from hashlib import sha256


def winsorize(xs: list[float], pctl: float) -> list[float]:
    """Clamp values above the chosen percentile."""
    if not xs:
        return []
    xs2 = sorted(xs)
    idx = int(len(xs2) * pctl)
    idx = min(max(idx, 0), len(xs2) - 1)
    cap = xs2[idx]
    return [min(x, cap) for x in xs2]


def bootstrap_ci_lower_bound(
    baseline: list[float],
    new: list[float],
    iters: int,
    alpha: float,
    seed: int,
) -> float:
    """
    Bootstrap a lower bound for mean latency improvement:
    improvement = (mean(b) - mean(n)) / mean(b)
    """
    if not baseline or not new:
        return 0.0
    b_mean = statistics.mean(baseline)
    if b_mean <= 0:
        return 0.0
    rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
    samples: list[float] = []
    for _ in range(max(1, iters)):
        b = [rng.choice(baseline) for _ in range(len(baseline))]
        n = [rng.choice(new) for _ in range(len(new))]
        b_m = statistics.mean(b)
        n_m = statistics.mean(n)
        if b_m <= 0:
            continue
        samples.append((b_m - n_m) / b_m)
    if not samples:
        return 0.0
    samples.sort()
    k = int(len(samples) * alpha)
    k = min(max(k, 0), len(samples) - 1)
    return float(samples[k])


def bootstrap_seed(
    *,
    layout_id: str,
    baseline_layout_id: str,
    start_time: datetime,
    end_time: datetime,
) -> int:
    s = (
        f"{layout_id}|{baseline_layout_id}|{start_time.isoformat()}|{end_time.isoformat()}"
    ).encode("utf-8")
    return int.from_bytes(sha256(s).digest()[:8], "big")
