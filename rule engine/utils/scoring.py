"""
utils/scoring.py
-----------------
Reusable helpers so every rule converts "how unusual is this value?"
into a 0-100 score the SAME way, instead of each rule inventing its
own ad-hoc math.

Two main strategies used across the 11 rules:

1. percentile_score(value, peer_series)
   -> "where does this value sit inside its peer group?"
      Good for R6, R11B (large payment / high sanction vs peers).

2. linear_score(value, low, high)
   -> simple clamp-and-scale between two thresholds you choose
      (e.g. overdue days, overrun %, mismatch %).
      Good for R1, R2, R5, R10, R11A.

3. robust_zscore(value, peer_series)
   -> median/MAD based z-score, more outlier-resistant than mean/std.
      Used as an alternative peer-comparison method.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def linear_score(value: float, low: float, high: float) -> float:
    """
    Scale `value` linearly onto 0-100 between [low, high].
    value <= low  -> 0
    value >= high -> 100
    Anything in between is interpolated.

    Use this when YOU define the meaningful cutoffs (e.g. "0 days
    overdue = 0, 120+ days overdue = 100").
    """
    if high == low:
        return 0.0 if value <= low else 100.0
    pct = (value - low) / (high - low) * 100
    return round(clamp(pct), 2)


def percentile_score(value: float, peer_values: pd.Series) -> float:
    """
    Where does `value` fall inside the distribution of `peer_values`?
    Returns 0-100, i.e. the percentile rank * 100.

    Use this when you want a DATA-DRIVEN cutoff instead of a
    hand-picked one (e.g. "this payment is bigger than 92% of
    payments for similar projects").
    """
    peers = pd.to_numeric(peer_values, errors="coerce").dropna()
    if len(peers) == 0:
        return 0.0
    # percentile rank of `value` within peers (inclusive)
    rank = (peers <= value).sum() / len(peers) * 100
    return round(clamp(rank), 2)


def robust_zscore_to_score(value: float, peer_values: pd.Series,
                            cap_z: float = 4.0) -> float:
    """
    Median/MAD based z-score -> 0-100 score.
    More resistant to outliers than mean/std z-score, because a
    couple of extreme peers won't distort the whole peer baseline.

    z is capped at `cap_z` and rescaled to 0-100, so a z of 0 -> 0
    and a z of cap_z (or more) -> 100.
    """
    peers = pd.to_numeric(peer_values, errors="coerce").dropna()
    if len(peers) < 3:
        # not enough peers for a meaningful comparison
        return 0.0
    median = peers.median()
    mad = (peers - median).abs().median()
    if mad == 0:
        # avoid divide-by-zero; fall back to whether value == median
        return 0.0 if value == median else 100.0
    z = 0.6745 * (value - median) / mad  # 0.6745 makes MAD ~ std for normal data
    z = max(0.0, z)  # only "unusually high" counts as risk here
    return round(clamp((z / cap_z) * 100), 2)


def make_result(rule_id: str, work_id, score, triggered: bool,
                 indicator: dict | None = None,
                 evidence: dict | None = None,
                 reason: str = "") -> dict:
    """
    Every rule returns results in this exact shape, so rule_runner
    and risk_engine don't need to know the internals of each rule.

    score = None means "this rule could not be evaluated for this
    project" (e.g. missing data) — NOT "risk is zero". risk_engine
    excludes None scores (and their weight) from the final average.
    """
    return {
        "rule_id": rule_id,
        "work_id": work_id,
        "score": None if score is None else round(float(score), 2),
        "triggered": bool(triggered),
        "indicator": indicator or {},
        "evidence": evidence or {},
        "reason": reason,
    }
