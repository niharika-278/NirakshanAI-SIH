"""
R4 — Payment Burst Pattern
----------------------------
Detects: a large share of a project's total payments happening inside
a very short rolling window (default 7 days). Can be legitimate
(e.g. milestone payment), so it's a supporting signal, not proof.

Indicator:
    BurstRatio% = Max 7-day rolling payment amount / Total project payment * 100

Scoring:
    Compared against the peer distribution of burst ratios across ALL
    projects (percentile_score), since "what's a normal burst ratio"
    depends on your data, not a fixed number.
"""

import pandas as pd
from config import PAYMENT_COLUMNS, THRESHOLDS
from utils.scoring import percentile_score, make_result

RULE_ID = "R4"


def _max_rolling_window_amount(dates: pd.Series, amounts: pd.Series, window_days: int) -> float:
    """
    For a single project's payments, find the maximum amount paid
    inside ANY `window_days`-day window (not just calendar-aligned
    weeks — a true sliding window).
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "amount": amounts}).sort_values("date")
    if df.empty:
        return 0.0

    max_amt = 0.0
    window = pd.Timedelta(days=window_days)
    # simple O(n^2) sliding window — fine for typical per-project payment counts;
    # switch to a two-pointer approach if a single project has thousands of payments
    dates_arr = df["date"].values
    amounts_arr = df["amount"].values
    start = 0
    running_sum = 0.0
    for end in range(len(df)):
        running_sum += amounts_arr[end]
        while dates_arr[end] - dates_arr[start] > window:
            running_sum -= amounts_arr[start]
            start += 1
        max_amt = max(max_amt, running_sum)
    return max_amt


def run(payments: pd.DataFrame) -> list[dict]:
    results = []
    p = PAYMENT_COLUMNS
    window_days = THRESHOLDS["R4_WINDOW_DAYS"]

    # first pass: compute burst ratio per work_id
    # PATCH: 21,955 of 30,562 projects (72%) have exactly ONE payment record.
    # For a single payment, burst_ratio_pct is mathematically ALWAYS 100%
    # (the whole total trivially falls inside any window containing it),
    # which isn't a real "burst" — it's an artifact of having 1 data point.
    # Original code had no guard for this, so R4 fired on 93% of projects,
    # which also polluted the percentile peer pool used to score everyone
    # else. Require at least 3 payments before this rule applies.
    MIN_PAYMENTS = 3
    burst_ratios = {}
    for work_id, group in payments.groupby(p["work_id"]):
        total = group[p["amount"]].sum()
        if total == 0 or len(group) < MIN_PAYMENTS:
            continue
        max_window_amt = _max_rolling_window_amount(group[p["date"]], group[p["amount"]], window_days)
        burst_ratios[work_id] = {
            "burst_ratio_pct": max_window_amt / total * 100,
            "max_window_amount": max_window_amt,
            "total_payment": total,
        }

    peer_series = pd.Series([v["burst_ratio_pct"] for v in burst_ratios.values()])

    evaluated_ids = set(burst_ratios.keys())
    for work_id in payments[p["work_id"]].unique():
        if work_id not in evaluated_ids:
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason=f"Fewer than {MIN_PAYMENTS} payments — burst pattern not meaningful"))

    for work_id, vals in burst_ratios.items():
        score = percentile_score(vals["burst_ratio_pct"], peer_series)
        triggered = vals["burst_ratio_pct"] > 60  # more than 60% of money in one week

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"burst_ratio_pct": round(vals["burst_ratio_pct"], 2)},
            evidence={"max_window_amount": vals["max_window_amount"],
                      "total_payment": vals["total_payment"],
                      "window_days": window_days},
            reason=(f"{vals['burst_ratio_pct']:.1f}% of payments concentrated in a "
                    f"{window_days}-day window" if triggered else "Payments spread normally"),
        ))

    return results
