"""
R5 — Cost / Sanction Overrun
------------------------------
Detects: disbursed amount exceeding the sanctioned amount. A direct
financial control breach if it happens.

Indicator:
    Overrun% = (Disbursed - Sanctioned) / Sanctioned * 100

Scoring:
    0% or negative overrun -> 0
    Linear up to a defined ceiling (e.g. 50% overrun -> 100)

Note: your current dataset may show 0 hits for this rule if amounts
are pre-capped at the sanctioned value — that's still a valid,
correctly-implemented control, just with no observed violations yet.
Don't delete the rule because hits == 0.
"""

import pandas as pd
from config import COLUMNS
from utils.scoring import linear_score, make_result

RULE_ID = "R5"

OVERRUN_FLOOR = 0
OVERRUN_CEIL = 50


def run(projects: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        sanctioned = row.get(c["sanction_amount"])
        disbursed = row.get(c["total_disbursed"])

        if pd.isna(sanctioned) or sanctioned == 0 or pd.isna(disbursed):
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing/zero sanction or disbursed amount"))
            continue

        overrun_pct = (disbursed - sanctioned) / sanctioned * 100
        triggered = overrun_pct > 0
        score = linear_score(overrun_pct, OVERRUN_FLOOR, OVERRUN_CEIL) if triggered else 0.0

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"overrun_pct": round(overrun_pct, 2)},
            evidence={"sanction_amount": sanctioned, "total_disbursed": disbursed},
            reason=(f"Disbursed exceeds sanction by {overrun_pct:.1f}%"
                    if triggered else "No overrun"),
        ))

    return results
