"""
R1 — Completed but Underfunded
-------------------------------
Detects: projects marked COMPLETED where only a small fraction of the
sanctioned money was actually disbursed. A finished project with very
low fund utilization is worth verifying.

Indicator:
    Utilization%  = TotalDisbursed / SanctionAmount * 100
    Shortfall%    = 100 - Utilization%

Scoring:
    Only evaluated for COMPLETED projects.
    Higher shortfall -> higher score (linear between 0% and 70% shortfall).
    A COMPLETED project with 0% shortfall (fully utilized) -> score 0.
    A COMPLETED project with >=70% shortfall -> score 100.
"""

import pandas as pd
from config import COLUMNS
from utils.scoring import linear_score, make_result

RULE_ID = "R1"

# tune these two numbers to change how aggressively shortfall is punished
SHORTFALL_FLOOR = 0    # 0% shortfall -> score 0
SHORTFALL_CEIL = 70    # 70%+ shortfall -> score 100


def run(projects: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        status = str(row.get(c["status"], "")).upper()

        if status != "COMPLETED":
            # rule only applies to completed projects
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Not applicable — project not COMPLETED"))
            continue

        sanctioned = row.get(c["sanction_amount"])
        disbursed = row.get(c["total_disbursed"])

        if pd.isna(sanctioned) or sanctioned == 0 or pd.isna(disbursed):
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing/zero sanction or disbursed amount"))
            continue

        utilization_pct = (disbursed / sanctioned) * 100
        shortfall_pct = 100 - utilization_pct

        score = linear_score(shortfall_pct, SHORTFALL_FLOOR, SHORTFALL_CEIL)
        triggered = shortfall_pct > 10   # flag if more than 10% shortfall

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"utilization_pct": round(utilization_pct, 2),
                       "shortfall_pct": round(shortfall_pct, 2)},
            evidence={"sanction_amount": sanctioned, "total_disbursed": disbursed},
            reason=(f"Completed with only {utilization_pct:.1f}% of sanctioned "
                    f"amount disbursed" if triggered else "Utilization looks normal"),
        ))

    return results
