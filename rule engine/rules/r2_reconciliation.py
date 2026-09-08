"""
R2 — Reconciliation Mismatch
------------------------------
Detects: the disbursed amount recorded on the PROJECT record doesn't
match the sum of PAYMENT records for that project. A direct
data/financial integrity signal.

Indicator:
    Mismatch% = |ProjectDisbursed - PaymentTotal| / ProjectDisbursed * 100

Scoring:
    Linear between 0% (score 0) and 40% (score 100) mismatch.
    If a project has zero payment records at all, score = None
    (can't reconcile what doesn't exist in the payments table —
    that's a data-completeness issue, not necessarily fraud).
"""

import pandas as pd
from config import COLUMNS, PAYMENT_COLUMNS
from utils.scoring import linear_score, make_result

RULE_ID = "R2"

MISMATCH_FLOOR = 0
MISMATCH_CEIL = 40


def run(projects: pd.DataFrame, payments: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS
    p = PAYMENT_COLUMNS

    payment_totals = (
        payments.groupby(p["work_id"])[p["amount"]]
        .sum()
        .to_dict()
    )

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        project_disbursed = row.get(c["total_disbursed"])

        if pd.isna(project_disbursed) or project_disbursed == 0:
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing/zero project disbursed amount"))
            continue

        payment_total = payment_totals.get(work_id)
        if payment_total is None:
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="No payment records found for this work_id"))
            continue

        mismatch_pct = abs(project_disbursed - payment_total) / project_disbursed * 100
        score = linear_score(mismatch_pct, MISMATCH_FLOOR, MISMATCH_CEIL)
        triggered = mismatch_pct > 5   # >5% mismatch is worth a look

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"mismatch_pct": round(mismatch_pct, 2)},
            evidence={"project_disbursed": project_disbursed, "payment_total": payment_total},
            reason=(f"Project record vs payment records differ by {mismatch_pct:.1f}%"
                    if triggered else "Amounts reconcile within tolerance"),
        ))

    return results
