"""
R3 — Payment Timing Integrity
--------------------------------
Detects: a payment that was made BEFORE its project was even sanctioned
(e.g. sanction_date = 15 March, but a payment is recorded on 10 March).
This is a logical/sequencing impossibility in a well-run scheme —
money shouldn't move before the sanction that authorizes it exists.

Indicator:
    DaysEarly = SanctionDate - EarliestPaymentDate   (only if > 0)

Scoring:
    0 (or negative) days early -> 0, not triggered.
    Linear between 0 and 60 days early -> 0 to 100.
    (On the reference dataset this rule currently finds ZERO violations
    across all ~50,000 payment records — every payment lands on or after
    its project's sanction date. Kept in the engine anyway, same
    rationale as R5/Cost Overrun: a correctly-implemented control with
    no observed hits yet is still a control worth having, in case future
    data (or a different state's export) does contain a violation.)
"""

import pandas as pd
from config import COLUMNS, PAYMENT_COLUMNS
from utils.scoring import linear_score, make_result

RULE_ID = "R3"

DAYS_EARLY_FLOOR = 0
DAYS_EARLY_CEIL = 60


def run(projects: pd.DataFrame, payments: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS
    p = PAYMENT_COLUMNS

    earliest_payment = (
        payments.groupby(p["work_id"])[p["date"]]
        .min()
        .to_dict()
    )

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        sanction_date = row.get(c["sanction_date"])

        if pd.isna(sanction_date):
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing sanction_date"))
            continue

        first_payment_date = earliest_payment.get(work_id)
        if first_payment_date is None or pd.isna(first_payment_date):
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="No payment records found for this work_id"))
            continue

        sanction_dt = pd.to_datetime(sanction_date)
        payment_dt = pd.to_datetime(first_payment_date)
        days_early = (sanction_dt - payment_dt).days

        triggered = days_early > 0
        score = linear_score(days_early, DAYS_EARLY_FLOOR, DAYS_EARLY_CEIL) if triggered else 0.0

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"days_early": max(days_early, 0)},
            evidence={"sanction_date": str(sanction_date), "earliest_payment_date": str(first_payment_date)},
            reason=(f"Earliest payment occurred {days_early} day(s) BEFORE sanction — sequencing violation"
                    if triggered else "All payments occur on or after sanction date"),
        ))

    return results
