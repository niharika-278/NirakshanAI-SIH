"""
R10 — Completion Delay / Overdue Project
--------------------------------------------
Detects: projects still incomplete after their expected completion
date has passed. (No actual_completion_date exists in this dataset,
so this measures OVERDUE STATUS, not the true completion delay.)

Indicator:
    OverdueDays = Today - ExpectedCompletionDate   (only if status != COMPLETED)

Scoring:
    Only projects that are BOTH incomplete AND overdue get a
    non-zero score. The score is the project's PERCENTILE among all
    other overdue (incomplete) projects — so cutoffs adapt to your
    data instead of a fixed "90 days = 100" rule, exactly as
    specified in the design doc.

Limitation (kept from the design doc): this measures overdue status,
not the exact number of days a project actually took to complete.
"""

import pandas as pd
from config import COLUMNS
from utils.scoring import percentile_score, make_result

RULE_ID = "R10"


def run(projects: pd.DataFrame, today: pd.Timestamp | None = None) -> list[dict]:
    results = []
    c = COLUMNS
    today = today or pd.Timestamp.now().normalize()

    overdue_days_map = {}
    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        status = str(row.get(c["status"], "")).upper()
        expected = row.get(c["expected_completion_date"])

        if status == "COMPLETED":
            continue  # rule doesn't apply — handled in second pass below
        if pd.isna(expected):
            continue

        expected_dt = pd.to_datetime(expected)
        overdue_days = (today - expected_dt).days
        if overdue_days > 0:
            overdue_days_map[work_id] = overdue_days

    peer_series = pd.Series(list(overdue_days_map.values()))

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        status = str(row.get(c["status"], "")).upper()
        expected = row.get(c["expected_completion_date"])

        if status == "COMPLETED":
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Not applicable — project COMPLETED"))
            continue
        if pd.isna(expected):
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing expected_completion_date"))
            continue

        overdue_days = overdue_days_map.get(work_id)
        if overdue_days is None:
            # incomplete but not yet past its expected date
            results.append(make_result(
                RULE_ID, work_id, 0.0, False,
                indicator={"overdue_days": 0},
                reason="Not yet overdue",
            ))
            continue

        score = percentile_score(overdue_days, peer_series)
        results.append(make_result(
            RULE_ID, work_id, score, True,
            indicator={"overdue_days": overdue_days},
            evidence={"expected_completion_date": str(expected)},
            reason=f"{overdue_days} days overdue vs expected completion date",
        ))

    return results
