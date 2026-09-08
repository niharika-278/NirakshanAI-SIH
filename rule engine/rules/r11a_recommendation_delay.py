"""
R11A — Recommendation-to-Sanction Delay
--------------------------------------------
Detects: projects that took unusually long to get sanctioned after
being recommended. A process/efficiency anomaly (lower weight),
not a direct financial irregularity.

Indicator:
    ProcessingDays = SanctionDate - RecommendationDate

Scoring:
    percentile_score of this project's processing time against ALL
    other projects' processing times — "unusual" is defined relative
    to your own data's typical turnaround, not a fixed day count.
"""

import pandas as pd
from config import COLUMNS
from utils.scoring import percentile_score, make_result

RULE_ID = "R11A"


def run(projects: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS

    processing_days_map = {}
    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        rec_date = row.get(c["recommendation_date"])
        sanc_date = row.get(c["sanction_date"])
        if pd.isna(rec_date) or pd.isna(sanc_date):
            continue
        days = (pd.to_datetime(sanc_date) - pd.to_datetime(rec_date)).days
        if days >= 0:
            processing_days_map[work_id] = days

    peer_series = pd.Series(list(processing_days_map.values()))

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        days = processing_days_map.get(work_id)

        if days is None:
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing recommendation/sanction date"))
            continue

        score = percentile_score(days, peer_series)
        triggered = score >= 85

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"processing_days": days},
            reason=(f"Took {days} days to sanction — unusually long vs peers"
                    if triggered else "Normal recommendation-to-sanction turnaround"),
        ))

    return results
