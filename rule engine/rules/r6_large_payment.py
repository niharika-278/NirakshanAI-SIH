"""
R6 — Unusually Large Single Payment
--------------------------------------
Detects: a single payment that's unusually large compared to payments
made to PEER projects (same work_title / category).

Indicator:
    LargestPayment = max(payment_amount) for that work_id

Scoring:
    percentile_score of this project's largest payment against the
    peer group's distribution of largest payments.
    Peer group = other projects sharing the same work_title (falls
    back to the full dataset if the peer group is too small).
"""

import pandas as pd
from config import COLUMNS, PAYMENT_COLUMNS
from utils.scoring import percentile_score, make_result

RULE_ID = "R6"
MIN_PEER_GROUP_SIZE = 5


def run(projects: pd.DataFrame, payments: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS
    p = PAYMENT_COLUMNS

    largest_payment = payments.groupby(p["work_id"])[p["amount"]].max()
    title_lookup = projects.set_index(c["work_id"])[c["work_title"]].to_dict()

    # group largest payments by peer title for fast lookup
    by_title = {}
    for work_id, amt in largest_payment.items():
        title = title_lookup.get(work_id, "UNKNOWN")
        by_title.setdefault(title, []).append(amt)

    global_peer_series = largest_payment  # fallback peer pool = everyone

    for work_id, amt in largest_payment.items():
        title = title_lookup.get(work_id, "UNKNOWN")
        peers = pd.Series(by_title.get(title, []))

        if len(peers) < MIN_PEER_GROUP_SIZE:
            peers = global_peer_series  # not enough same-title peers, use global pool

        score = percentile_score(amt, peers)
        triggered = score >= 85  # top ~15% within its peer group

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"largest_payment": amt, "peer_group_size": len(peers)},
            evidence={"work_title": title, "peer_median": float(peers.median()) if len(peers) else None},
            reason=(f"Largest single payment (₹{amt:,.0f}) is an outlier vs peer "
                    f"projects" if triggered else "Largest payment is within normal range"),
        ))

    return results
