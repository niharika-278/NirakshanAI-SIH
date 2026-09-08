"""
R11B — High Sanction Amount vs Peers
-----------------------------------------
Detects: a project's sanctioned amount is unusually high compared to
similar projects (peer group = same work_title, per the design doc).
A pre-expenditure financial signal — flags unusual allocations before
looking at how the money was actually spent.

Indicator:
    Project's sanction_amount vs the distribution of sanction_amount
    for other projects sharing the same work_title.

Scoring:
    robust_zscore_to_score (median/MAD based) — resistant to a couple
    of other extreme peers skewing the baseline. Falls back to the
    global sanction_amount distribution if the peer group is too small.

Does NOT prove overpricing/fraud — flags for scrutiny only.
"""

import pandas as pd
from config import COLUMNS
from utils.scoring import robust_zscore_to_score, make_result

RULE_ID = "R11B"
MIN_PEER_GROUP_SIZE = 5


def run(projects: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS

    global_amounts = projects[c["sanction_amount"]]
    grouped = projects.groupby(c["work_title"])[c["sanction_amount"]]
    peer_groups = {title: grp for title, grp in grouped}

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        amount = row.get(c["sanction_amount"])
        title = row.get(c["work_title"])

        if pd.isna(amount):
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing sanction_amount"))
            continue

        peers = peer_groups.get(title, pd.Series(dtype=float))
        # exclude the project itself from its own peer comparison
        peers = peers[peers.index != row.name]
        if len(peers) < MIN_PEER_GROUP_SIZE:
            peers = global_amounts[global_amounts.index != row.name]

        score = robust_zscore_to_score(amount, peers)
        triggered = score >= 70

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"sanction_amount": amount, "peer_median": float(peers.median()) if len(peers) else None},
            evidence={"work_title": title, "peer_group_size": len(peers)},
            reason=(f"Sanction amount (₹{amount:,.0f}) is a strong outlier vs similar "
                    f"projects (peer median ₹{peers.median():,.0f})" if triggered and len(peers)
                    else "Sanction amount within normal range for peer group"),
        ))

    return results
