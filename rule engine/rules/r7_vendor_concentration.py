"""
R7 — Vendor Concentration & Local Dominance
----------------------------------------------
Detects: a vendor receiving an unusually large share of projects or
money within a constituency/area. NOT inherently fraud — a vendor may
legitimately be the only qualified local contractor — but worth
surfacing for review.

Indicator (per work_id, attributed to its vendor+constituency group):
    VendorProjectShare% = vendor's project count in area / total projects in area * 100
    VendorAmountShare%  = vendor's total amount in area / total amount in area * 100

Scoring:
    Average of the two share percentages, linearly scaled so that
    a 100% share (total monopoly) -> 100, and below 30% share -> 0.
    Every project belonging to that dominant vendor+area combo gets
    the same score (the concentration is a property of the group,
    not of an individual project).
"""

import pandas as pd
from config import COLUMNS
from utils.scoring import linear_score, make_result

RULE_ID = "R7"

SHARE_FLOOR = 30   # below 30% share -> score 0
SHARE_CEIL = 100   # 100% share (monopoly) -> score 100


def run(projects: pd.DataFrame) -> list[dict]:
    results = []
    c = COLUMNS

    if c["vendor"] not in projects.columns or c["constituency"] not in projects.columns:
        # can't run without vendor/area columns — mark all as not-applicable
        for _, row in projects.iterrows():
            results.append(make_result(RULE_ID, row[c["work_id"]], None, False,
                                        reason="Missing vendor or constituency column"))
        return results

    area_totals = projects.groupby(c["constituency"])[c["sanction_amount"]].sum()
    area_counts = projects.groupby(c["constituency"])[c["work_id"]].count()

    vendor_area_amounts = projects.groupby([c["constituency"], c["vendor"]])[c["sanction_amount"]].sum()
    vendor_area_counts = projects.groupby([c["constituency"], c["vendor"]])[c["work_id"]].count()

    for _, row in projects.iterrows():
        work_id = row[c["work_id"]]
        area = row.get(c["constituency"])
        vendor = row.get(c["vendor"])

        if pd.isna(area) or pd.isna(vendor):
            results.append(make_result(RULE_ID, work_id, None, False,
                                        reason="Missing vendor or constituency value"))
            continue

        total_amt = area_totals.get(area, 0)
        total_cnt = area_counts.get(area, 0)
        vendor_amt = vendor_area_amounts.get((area, vendor), 0)
        vendor_cnt = vendor_area_counts.get((area, vendor), 0)

        amount_share = (vendor_amt / total_amt * 100) if total_amt else 0
        project_share = (vendor_cnt / total_cnt * 100) if total_cnt else 0
        avg_share = (amount_share + project_share) / 2

        score = linear_score(avg_share, SHARE_FLOOR, SHARE_CEIL)
        triggered = avg_share > 50

        results.append(make_result(
            RULE_ID, work_id, score, triggered,
            indicator={"vendor_project_share_pct": round(project_share, 2),
                       "vendor_amount_share_pct": round(amount_share, 2)},
            evidence={"vendor": vendor, "constituency": area,
                      "vendor_project_count": int(vendor_cnt), "area_project_count": int(total_cnt)},
            reason=(f"Vendor '{vendor}' holds {avg_share:.0f}% share of projects/amount "
                    f"in {area}" if triggered else "Vendor distribution looks normal"),
        ))

    return results
