"""
risk_engine.py
--------------
Takes the flat list of per-rule results (from rule_runner) and, for
EVERY work_id, computes:

    FinalRisk = sum(score_i * weight_i) / sum(weight_i)     [i over rules with a non-None score]

    NOTE the denominator is the sum of AVAILABLE weights only.
    If R2 couldn't run for a project (score=None, e.g. no payment
    records), R2's weight is excluded from BOTH the numerator and the
    denominator — so missing data never silently drags the score down
    to look artificially safe. See config.py section 13 rationale.

Also collects the "evidence" from every TRIGGERED rule so the final
report is explainable, not just a bare number.
"""

from collections import defaultdict
import pandas as pd

from config import WEIGHTS, score_to_band


def compute_risk_scores(rule_results: list[dict]) -> pd.DataFrame:
    """
    rule_results: flat list of dicts from rule_runner.run_all_rules()

    Returns a DataFrame, one row per work_id, with columns:
        work_id, final_score, risk_level, rules_used, rules_available,
        <RULE_ID>_score for every rule, and a `top_reasons` column
        listing the reasons from triggered rules (highest weight first).
    """
    by_work_id = defaultdict(list)
    for r in rule_results:
        by_work_id[r["work_id"]].append(r)

    rows = []
    for work_id, results in by_work_id.items():
        weighted_sum = 0.0
        total_weight = 0.0
        rules_used = 0
        per_rule_scores = {}
        triggered_reasons = []  # (weight, rule_id, reason)

        for r in results:
            rule_id = r["rule_id"]
            weight = WEIGHTS.get(rule_id, 0)
            per_rule_scores[f"{rule_id}_score"] = r["score"]

            if r["score"] is not None:
                weighted_sum += r["score"] * weight
                total_weight += weight
                rules_used += 1

            if r["triggered"]:
                triggered_reasons.append((weight, rule_id, r["reason"]))

        final_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else None
        risk_level = score_to_band(final_score) if final_score is not None else "UNKNOWN"

        # most influential triggered reasons first
        triggered_reasons.sort(key=lambda x: x[0], reverse=True)
        top_reasons = "; ".join(f"[{rid}] {reason}" for _, rid, reason in triggered_reasons)

        row = {
            "work_id": work_id,
            "final_score": final_score,
            "risk_level": risk_level,
            "rules_used": rules_used,
            "rules_available": len(results),
            "top_reasons": top_reasons,
        }
        row.update(per_rule_scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    # rank highest risk first, so this is directly "who to investigate first"
    df = df.sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
