"""
rule_runner.py
--------------
Runs every rule against the loaded data and returns ONE FLAT LIST of
rule-result dicts (see utils.scoring.make_result for the shape).

This file knows WHICH rules need which datasets (some need only
projects, some need projects + payments), but it does NOT know how
any individual rule computes its score — that logic lives inside
rules/*.py. This separation is what makes it easy to add a 12th rule
later without touching risk_engine.py.
"""

import pandas as pd

from rules import (
    r1_underfunded,
    r2_reconciliation,
    r3_payment_timing,
    r4_payment_burst,
    r5_sanction_overrun,
    r6_large_payment,
    r7_vendor_concentration,
    r8_duplicate,
    r9_missing_evidence,
    r10_completion_delay,
    r11a_recommendation_delay,
    r11b_high_sanction,
)


def run_all_rules(projects: pd.DataFrame, payments: pd.DataFrame,
                   today: pd.Timestamp | None = None) -> list[dict]:
    """
    Returns a flat list like:
        [
          {"rule_id": "R1", "work_id": "W001", "score": 70.0, ...},
          {"rule_id": "R2", "work_id": "W001", "score": 20.0, ...},
          ...
        ]
    One entry per (rule, project) pair.
    """
    all_results: list[dict] = []

    all_results += r1_underfunded.run(projects)
    all_results += r2_reconciliation.run(projects, payments)
    all_results += r3_payment_timing.run(projects, payments)
    all_results += r4_payment_burst.run(payments)
    all_results += r5_sanction_overrun.run(projects)
    all_results += r6_large_payment.run(projects, payments)
    all_results += r7_vendor_concentration.run(projects)
    all_results += r8_duplicate.run(projects)
    all_results += r9_missing_evidence.run(projects)
    all_results += r10_completion_delay.run(projects, today=today)
    all_results += r11a_recommendation_delay.run(projects)
    all_results += r11b_high_sanction.run(projects)

    return all_results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten the list of rule-result dicts into a wide DataFrame:
    one row per work_id, one column per rule's score — handy for
    quick inspection / export to CSV."""
    df = pd.DataFrame(results)
    wide = df.pivot_table(index="work_id", columns="rule_id", values="score", aggfunc="first")
    return wide
