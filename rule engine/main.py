"""
main.py
-------
Entry point. Loads projects.csv + payments.csv, runs all 11 rules,
combines them into a weighted final risk score, and writes:

    output/rule_scores_wide.csv   -> one row per project, one column per rule
    output/final_risk_report.csv  -> ranked final risk score + reasons
    output/rule_level_detail.json -> full per-rule evidence, for drill-down UIs

Usage:
    python main.py --projects sample_data/projects.csv --payments sample_data/payments.csv
"""

import argparse
import json
import os

import pandas as pd

from config import COLUMNS
from rule_runner import run_all_rules, results_to_dataframe
from risk_engine import compute_risk_scores


DATE_COLUMNS = [
    COLUMNS["expected_completion_date"],
    COLUMNS["recommendation_date"],
    COLUMNS["sanction_date"],
]


def load_data(projects_path: str, payments_path: str):
    projects = pd.read_csv(projects_path)
    payments = pd.read_csv(payments_path)

    for col in DATE_COLUMNS:
        if col in projects.columns:
            projects[col] = pd.to_datetime(projects[col], errors="coerce")
    if COLUMNS["work_id"] in projects.columns:
        payments["work_id"] = payments["work_id"]  # explicit, keeps join key visible

    from config import PAYMENT_COLUMNS
    if PAYMENT_COLUMNS["date"] in payments.columns:
        payments[PAYMENT_COLUMNS["date"]] = pd.to_datetime(payments[PAYMENT_COLUMNS["date"]], errors="coerce")

    # PATCH: projects.csv has no vendor column at all — R7 needs one.
    # ~6% of work_ids have more than one distinct vendor across their payments;
    # we attribute each project to whichever vendor received the LARGEST total
    # amount for that work_id (a defensible "dominant vendor" choice).
    vendor_col = PAYMENT_COLUMNS.get("work_id", "work_id")
    if "vendor_name_normalized" in payments.columns:
        vendor_totals = (
            payments.groupby([vendor_col, "vendor_name_normalized"])[PAYMENT_COLUMNS["amount"]]
            .sum()
            .reset_index()
        )
        dominant_vendor = (
            vendor_totals.sort_values(PAYMENT_COLUMNS["amount"], ascending=False)
            .drop_duplicates(subset=vendor_col)
            .set_index(vendor_col)["vendor_name_normalized"]
        )
        projects[COLUMNS["vendor"]] = projects[COLUMNS["work_id"]].map(dominant_vendor)

    return projects, payments


def main():
    parser = argparse.ArgumentParser(description="Run the 11-rule fraud/risk scoring engine.")
    parser.add_argument("--projects", default="sample_data/projects.csv")
    parser.add_argument("--payments", default="sample_data/payments.csv")
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    projects, payments = load_data(args.projects, args.payments)
    print(f"Loaded {len(projects)} projects and {len(payments)} payments.")

    rule_results = run_all_rules(projects, payments)
    print(f"Computed {len(rule_results)} rule-level results "
          f"({len(rule_results) // 11 if len(rule_results) else 0} approx. projects x 11 rules).")

    wide_scores = results_to_dataframe(rule_results)
    wide_scores.to_csv(os.path.join(args.outdir, "rule_scores_wide.csv"))

    risk_report = compute_risk_scores(rule_results)
    risk_report.to_csv(os.path.join(args.outdir, "final_risk_report.csv"), index=False)

    with open(os.path.join(args.outdir, "rule_level_detail.json"), "w") as f:
        json.dump(rule_results, f, default=str, indent=2)

    print("\nTop 10 highest-risk projects:")
    cols = ["rank", "work_id", "final_score", "risk_level", "top_reasons"]
    print(risk_report[cols].head(10).to_string(index=False))

    print(f"\nFull outputs written to: {args.outdir}/")


if __name__ == "__main__":
    main()
