#!/usr/bin/env python3
"""
prepare_state_data.py

Reusable data-preparation pipeline for MPLADS (Members of Parliament Local
Area Development Scheme) state-level portal exports.

Given three raw portal CSV exports for a single state:
  1. Works Sanctioned
  2. Works Completed
  3. Expenditure on Completed and On-going Works as on Date

...this program produces three clean, analysis-ready CSV files:
  <state>_payment_events.csv        (one row per expenditure/payment event)
  <state>_projects.csv              (one row per sanctioned Work ID)
  <state>_data_quality_report.csv   (data quality findings)

Usage:
    python prepare_state_data.py \
        --state "Chandigarh" \
        --sanctioned "Works Sanctioned.csv" \
        --completed "Works Completed.csv" \
        --expenditure "Expenditure on Completed and On-going Works as on Date.csv" \
        --output-dir "./output" \
        --as-of-date 2026-09-02

Notes:
  - Input files are never modified.
  - This script performs ONLY data cleaning / preparation / joining.
    It does not compute any ML model, anomaly score, fraud score, or
    risk prediction of any kind.
"""

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def log(msg):
    print(msg)


def safe_state_slug(state_name):
    """Lowercase, safe, filesystem-friendly slug for a state name."""
    s = state_name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "state"


def normalize_whitespace(value):
    """Trim whitespace and collapse repeated internal spaces."""
    if pd.isna(value):
        return value
    s = str(value)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_amount(value):
    """
    Convert a raw amount field (possibly containing currency symbols,
    commas, Indian-style grouping, or stray whitespace) into a numeric
    float. Returns np.nan if the value cannot be parsed.
    """
    if pd.isna(value):
        return np.nan
    s = str(value)
    s = s.replace("\xa0", " ").strip()
    if s == "" or s.lower() in ("nan", "none", "-", "na", "n/a"):
        return np.nan
    # Remove currency symbols, commas, spaces; keep digits, dot, minus
    s = re.sub(r"[₹$,\s]", "", s)
    s = s.replace("Rs.", "").replace("Rs", "")
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except ValueError:
        # Attempt to strip any remaining non-numeric characters except . and -
        s2 = re.sub(r"[^0-9.\-]", "", s)
        if s2 in ("", "-", "."):
            return np.nan
        try:
            return float(s2)
        except ValueError:
            return np.nan


def parse_date(value):
    """
    Parse portal-style dates such as '03-Jul-2025' or '03-Jul-25' into
    a pandas.Timestamp (normalized to midnight). Returns pd.NaT if the
    value cannot be parsed.
    """
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "-", "na", "n/a"):
        return pd.NaT
    # Try a handful of explicit formats first (day-month-year variants),
    # since these are the formats actually used by the MPLADS portal.
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%d/%m/%Y", "%d/%m/%y",
                "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    # Fall back to pandas' general parser (day-first, since this is Indian
    # government data).
    try:
        return pd.to_datetime(s, dayfirst=True, errors="raise")
    except Exception:
        return pd.NaT


def to_iso(ts):
    """Format a Timestamp as YYYY-MM-DD, or '' if NaT."""
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


_WORK_ID_PATTERN = re.compile(r'^(WS/MP\d+/\d{4}-\d{4}/\d+)-(.*)$', re.DOTALL)


def split_work_column(value):
    """
    Split a 'Work' column value of the form
        'WS/MP18047/2024-2025/155610-Providing CCTV camera system'
    into (work_id, work_title).

    IMPORTANT: this matches the known, fixed work-ID format directly via
    regex, rather than locating the last '/' in the string. An earlier
    version used `s.rfind("/")` to find the split point, which silently
    broke on ~8% of real Rajasthan rows (184/2252) whose work TITLE also
    contains a '/' character (e.g. "Purchase of books for public
    libraries/ digitization of libraries", "Crematoriums/energy efficient
    crematoriums..."). In those cases rfind("/") found the slash inside
    the title, not the one before the numeric serial, and the whole raw
    string was returned as the "work_id" -- corrupting the join key for
    those rows so they could never match Expenditure or Completed data.
    Matching the fixed ID pattern directly avoids this entirely, since
    the title can contain any characters including '/' or '-' without
    affecting the match.
    """
    if pd.isna(value):
        return (np.nan, np.nan)
    s = str(value).strip()
    m = _WORK_ID_PATTERN.match(s)
    if m:
        work_id = m.group(1).strip()
        work_title = normalize_whitespace(m.group(2).strip())
        return (work_id, work_title)
    # Unexpected format (doesn't match the known ID pattern at all).
    # Fall back to splitting on first hyphen so we don't silently lose
    # the row, but this should be rare -- worth checking manually if it
    # ever fires on real data.
    parts = s.split("-", 1)
    if len(parts) == 2:
        return (parts[0].strip(), normalize_whitespace(parts[1].strip()))
    return (s, np.nan)


def normalize_vendor(value):
    """Normalized vendor name for grouping/analysis purposes only."""
    if pd.isna(value):
        return ""
    s = normalize_whitespace(value)
    s = s.upper()
    # Collapse common punctuation variants that don't change vendor identity.
    s = re.sub(r"[.,]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def filter_numeric_sr_no(df, sr_col):
    """
    Keep only rows whose Sr. No. column is numeric. This drops blank
    rows and the portal's trailing 'Grand Total' row in one step.
    Returns (filtered_df, n_input_rows, n_dropped_rows).
    """
    n_input = len(df)
    sr_numeric = pd.to_numeric(
        df[sr_col].astype(str).str.strip(), errors="coerce"
    )
    mask = sr_numeric.notna()
    out = df.loc[mask].copy()
    n_dropped = n_input - len(out)
    return out, n_input, n_dropped


def find_col(df, *candidates):
    """Find the first matching column name (case/whitespace tolerant)."""
    norm_map = {re.sub(r"\s+", " ", c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\s+", " ", cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    return None


# --------------------------------------------------------------------------
# Loaders for each raw file
# --------------------------------------------------------------------------

def load_sanctioned(path, state):
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    raw.columns = [c.strip() for c in raw.columns]

    sr_col = find_col(raw, "Sr. No.", "Sr.No.", "Sr No")
    work_col = find_col(raw, "Work")
    cat_col = find_col(raw, "Work category", "Work Category")
    ida_col = find_col(raw, "IDA")
    mp_col = find_col(raw, "Hon'ble Members of Parliament")
    const_col = find_col(raw, "Constituency")
    desc_col = find_col(raw, "Work description", "Work Description")
    rec_date_col = find_col(raw, "Recommended date", "Recommended Date")
    sanc_date_col = find_col(raw, "Sanction Date")
    sanc_amt_col = find_col(raw, "Sanction Amount ( ₹ )", "Sanction Amount")
    status_col = find_col(raw, "Work Status")

    df, n_input, n_dropped_blank = filter_numeric_sr_no(raw, sr_col)

    work_ids, work_titles = zip(*df[work_col].map(split_work_column)) if len(df) else ([], [])
    df["work_id"] = list(work_ids)
    df["work_title"] = list(work_titles)

    out = pd.DataFrame({
        "work_id": df["work_id"],
        "state": state,
        "ida": df[ida_col].map(normalize_whitespace) if ida_col else "",
        "mp_name": df[mp_col].map(normalize_whitespace) if mp_col else "",
        "constituency": df[const_col].map(normalize_whitespace) if const_col else "",
        "work_category": df[cat_col].map(normalize_whitespace) if cat_col else "",
        "work_title": df["work_title"],
        "work_description": df[desc_col].map(normalize_whitespace) if desc_col else "",
        "recommended_date": df[rec_date_col].map(parse_date).map(to_iso) if rec_date_col else "",
        "sanction_date_raw": df[sanc_date_col].map(parse_date) if sanc_date_col else pd.NaT,
        "sanction_amount": df[sanc_amt_col].map(clean_amount) if sanc_amt_col else np.nan,
        "portal_execution_status": df[status_col].map(normalize_whitespace) if status_col else "",
        "source_serial_number": pd.to_numeric(df[sr_col], errors="coerce"),
    })
    out["sanction_date"] = out["sanction_date_raw"].map(to_iso)

    stats = {
        "file": "Works Sanctioned",
        "input_rows": n_input,
        "cleaned_rows": len(out),
        "dropped_blank_or_total_rows": n_dropped_blank,
    }
    return out, stats


def load_completed(path, state):
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    raw.columns = [c.strip() for c in raw.columns]

    sr_col = find_col(raw, "Sr. No.", "Sr.No.", "Sr No")
    work_col = find_col(raw, "Work")
    img_col = find_col(raw, "Image")
    comp_date_col = find_col(raw, "Completion Date")
    disb_col = find_col(raw, "Amount Disbursed ( ₹ )", "Amount Disbursed")

    df, n_input, n_dropped_blank = filter_numeric_sr_no(raw, sr_col)

    work_ids, work_titles = zip(*df[work_col].map(split_work_column)) if len(df) else ([], [])
    df["work_id"] = list(work_ids)

    out = pd.DataFrame({
        "work_id": df["work_id"],
        "image_reference": df[img_col].map(normalize_whitespace) if img_col else "",
        "completion_date_raw": df[comp_date_col].map(parse_date) if comp_date_col else pd.NaT,
        "completion_reported_disbursed_amount": df[disb_col].map(clean_amount) if disb_col else np.nan,
        "completed_source_serial_number": pd.to_numeric(df[sr_col], errors="coerce"),
    })
    out["completion_date"] = out["completion_date_raw"].map(to_iso)

    stats = {
        "file": "Works Completed",
        "input_rows": n_input,
        "cleaned_rows": len(out),
        "dropped_blank_or_total_rows": n_dropped_blank,
    }
    return out, stats


def load_expenditure(path, state):
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    raw.columns = [c.strip() for c in raw.columns]

    sr_col = find_col(raw, "Sr. No.", "Sr.No.", "Sr No")
    workid_col = find_col(raw, "Work ID")
    ida_col = find_col(raw, "IDA")
    mp_col = find_col(raw, "Hon'ble Members of Parliament")
    const_col = find_col(raw, "Constituency")
    exp_date_col = find_col(raw, "Expenditure Date")
    vendor_col = find_col(raw, "Vendor Name")
    status_col = find_col(raw, "Payment Status")
    amt_col = find_col(raw, "Fund Disbursed Amount ( ₹ )", "Fund Disbursed Amount")

    df, n_input, n_dropped_blank = filter_numeric_sr_no(raw, sr_col)

    out = pd.DataFrame({
        "work_id": df[workid_col].map(normalize_whitespace) if workid_col else "",
        "state": state,
        "ida": df[ida_col].map(normalize_whitespace) if ida_col else "",
        "mp_name": df[mp_col].map(normalize_whitespace) if mp_col else "",
        "constituency": df[const_col].map(normalize_whitespace) if const_col else "",
        "expenditure_date_raw": df[exp_date_col].map(parse_date) if exp_date_col else pd.NaT,
        "vendor_name_raw": df[vendor_col].map(normalize_whitespace) if vendor_col else "",
        "payment_status": df[status_col].map(normalize_whitespace).str.upper() if status_col else "",
        "reported_fund_disbursed_amount": df[amt_col].map(clean_amount) if amt_col else np.nan,
        "source_serial_number": pd.to_numeric(df[sr_col], errors="coerce"),
    })
    out["expenditure_date"] = out["expenditure_date_raw"].map(to_iso)
    out["vendor_name_normalized"] = out["vendor_name_raw"].map(normalize_vendor)

    stats = {
        "file": "Expenditure on Completed and On-going Works",
        "input_rows": n_input,
        "cleaned_rows": len(out),
        "dropped_blank_or_total_rows": n_dropped_blank,
    }
    return out, stats


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def build_payment_events(expenditure_df):
    cols = [
        "work_id", "state", "ida", "mp_name", "constituency",
        "expenditure_date", "vendor_name_raw", "vendor_name_normalized",
        "payment_status", "reported_fund_disbursed_amount",
        "source_serial_number",
    ]
    return expenditure_df[cols].copy()


def build_projects(sanctioned_df, completed_df, expenditure_df):
    df = sanctioned_df.copy()

    # --- Join completion info (Works Completed can have duplicate work_ids
    # in principle; keep the row with the latest completion date per work_id
    # so the projects table stays one-row-per-sanctioned-Work-ID.)
    comp = completed_df.sort_values("completion_date_raw").drop_duplicates(
        "work_id", keep="last"
    )
    df = df.merge(
        comp[["work_id", "image_reference", "completion_date",
              "completion_date_raw", "completion_reported_disbursed_amount"]],
        on="work_id", how="left"
    )

    # --- work_completion_status
    df["work_completion_status"] = np.where(
        df["completion_date_raw"].notna() | df["work_id"].isin(completed_df["work_id"]),
        "COMPLETED", "NOT_LISTED_COMPLETED"
    )

    # --- expected_completion_date_proxy = sanction_date + 1 calendar year
    def add_one_year(ts):
        if pd.isna(ts):
            return pd.NaT
        try:
            return ts.replace(year=ts.year + 1)
        except ValueError:
            # Feb 29 -> Feb 28 in non-leap target year
            return ts.replace(month=2, day=28, year=ts.year + 1)

    df["expected_completion_date_proxy_raw"] = df["sanction_date_raw"].map(add_one_year)
    df["expected_completion_date_proxy"] = df["expected_completion_date_proxy_raw"].map(to_iso)
    df["expected_date_source"] = "GUIDELINE_ONE_YEAR_PROXY"

    final_cols = [
        "work_id", "state", "ida", "mp_name", "constituency",
        "work_category", "work_title", "work_description",
        "recommended_date", "sanction_date", "sanction_amount",
        "portal_execution_status",
        "work_completion_status", "expected_completion_date_proxy",
        "expected_date_source", "image_reference", "completion_date",
        "completion_reported_disbursed_amount",
    ]
    return df[final_cols].copy()


def build_quality_report(state, sanctioned_df, completed_df, expenditure_df,
                          load_stats, sanctioned_raw_dates_invalid,
                          completed_raw_dates_invalid,
                          expenditure_raw_dates_invalid,
                          sanctioned_amount_invalid,
                          completed_amount_invalid,
                          expenditure_amount_invalid):
    rows = []

    for s in load_stats:
        rows.append({"check": f"{s['file']}: input rows", "value": s["input_rows"]})
        rows.append({"check": f"{s['file']}: cleaned rows (after removing blank/Grand Total)",
                      "value": s["cleaned_rows"]})
        rows.append({"check": f"{s['file']}: dropped blank/Grand Total rows",
                      "value": s["dropped_blank_or_total_rows"]})

    # Missing work IDs
    missing_sanctioned_wid = sanctioned_df["work_id"].isna().sum() + (sanctioned_df["work_id"] == "").sum()
    missing_completed_wid = completed_df["work_id"].isna().sum() + (completed_df["work_id"] == "").sum()
    missing_expenditure_wid = expenditure_df["work_id"].isna().sum() + (expenditure_df["work_id"] == "").sum()
    rows.append({"check": "Works Sanctioned: rows with missing Work ID", "value": int(missing_sanctioned_wid)})
    rows.append({"check": "Works Completed: rows with missing Work ID", "value": int(missing_completed_wid)})
    rows.append({"check": "Expenditure: rows with missing Work ID", "value": int(missing_expenditure_wid)})

    # Unique work IDs
    rows.append({"check": "Works Sanctioned: unique Work IDs", "value": sanctioned_df["work_id"].nunique()})
    rows.append({"check": "Works Completed: unique Work IDs", "value": completed_df["work_id"].nunique()})
    rows.append({"check": "Expenditure: unique Work IDs", "value": expenditure_df["work_id"].nunique()})

    # Duplicate work IDs in supposedly one-row-per-work files
    dup_sanctioned = sanctioned_df["work_id"].duplicated().sum()
    dup_completed_raw = completed_df["work_id"].duplicated().sum()
    rows.append({"check": "Works Sanctioned: duplicate Work ID rows (should be 0)",
                  "value": int(dup_sanctioned)})
    rows.append({"check": "Works Completed: duplicate Work ID rows (portal may list re-inspections)",
                  "value": int(dup_completed_raw)})

    # Unmatched works between tables
    sanctioned_ids = set(sanctioned_df["work_id"].dropna()) - {""}
    completed_ids = set(completed_df["work_id"].dropna()) - {""}
    expenditure_ids = set(expenditure_df["work_id"].dropna()) - {""}

    completed_not_in_sanctioned = completed_ids - sanctioned_ids
    expenditure_not_in_sanctioned = expenditure_ids - sanctioned_ids
    sanctioned_with_no_expenditure = sanctioned_ids - expenditure_ids

    rows.append({"check": "Work IDs in Works Completed but NOT in Works Sanctioned",
                  "value": len(completed_not_in_sanctioned)})
    rows.append({"check": "Work IDs in Expenditure but NOT in Works Sanctioned",
                  "value": len(expenditure_not_in_sanctioned)})
    rows.append({"check": "Work IDs in Works Sanctioned with NO expenditure rows at all",
                  "value": len(sanctioned_with_no_expenditure)})

    # Invalid dates / amounts
    rows.append({"check": "Works Sanctioned: unparseable Sanction Date values",
                  "value": int(sanctioned_raw_dates_invalid)})
    rows.append({"check": "Works Completed: unparseable Completion Date values",
                  "value": int(completed_raw_dates_invalid)})
    rows.append({"check": "Expenditure: unparseable Expenditure Date values",
                  "value": int(expenditure_raw_dates_invalid)})
    rows.append({"check": "Works Sanctioned: unparseable Sanction Amount values",
                  "value": int(sanctioned_amount_invalid)})
    rows.append({"check": "Works Completed: unparseable Amount Disbursed values",
                  "value": int(completed_amount_invalid)})
    rows.append({"check": "Expenditure: unparseable Fund Disbursed Amount values",
                  "value": int(expenditure_amount_invalid)})

    # Possible duplicate payment events (same work_id + date + vendor + amount).
    # Flagged only, never removed, since there is no true transaction ID.
    dup_key_cols = ["work_id", "expenditure_date", "vendor_name_normalized",
                     "reported_fund_disbursed_amount"]
    possible_dupes = expenditure_df.duplicated(subset=dup_key_cols, keep=False).sum()
    rows.append({"check": "Expenditure: rows sharing identical (Work ID, date, vendor, amount) "
                           "— possible duplicate payment events, NOT removed",
                  "value": int(possible_dupes)})

    report = pd.DataFrame(rows, columns=["check", "value"])
    report.insert(0, "state", state)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Prepare clean MPLADS state-level datasets from raw portal CSV exports."
    )
    parser.add_argument("--state", required=True, help="State name, e.g. 'Chandigarh'")
    parser.add_argument("--sanctioned", required=True, help="Path to Works Sanctioned.csv")
    parser.add_argument("--completed", required=True, help="Path to Works Completed.csv")
    parser.add_argument("--expenditure", required=True,
                         help="Path to Expenditure on Completed and On-going Works as on Date.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write output files into")
    parser.add_argument("--as-of-date", required=True, help="Reference date YYYY-MM-DD for overdue calculations")
    args = parser.parse_args()

    state = args.state.strip()
    as_of_date = args.as_of_date.strip()
    try:
        datetime.strptime(as_of_date, "%Y-%m-%d")
    except ValueError:
        sys.exit(f"ERROR: --as-of-date must be in YYYY-MM-DD format, got: {as_of_date!r}")

    os.makedirs(args.output_dir, exist_ok=True)
    slug = safe_state_slug(state)

    log(f"=== Preparing MPLADS data for state: {state} (as of {as_of_date}) ===\n")

    log("Loading Works Sanctioned...")
    sanctioned_df, sanctioned_stats = load_sanctioned(args.sanctioned, state)

    log("Loading Works Completed...")
    completed_df, completed_stats = load_completed(args.completed, state)

    log("Loading Expenditure on Completed and On-going Works...")
    expenditure_df, expenditure_stats = load_expenditure(args.expenditure, state)

    # Invalid date/amount diagnostics (computed on the cleaned frames)
    sanctioned_dates_invalid = int(sanctioned_df["sanction_date_raw"].isna().sum())
    completed_dates_invalid = int(completed_df["completion_date_raw"].isna().sum())
    expenditure_dates_invalid = int(expenditure_df["expenditure_date_raw"].isna().sum())

    sanctioned_amount_invalid = int(sanctioned_df["sanction_amount"].isna().sum())
    completed_amount_invalid = int(completed_df["completion_reported_disbursed_amount"].isna().sum())
    expenditure_amount_invalid = int(expenditure_df["reported_fund_disbursed_amount"].isna().sum())

    log("\nBuilding payment_events table...")
    payment_events = build_payment_events(expenditure_df)

    log("Building projects table (sanctioned + completed)...")
    projects = build_projects(sanctioned_df, completed_df, expenditure_df)

    log("Building data quality report...\n")
    quality_report = build_quality_report(
        state, sanctioned_df, completed_df, expenditure_df,
        [sanctioned_stats, completed_stats, expenditure_stats],
        sanctioned_dates_invalid, completed_dates_invalid, expenditure_dates_invalid,
        sanctioned_amount_invalid, completed_amount_invalid, expenditure_amount_invalid,
    )

    # --- Print concise console summary ---
    log("--- Quality summary ---")
    log(f"Works Sanctioned : {sanctioned_stats['input_rows']} input rows -> "
        f"{sanctioned_stats['cleaned_rows']} cleaned rows "
        f"({sanctioned_df['work_id'].nunique()} unique Work IDs, "
        f"{int(sanctioned_df['work_id'].duplicated().sum())} duplicate Work IDs)")
    log(f"Works Completed  : {completed_stats['input_rows']} input rows -> "
        f"{completed_stats['cleaned_rows']} cleaned rows "
        f"({completed_df['work_id'].nunique()} unique Work IDs)")
    log(f"Expenditure      : {expenditure_stats['input_rows']} input rows -> "
        f"{expenditure_stats['cleaned_rows']} cleaned rows "
        f"({expenditure_df['work_id'].nunique()} unique Work IDs)")

    sanctioned_ids = set(sanctioned_df["work_id"].dropna()) - {""}
    completed_ids = set(completed_df["work_id"].dropna()) - {""}
    expenditure_ids = set(expenditure_df["work_id"].dropna()) - {""}
    log(f"Work IDs in Completed but not in Sanctioned : {len(completed_ids - sanctioned_ids)}")
    log(f"Work IDs in Expenditure but not in Sanctioned : {len(expenditure_ids - sanctioned_ids)}")
    log(f"Sanctioned Work IDs with no expenditure rows : {len(sanctioned_ids - expenditure_ids)}")
    log("")

    # --- Write outputs ---
    payment_events_path = os.path.join(args.output_dir, f"{slug}_payment_events.csv")
    projects_path = os.path.join(args.output_dir, f"{slug}_projects.csv")
    quality_report_path = os.path.join(args.output_dir, f"{slug}_data_quality_report.csv")

    payment_events.to_csv(payment_events_path, index=False)
    projects.to_csv(projects_path, index=False)
    quality_report.to_csv(quality_report_path, index=False)

    log("=== Files created ===")
    log(f"{payment_events_path}  ({len(payment_events)} rows)")
    log(f"{projects_path}  ({len(projects)} rows)")
    log(f"{quality_report_path}  ({len(quality_report)} rows)")


if __name__ == "__main__":
    main()
