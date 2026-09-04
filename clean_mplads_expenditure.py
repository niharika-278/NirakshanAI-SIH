#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPLADS Expenditure CSV -> Clean Payment-Events CSV
====================================================

Converts a raw "Expenditure on Completed and On-going Works as on Date.csv"
export into a clean, source-preserving payment-events CSV.

Design principle (see prompt spec): PRESERVE the source data unless a row is
explicitly required to be removed by the specification. No aggregation, no
deduplication of payment events, no invented values.

Windows-compatible. Reusable for any state file with the same raw schema.

Usage:
    python clean_mplads_expenditure.py <input_csv> [output_dir]

If no arguments are given, INPUT_CSV / OUTPUT_DIR below are used.
"""

import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. INPUT / OUTPUT CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_CSV = "Expenditure on Completed and On-going Works as on Date.csv"
OUTPUT_DIR = "."

# Required raw -> clean column mapping (exact source names, per spec).
COLUMN_RENAME_MAP = {
    "Sr. No.": "source_serial_number",
    "Work ID": "work_id",
    "Work": "work_title",
    "State": "state",
    "IDA": "ida",
    "Hon'ble Members of Parliament": "mp_name",
    "Constituency": "constituency",
    "Expenditure Date": "expenditure_date",
    "Vendor Name": "vendor_name",
    "Payment Status": "payment_status",
    "Fund Disbursed Amount (\u20b9)": "reported_fund_disbursed_amount",
}

FINAL_COLUMN_ORDER = [
    "source_serial_number",
    "work_id",
    "work_title",
    "state",
    "ida",
    "mp_name",
    "constituency",
    "expenditure_date",
    "vendor_name_normalized",
    "payment_status",
    "reported_fund_disbursed_amount",
]

WORK_ID_PATTERN = re.compile(r"^WS/MP\d+/\d{4}-\d{4}/\d+$")

# Candidate explicit date formats to auto-detect from the sample data.
# Kept explicit (not a fuzzy/ambiguous parser) per spec section 6.
CANDIDATE_DATE_FORMATS = [
    "%d-%b-%y",   # 27-Aug-26
    "%d-%b-%Y",   # 27-Aug-2026
    "%d-%m-%Y",   # 27-08-2026
    "%d/%m/%Y",   # 27/08/2026
    "%Y-%m-%d",   # 2026-08-27
    "%d-%B-%y",   # 27-August-26
    "%d-%B-%Y",   # 27-August-2026
]

INTERNAL_ROW_COL = "_physical_csv_row"


# ---------------------------------------------------------------------------
# Helper: normalize a header string for tolerant (non-fuzzy) matching
# ---------------------------------------------------------------------------
def _normalize_header(h):
    """Collapse all whitespace and normalize unicode so that harmless
    spacing differences (e.g. 'Fund Disbursed Amount ( \u20b9 )' vs
    'Fund Disbursed Amount (\u20b9)') match, without fuzzy-mapping unrelated
    column names."""
    if h is None:
        return ""
    h = unicodedata.normalize("NFKC", str(h))
    h = h.replace("\xa0", " ")
    h = re.sub(r"\s+", "", h)  # remove ALL whitespace for comparison
    return h.strip().lower()


# ---------------------------------------------------------------------------
# 2. LOAD CSV
# ---------------------------------------------------------------------------
def load_csv(path):
    """Load the raw CSV as strings (no dtype guessing), tagging each row
    with its original physical CSV row number (header = row 1, first data
    row = row 2)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    # utf-8-sig strips a BOM if present; the sample file has one.
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=True)

    # Physical CSV row number: index 0 -> CSV row 2 (one header row).
    df[INTERNAL_ROW_COL] = df.index + 2

    print(f"[LOAD] Read {len(df)} data rows from: {path}")
    return df


# ---------------------------------------------------------------------------
# 3. VALIDATE + RENAME COLUMNS
# ---------------------------------------------------------------------------
def validate_and_rename_columns(df):
    """Match raw headers to the required schema tolerating only harmless
    whitespace differences, then rename. Raises a clear error if a required
    column is missing. Drops any extra/unnecessary columns."""
    normalized_to_actual = {_normalize_header(c): c for c in df.columns if c != INTERNAL_ROW_COL}

    rename_actual = {}
    missing = []
    for raw_expected, new_name in COLUMN_RENAME_MAP.items():
        key = _normalize_header(raw_expected)
        actual_col = normalized_to_actual.get(key)
        if actual_col is None:
            missing.append(raw_expected)
        else:
            rename_actual[actual_col] = new_name

    if missing:
        raise ValueError(
            "Required column(s) not found in input CSV (checked with "
            "whitespace-tolerant matching only): "
            + ", ".join(missing)
            + f"\nColumns found in file: {list(df.columns)}"
        )

    df = df.rename(columns=rename_actual)

    # Keep only the required (renamed) columns + internal row tracker.
    keep_cols = list(COLUMN_RENAME_MAP.values()) + [INTERNAL_ROW_COL]
    dropped = [c for c in df.columns if c not in keep_cols]
    if dropped:
        print(f"[COLUMNS] Dropping extra/unnecessary columns: {dropped}")
    df = df[keep_cols]

    print(f"[COLUMNS] All {len(COLUMN_RENAME_MAP)} required columns found and renamed.")
    return df


# ---------------------------------------------------------------------------
# 4. REMOVE GRAND TOTAL ROW(S)
# ---------------------------------------------------------------------------
def remove_grand_total_rows(df):
    """Robustly identify Grand Total row(s) by inspecting row content
    (not a fixed row number). A row is treated as a Grand Total row only
    if one of the key identifying fields (source_serial_number, work_title,
    state, work_id) is an EXACT case-insensitive match for 'grand total'
    after stripping whitespace -- never a substring search inside ordinary
    text, so a legitimate row that merely mentions 'grand total' inside a
    sentence is never removed."""

    def is_exact_grand_total(val):
        if pd.isna(val):
            return False
        return str(val).strip().casefold() == "grand total"

    mask = (
        df["source_serial_number"].apply(is_exact_grand_total)
        | df["work_title"].apply(is_exact_grand_total)
        | df["state"].apply(is_exact_grand_total)
        | df["work_id"].apply(is_exact_grand_total)
    )

    removed_rows = df.loc[mask, INTERNAL_ROW_COL].tolist()
    n_removed = int(mask.sum())

    print(f"[GRAND TOTAL] Found {n_removed} Grand Total row(s).")
    if removed_rows:
        print(f"[GRAND TOTAL] Original CSV row number(s): {removed_rows}")

    df_clean = df.loc[~mask].copy()
    return df_clean, n_removed, removed_rows


# ---------------------------------------------------------------------------
# 5. WORK ID CLEANING + VALIDATION
# ---------------------------------------------------------------------------
def clean_work_id(raw):
    """Remove accidental/harmless whitespace (spaces, tabs, newlines,
    non-breaking spaces) from a Work ID without altering any non-whitespace
    character. Valid Work IDs never legitimately contain internal
    whitespace, so stripping all whitespace characters is safe and does not
    risk corrupting a genuine value."""
    if pd.isna(raw):
        return raw
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", "", s)  # remove all whitespace, wherever it occurs
    return s


def validate_work_id(cleaned):
    if pd.isna(cleaned) or cleaned == "":
        return False
    return bool(WORK_ID_PATTERN.match(cleaned))


def clean_and_validate_work_ids(df):
    df = df.copy()
    df["work_id"] = df["work_id"].apply(clean_work_id)
    valid_mask = df["work_id"].apply(validate_work_id)

    invalid_rows = df.loc[~valid_mask]
    n_invalid = len(invalid_rows)
    invalid_values = invalid_rows["work_id"].tolist()
    invalid_row_numbers = invalid_rows[INTERNAL_ROW_COL].tolist()

    print(f"[WORK ID] Invalid Work IDs found: {n_invalid}")
    if n_invalid:
        preview = list(zip(invalid_row_numbers, invalid_values))[:20]
        print(f"[WORK ID] Sample invalid (csv_row, value): {preview}")

    return df, valid_mask, n_invalid, invalid_values, invalid_row_numbers


# ---------------------------------------------------------------------------
# 6. VENDOR NAME NORMALIZATION
# ---------------------------------------------------------------------------
def normalize_vendor_name(raw):
    if pd.isna(raw):
        return raw
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.replace("\xa0", " ")
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)  # collapse repeated internal whitespace
    s = s.strip()
    return s


# ---------------------------------------------------------------------------
# 7. EXPENDITURE DATE CONVERSION + VALIDATION
# ---------------------------------------------------------------------------
def _detect_date_format(sample_values):
    """Pick the explicit strptime format (from CANDIDATE_DATE_FORMATS) that
    successfully parses the largest number of non-blank sample values. This
    avoids ambiguous, per-row automatic date guessing while remaining
    reusable across different state files."""
    best_fmt, best_count = None, -1
    for fmt in CANDIDATE_DATE_FORMATS:
        count = 0
        for v in sample_values:
            try:
                pd.to_datetime(v, format=fmt)
                count += 1
            except (ValueError, TypeError):
                pass
        if count > best_count:
            best_count = count
            best_fmt = fmt
    return best_fmt, best_count


def convert_and_validate_dates(df):
    df = df.copy()
    raw_dates = df["expenditure_date"]

    non_blank = raw_dates.dropna()
    non_blank = non_blank[non_blank.astype(str).str.strip() != ""]
    sample = non_blank.astype(str).str.strip().unique()[:500]

    detected_fmt, matched = _detect_date_format(sample)
    print(f"[DATE] Detected date format from sample: {detected_fmt} "
          f"(matched {matched}/{len(sample)} sampled unique values)")

    def parse_one(v):
        if pd.isna(v):
            return (None, "missing")
        s = str(v).strip()
        if s == "":
            return (None, "missing")
        if detected_fmt is not None:
            try:
                dt = pd.to_datetime(s, format=detected_fmt)
                return (dt.strftime("%Y-%m-%d"), "valid")
            except (ValueError, TypeError):
                return (None, "invalid")
        return (None, "invalid")

    parsed = raw_dates.apply(parse_one)
    df["expenditure_date"] = parsed.apply(lambda t: t[0])
    status = parsed.apply(lambda t: t[1])

    n_valid = int((status == "valid").sum())
    n_missing = int((status == "missing").sum())
    n_invalid = int((status == "invalid").sum())

    invalid_mask = status.isin(["missing", "invalid"])
    invalid_rows = df.loc[invalid_mask]
    invalid_values = raw_dates.loc[invalid_mask].tolist()
    invalid_row_numbers = invalid_rows[INTERNAL_ROW_COL].tolist()

    print(f"[DATE] Valid: {n_valid} | Missing: {n_missing} | Invalid: {n_invalid}")
    if n_invalid or n_missing:
        preview = list(zip(invalid_row_numbers, invalid_values))[:20]
        print(f"[DATE] Sample missing/invalid (csv_row, value): {preview}")

    valid_mask = ~invalid_mask
    return df, valid_mask, n_valid, n_missing, n_invalid, invalid_values, invalid_row_numbers


# ---------------------------------------------------------------------------
# 8. AMOUNT CONVERSION + VALIDATION
# ---------------------------------------------------------------------------
def _clean_amount_string(raw):
    """Strip currency symbols, thousands separators and whitespace from a
    monetary string, returning a string suitable for float() conversion, or
    None if the value is blank."""
    if pd.isna(raw):
        return None
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.replace("\xa0", " ").strip()
    if s == "":
        return None
    # remove currency symbol(s) and any stray whitespace
    s = s.replace("\u20b9", "")  # ₹
    s = s.replace(" ", "")
    # remove thousands-separator commas (Indian or standard grouping)
    s = s.replace(",", "")
    return s


def convert_and_validate_amounts(df):
    df = df.copy()
    raw_amounts = df["reported_fund_disbursed_amount"]

    def parse_one(v):
        cleaned = _clean_amount_string(v)
        if cleaned is None:
            return (np.nan, "missing")
        try:
            f = float(cleaned)
        except ValueError:
            return (np.nan, "invalid")
        if not np.isfinite(f):
            return (np.nan, "invalid")
        return (f, "valid")

    parsed = raw_amounts.apply(parse_one)
    df["reported_fund_disbursed_amount"] = parsed.apply(lambda t: t[0]).astype("float64")
    status = parsed.apply(lambda t: t[1])

    n_valid = int((status == "valid").sum())
    n_missing = int((status == "missing").sum())
    n_invalid = int((status == "invalid").sum())

    invalid_mask = status.isin(["missing", "invalid"])
    invalid_rows = df.loc[invalid_mask]
    invalid_values = raw_amounts.loc[invalid_mask].tolist()
    invalid_row_numbers = invalid_rows[INTERNAL_ROW_COL].tolist()

    print(f"[AMOUNT] Valid: {n_valid} | Missing: {n_missing} | Invalid: {n_invalid}")
    print(f"[AMOUNT] dtype after conversion: {df['reported_fund_disbursed_amount'].dtype}")
    if n_invalid or n_missing:
        preview = list(zip(invalid_row_numbers, invalid_values))[:20]
        print(f"[AMOUNT] Sample missing/invalid (csv_row, value): {preview}")

    valid_mask = ~invalid_mask
    return df, valid_mask, n_valid, n_missing, n_invalid, invalid_values, invalid_row_numbers


# ---------------------------------------------------------------------------
# 9. PAYMENT STATUS NORMALIZATION
# ---------------------------------------------------------------------------
def normalize_payment_status(raw):
    if pd.isna(raw):
        return raw
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.replace("\xa0", " ")
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


# ---------------------------------------------------------------------------
# 10/11. REPEATED source_serial_number DETECTION + REMOVAL
# ---------------------------------------------------------------------------
def detect_repeated_source_serials(df):
    counts = df["source_serial_number"].value_counts()
    repeated_values = counts[counts > 1].index.tolist()

    detail = {}
    for val in repeated_values:
        rows = df.loc[df["source_serial_number"] == val, INTERNAL_ROW_COL].tolist()
        detail[val] = rows

    return repeated_values, detail


def remove_later_duplicate_serials(df):
    """Keep only the FIRST occurrence (in original source/file order) of
    each source_serial_number; later occurrences are removed. The retained
    value is never renumbered."""
    df_sorted = df.sort_values(INTERNAL_ROW_COL, kind="stable")
    is_first = ~df_sorted["source_serial_number"].duplicated(keep="first")

    removed_rows = df_sorted.loc[~is_first, INTERNAL_ROW_COL].tolist()
    removed_values = df_sorted.loc[~is_first, "source_serial_number"].tolist()

    df_final = df_sorted.loc[is_first].copy()
    return df_final, len(removed_rows), removed_rows, removed_values


# ---------------------------------------------------------------------------
# Payment-event "identical rows" analysis (informational only -- NOT removed)
# ---------------------------------------------------------------------------
def analyze_identical_payment_events(df):
    compare_cols = [
        "work_id", "work_title", "state", "ida", "mp_name", "constituency",
        "expenditure_date", "vendor_name_normalized", "payment_status",
        "reported_fund_disbursed_amount",
    ]
    dup_mask = df.duplicated(subset=compare_cols, keep=False)
    n_rows_in_groups = int(dup_mask.sum())

    groups = df.loc[dup_mask].groupby(compare_cols, dropna=False)
    n_groups = groups.ngroups
    affected_work_ids = sorted(set(df.loc[dup_mask, "work_id"].tolist()))

    return n_groups, n_rows_in_groups, affected_work_ids


def report_work_id_occurrence_histogram(df):
    counts = df["work_id"].value_counts()
    hist = Counter(counts.values.tolist())
    print("[WORK ID] Occurrence histogram (times occurring -> number of work_ids):")
    for n_times in sorted(hist.keys()):
        print(f"    Work IDs occurring {n_times} time(s): {hist[n_times]}")
    n_repeated = int((counts > 1).sum())
    max_occurrence = int(counts.max()) if len(counts) else 0
    return n_repeated, max_occurrence


# ---------------------------------------------------------------------------
# Determine output state name generically
# ---------------------------------------------------------------------------
def determine_state_name(df, input_path):
    non_blank_states = df["state"].dropna().astype(str).str.strip()
    non_blank_states = non_blank_states[non_blank_states != ""]
    if len(non_blank_states) > 0:
        state_name = non_blank_states.mode().iloc[0]
    else:
        # fall back to input filename
        base = os.path.splitext(os.path.basename(input_path))[0]
        state_name = base

    safe = re.sub(r"[^A-Za-z0-9]+", "_", state_name).strip("_")
    return safe if safe else "output"


# ---------------------------------------------------------------------------
# SAVE + VERIFY
# ---------------------------------------------------------------------------
def save_final_csv(df, output_path):
    out_df = df[FINAL_COLUMN_ORDER].copy()
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"[SAVE] Wrote {len(out_df)} rows to: {output_path}")
    return out_df


def verify_saved_csv(output_path, expected_len):
    """Re-read the SAVED file from disk and verify it independently
    satisfies the structural requirements (not just the in-memory frame)."""
    check = pd.read_csv(output_path, dtype=str, encoding="utf-8-sig")

    errors = []

    if list(check.columns) != FINAL_COLUMN_ORDER:
        errors.append(f"Column order mismatch. Found: {list(check.columns)}")

    if "vendor_name" in check.columns:
        errors.append("'vendor_name' column present in saved file (must be absent).")

    if len(check) != expected_len:
        errors.append(f"Row count mismatch: expected {expected_len}, found {len(check)}.")

    # work_id validity
    bad_wid = check["work_id"].apply(lambda v: not validate_work_id(v))
    if bad_wid.any():
        errors.append(f"{int(bad_wid.sum())} invalid work_id value(s) present in saved file.")

    # date validity (must be YYYY-MM-DD, non-blank, real calendar date)
    def is_iso_date(v):
        if pd.isna(v):
            return False
        try:
            pd.to_datetime(str(v), format="%Y-%m-%d")
            return True
        except (ValueError, TypeError):
            return False
    bad_date = ~check["expenditure_date"].apply(is_iso_date)
    if bad_date.any():
        errors.append(f"{int(bad_date.sum())} invalid/missing expenditure_date value(s) in saved file.")

    # amount validity (must parse as float, non-blank)
    def is_valid_amount(v):
        if pd.isna(v):
            return False
        try:
            float(v)
            return True
        except (ValueError, TypeError):
            return False
    bad_amt = ~check["reported_fund_disbursed_amount"].apply(is_valid_amount)
    if bad_amt.any():
        errors.append(f"{int(bad_amt.sum())} invalid/missing reported_fund_disbursed_amount value(s) in saved file.")

    # source_serial_number duplicate check
    dup_serial = check["source_serial_number"].duplicated(keep=False)
    if dup_serial.any():
        errors.append(f"{int(dup_serial.sum())} duplicate source_serial_number value(s) remain in saved file.")

    if errors:
        raise AssertionError(
            "SAVED CSV FAILED VERIFICATION:\n  - " + "\n  - ".join(errors)
        )

    print("[VERIFY] Saved CSV passed all structural verification checks.")
    return check


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main(input_path, output_dir):
    print("=" * 78)
    print("MPLADS EXPENDITURE -> PAYMENT-EVENTS CLEANING PIPELINE")
    print("=" * 78)

    # --- Load ---
    raw_df = load_csv(input_path)
    n_input_rows = len(raw_df)

    # --- Validate + rename columns ---
    df = validate_and_rename_columns(raw_df)

    # --- Remove Grand Total row(s) ---
    df, n_grand_total_removed, grand_total_rows = remove_grand_total_rows(df)
    n_after_grand_total = len(df)

    # --- Clean + validate Work IDs ---
    df, wid_valid_mask, n_invalid_wid, invalid_wid_values, invalid_wid_rows = \
        clean_and_validate_work_ids(df)
    invalid_wid_row_set = set(invalid_wid_rows)

    # --- Normalize vendor name (before dropping raw vendor_name) ---
    df["vendor_name_normalized"] = df["vendor_name"].apply(normalize_vendor_name)
    df = df.drop(columns=["vendor_name"])

    # --- Convert + validate dates ---
    df, date_valid_mask, n_valid_dates, n_missing_dates, n_invalid_dates, \
        invalid_date_values, invalid_date_rows = convert_and_validate_dates(df)
    invalid_date_row_set = set(invalid_date_rows)

    # --- Convert + validate amounts ---
    df, amount_valid_mask, n_valid_amounts, n_missing_amounts, n_invalid_amounts, \
        invalid_amount_values, invalid_amount_rows = convert_and_validate_amounts(df)
    invalid_amount_row_set = set(invalid_amount_rows)

    # --- Normalize payment status ---
    df["payment_status"] = df["payment_status"].apply(normalize_payment_status)

    # --- Apply combined validity mask (Work ID AND date AND amount) ---
    combined_valid_mask = wid_valid_mask & date_valid_mask & amount_valid_mask
    df_valid = df.loc[combined_valid_mask].copy()

    unique_date_amount_removed_rows = (invalid_date_row_set | invalid_amount_row_set) - invalid_wid_row_set
    n_unique_date_amount_removed = len(unique_date_amount_removed_rows)

    # --- Detect repeated source_serial_number (before dedup, informational) ---
    repeated_serials, repeated_serial_detail = detect_repeated_source_serials(df_valid)

    # --- Remove later duplicate source_serial_number occurrences ---
    df_final, n_dup_serial_removed, dup_serial_removed_rows, dup_serial_removed_values = \
        remove_later_duplicate_serials(df_valid)

    n_final = len(df_final)

    # --- Informational: identical payment-event row analysis (NOT removed) ---
    n_identical_groups, n_identical_rows, affected_work_ids = \
        analyze_identical_payment_events(df_final)

    # --- Work ID occurrence histogram ---
    n_repeated_wid, max_wid_occurrence = report_work_id_occurrence_histogram(df_final)

    # --- Determine output filename ---
    state_name = determine_state_name(df_final, input_path)
    output_filename = f"{state_name}_payment_events.csv"
    output_path = os.path.join(output_dir, output_filename)

    # --- Save ---
    saved_df = save_final_csv(df_final, output_path)

    # --- Verify saved file independently ---
    verify_saved_csv(output_path, n_final)

    # =======================================================================
    # VALIDATION / RECONCILIATION REPORT
    # =======================================================================
    print()
    print("=" * 78)
    print("VALIDATION REPORT")
    print("=" * 78)

    print(f"Input filename: {input_path}")
    print(f"Rows before cleaning: {n_input_rows}")
    print(f"Grand Total rows removed: {n_grand_total_removed}")
    print(f"Rows remaining after Grand Total removal: {n_after_grand_total}")
    print(f"Invalid Work ID rows removed: {n_invalid_wid}")
    print(f"  Missing dates: {n_missing_dates}")
    print(f"  Invalid dates: {n_invalid_dates}")
    print(f"  Missing amounts: {n_missing_amounts}")
    print(f"  Invalid amounts: {n_invalid_amounts}")
    print(f"Unique rows removed due to missing/invalid date OR amount "
          f"(excluding rows already removed for invalid Work ID): {n_unique_date_amount_removed}")
    print(f"Repeated source_serial_number rows removed: {n_dup_serial_removed}")
    print(f"Final number of rows: {n_final}")

    print()
    print("--- Reconciliation ---")
    print(f"  {n_input_rows} (input rows)")
    print(f"  - {n_grand_total_removed} (Grand Total rows removed)")
    print(f"  - {n_invalid_wid} (invalid Work ID rows removed)")
    print(f"  - {n_unique_date_amount_removed} (unique rows removed: missing/invalid date or amount)")
    print(f"  - {n_dup_serial_removed} (duplicate source_serial_number rows removed)")
    reconciled = (n_input_rows - n_grand_total_removed - n_invalid_wid
                  - n_unique_date_amount_removed - n_dup_serial_removed)
    print(f"  = {reconciled} (computed)")
    print(f"    vs. actual final rows = {n_final}  "
          f"{'MATCH' if reconciled == n_final else 'MISMATCH -- INVESTIGATE'}")
    if reconciled != n_final:
        raise AssertionError("Row-count reconciliation failed -- pipeline logic error.")

    print()
    print("--- Work ID validation ---")
    print(f"Unique work_id values in final output: {df_final['work_id'].nunique()}")
    print(f"Work IDs occurring more than once: {n_repeated_wid}")
    print(f"Maximum occurrences of any single Work ID: {max_wid_occurrence}")
    print(f"Invalid Work IDs removed: {n_invalid_wid}")
    if n_invalid_wid:
        preview = list(zip(invalid_wid_rows, invalid_wid_values))[:20]
        print(f"Invalid Work ID sample (csv_row, value): {preview}")

    print()
    print("--- Date validation ---")
    print(f"Valid dates: {n_valid_dates}")
    print(f"Missing dates: {n_missing_dates}")
    print(f"Invalid/non-parsable dates: {n_invalid_dates}")
    if invalid_date_rows:
        preview = list(zip(invalid_date_rows, invalid_date_values))[:20]
        print(f"Invalid date sample (csv_row, value): {preview}")

    print()
    print("--- Amount validation ---")
    print(f"Valid amounts converted: {n_valid_amounts}")
    print(f"Missing/blank amounts: {n_missing_amounts}")
    print(f"Invalid/unparseable amounts: {n_invalid_amounts}")
    if invalid_amount_rows:
        preview = list(zip(invalid_amount_rows, invalid_amount_values))[:20]
        print(f"Invalid amount sample (csv_row, value): {preview}")
    print(f"Final dtype of reported_fund_disbursed_amount: {df_final['reported_fund_disbursed_amount'].dtype}")

    print()
    print("--- Missing values in final output (key columns) ---")
    for col in ["work_id", "expenditure_date", "reported_fund_disbursed_amount"]:
        n_missing_final = df_final[col].isna().sum()
        print(f"  {col}: {n_missing_final} missing")
    print(f"  'vendor_name' present in final columns: {'vendor_name' in df_final.columns}")

    print()
    print("--- Source serial number validation ---")
    print(f"Unique source_serial_number values (original, after Grand Total removal): "
          f"{df_valid['source_serial_number'].nunique() + len(dup_serial_removed_values)}")
    print(f"Repeated source_serial_number values found: {len(repeated_serials)}")
    if repeated_serials:
        preview = {k: repeated_serial_detail[k] for k in list(repeated_serial_detail)[:20]}
        print(f"Repeated values -> original csv rows (sample): {preview}")
    print(f"Duplicate source-serial rows removed: {n_dup_serial_removed}")
    print("Confirmation: only the first occurrence (by original file order) of each "
          "repeated source_serial_number was retained.")

    print()
    print("--- Payment-event duplicate analysis (informational; NOT removed) ---")
    print(f"Exact-identical payment-event groups found: {n_identical_groups}")
    print(f"Rows belonging to such groups: {n_identical_rows}")
    print(f"Work IDs affected (sample, up to 20): {affected_work_ids[:20]}")
    print("Confirmation: identical payment-event rows were RETAINED because their "
          "source_serial_number values differ.")
    print(f"Work IDs occurring more than once: {n_repeated_wid} "
          f"(maximum occurrence: {max_wid_occurrence}) -- all such repeats were RETAINED.")

    print()
    print(f"Output file: {output_path}")
    print("=" * 78)
    print("DONE.")
    print("=" * 78)

    return output_path


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    main(in_path, out_dir)
