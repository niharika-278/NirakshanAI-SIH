#!/usr/bin/env python3
"""
combine_all_states.py

Combines already-cleaned, state-wise MPLADS CSV files (produced by
prepare_state_data.py) into two national-level datasets for the ML team:

    all_states_projects.csv         (one row per sanctioned work/project)
    all_states_payment_events.csv   (one row per payment event)

This script does NOT clean, transform, aggregate, or feature-engineer
anything. Its only responsibilities are:

    read state-wise cleaned files
            |
    validate schemas
            |
    vertically concatenate same-type datasets (pd.concat, never pd.merge)
            |
    validate final combined datasets
            |
    save two national CSVs

Usage:
    python combine_all_states.py \
        --input-dir "states_edited_data" \
        --output-dir "states_edited_data/all_states_output"

The script automatically discovers state folders/files under --input-dir.
It looks for files matching:
    *_projects.csv
    *_payment_events.csv
(and, for informational purposes only, *_data_quality_report.csv)

It never reads its own output files back in, even if --output-dir is
nested inside --input-dir.
"""

import argparse
import os
import re
import sys

import pandas as pd


# --------------------------------------------------------------------------
# Expected schemas (exact columns, in this exact order in the final files)
# --------------------------------------------------------------------------

PROJECT_COLUMNS = [
    "work_id",
    "state",
    "ida",
    "mp_name",
    "constituency",
    "work_category",
    "work_title",
    "work_description",
    "recommended_date",
    "sanction_date",
    "sanction_amount",
    "portal_execution_status",
    "work_completion_status",
    "expected_completion_date_proxy",
    "expected_date_source",
    "image_reference",
    "completion_date",
    "completion_reported_disbursed_amount",
]

PAYMENT_COLUMNS = [
    "work_id",
    "state",
    "ida",
    "mp_name",
    "constituency",
    "expenditure_date",
    "vendor_name_raw",
    "vendor_name_normalized",
    "payment_status",
    "reported_fund_disbursed_amount",
    "source_serial_number",
]

PROJECTS_SUFFIX = "_projects.csv"
PAYMENTS_SUFFIX = "_payment_events.csv"
QUALITY_SUFFIX = "_data_quality_report.csv"

# Defensive: never re-ingest our own combined outputs by these exact names,
# no matter where they are found.
OWN_OUTPUT_FILENAMES = {"all_states_projects.csv", "all_states_payment_events.csv"}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def log(msg=""):
    print(msg)


def display_state_name(slug):
    """Turn a filename-derived slug like 'andhra_pradesh' into 'Andhra Pradesh'."""
    return slug.replace("_", " ").strip().title()


def read_csv_safely(path):
    """
    Read a CSV trying a few common encodings. Returns (df, error_message).
    On success error_message is None. On failure df is None.
    Handles the empty-file case explicitly (zero bytes / no header).
    """
    if os.path.getsize(path) == 0:
        return None, "file is empty (0 bytes)"

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, dtype=None, encoding=enc, keep_default_na=True)
            return df, None
        except pd.errors.EmptyDataError:
            return None, "file has no columns/header (empty data)"
        except UnicodeDecodeError as e:
            last_err = f"encoding error with {enc}: {e}"
            continue
        except Exception as e:
            return None, f"failed to read file: {e}"
    return None, last_err or "unknown read error"


def is_excluded_path(path, excluded_root):
    """True if `path` is inside `excluded_root` (or equal to it)."""
    path = os.path.abspath(path)
    excluded_root = os.path.abspath(excluded_root)
    try:
        common = os.path.commonpath([path, excluded_root])
    except ValueError:
        # On different drives (Windows) etc. - treat as not excluded.
        return False
    return common == excluded_root


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_files(input_dir, output_dir):
    """
    Walk input_dir (skipping anything inside output_dir) and group files
    by type (projects / payment_events / quality_report) and by state slug
    parsed from the filename.

    Returns three dicts: projects_files, payments_files, quality_files
    each mapping state_slug -> file path. Also returns a list of warning
    strings for conflicts (e.g. the same state/type found more than once).
    """
    projects_files = {}
    payments_files = {}
    quality_files = {}
    warnings = []

    for root, dirs, files in os.walk(input_dir):
        if is_excluded_path(root, output_dir):
            dirs[:] = []  # prune traversal into the output directory
            continue
        # Prevent descending into the output directory if it's a subfolder
        dirs[:] = [d for d in dirs if not is_excluded_path(os.path.join(root, d), output_dir)]

        for fname in sorted(files):
            if not fname.lower().endswith(".csv"):
                continue
            if fname in OWN_OUTPUT_FILENAMES:
                continue

            fpath = os.path.join(root, fname)

            if fname.endswith(PROJECTS_SUFFIX):
                slug = fname[: -len(PROJECTS_SUFFIX)].strip().lower()
                if not slug:
                    continue
                if slug in projects_files:
                    warnings.append(
                        f"Multiple '*_projects.csv' files found for state '{slug}'. "
                        f"Using '{projects_files[slug]}', ignoring '{fpath}'."
                    )
                else:
                    projects_files[slug] = fpath

            elif fname.endswith(PAYMENTS_SUFFIX):
                slug = fname[: -len(PAYMENTS_SUFFIX)].strip().lower()
                if not slug:
                    continue
                if slug in payments_files:
                    warnings.append(
                        f"Multiple '*_payment_events.csv' files found for state '{slug}'. "
                        f"Using '{payments_files[slug]}', ignoring '{fpath}'."
                    )
                else:
                    payments_files[slug] = fpath

            elif fname.endswith(QUALITY_SUFFIX):
                slug = fname[: -len(QUALITY_SUFFIX)].strip().lower()
                if not slug:
                    continue
                quality_files[slug] = fpath

            # Any other CSV under input_dir is intentionally ignored - it
            # doesn't match one of the three expected file patterns.

    return projects_files, payments_files, quality_files, warnings


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------

def validate_and_reorder(df, expected_columns, file_label, issues):
    """
    Check that df has exactly the expected columns (order-insensitive).
    If valid, return a copy reordered to match expected_columns.
    If invalid, append a description to `issues` and return None (the
    file is excluded from concatenation - it is never silently patched).
    """
    actual = list(df.columns)
    missing = [c for c in expected_columns if c not in actual]
    extra = [c for c in actual if c not in expected_columns]

    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing columns {missing}")
        if extra:
            parts.append(f"unexpected columns {extra}")
        issues.append(f"{file_label}: " + "; ".join(parts) + " -> EXCLUDED from combination")
        return None

    if actual != expected_columns:
        issues.append(f"{file_label}: columns present but in different order -> reordered to expected schema")

    return df[expected_columns].copy()


# --------------------------------------------------------------------------
# Quality report inspection (informational only - never written to output)
# --------------------------------------------------------------------------

def inspect_quality_reports(quality_files):
    """
    Lightly inspect each state's data-quality report (if present) and
    surface any checks that flagged a non-zero count, so problems don't
    get lost. This is purely informational - nothing here is written to
    the combined output files.
    """
    if not quality_files:
        log("No *_data_quality_report.csv files found (skipped inspection).")
        return

    flagged_keywords = ("missing", "duplicate", "unmatched", "invalid", "unparseable")
    any_flagged = False

    for slug in sorted(quality_files):
        path = quality_files[slug]
        df, err = read_csv_safely(path)
        if err:
            log(f"  {display_state_name(slug)}: could not read quality report ({err})")
            continue
        if df is None or "check" not in df.columns or "value" not in df.columns:
            continue

        try:
            flagged = df[
                df["check"].astype(str).str.lower().str.contains("|".join(flagged_keywords))
                & pd.to_numeric(df["value"], errors="coerce").fillna(0).gt(0)
            ]
        except Exception:
            continue

        if len(flagged) > 0:
            any_flagged = True
            log(f"  {display_state_name(slug)}: {len(flagged)} quality-report item(s) with non-zero counts")

    if not any_flagged:
        log("  No non-zero data-quality flags found across available reports.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combine state-wise cleaned MPLADS CSVs into national-level datasets."
    )
    parser.add_argument("--input-dir", required=True,
                         help="Root directory containing state output folders/files")
    parser.add_argument("--output-dir", required=True,
                         help="Directory to write all_states_projects.csv and all_states_payment_events.csv")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        sys.exit(f"ERROR: --input-dir does not exist or is not a directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    log("=" * 50)
    log("NIRAKSHANAI — ALL STATES DATA COMBINATION")
    log("=" * 50)
    log()

    projects_files, payments_files, quality_files, discovery_warnings = discover_files(
        input_dir, output_dir
    )

    all_state_slugs = sorted(set(projects_files) | set(payments_files))
    log(f"States discovered: {len(all_state_slugs)}")
    log()

    if discovery_warnings:
        log("DISCOVERY WARNINGS")
        for w in discovery_warnings:
            log(f"  - {w}")
        log()

    # Report states missing one of the two required files.
    missing_notes = []
    for slug in all_state_slugs:
        if slug not in projects_files:
            missing_notes.append(f"{display_state_name(slug)}: no '*_projects.csv' file found")
        if slug not in payments_files:
            missing_notes.append(f"{display_state_name(slug)}: no '*_payment_events.csv' file found")
    if missing_notes:
        log("MISSING FILES")
        for m in missing_notes:
            log(f"  - {m}")
        log()

    schema_issues = []

    # --- Load & validate PROJECTS files ---
    log("PROJECT DATA")
    project_frames = []
    project_row_counts = {}
    for slug in all_state_slugs:
        path = projects_files.get(slug)
        if path is None:
            continue
        df, err = read_csv_safely(path)
        if err:
            schema_issues.append(f"{display_state_name(slug)} projects ({path}): {err} -> EXCLUDED")
            log(f"{display_state_name(slug):<14}{'ERROR':>10}  ({err})")
            continue
        if df is None or len(df.columns) == 0:
            schema_issues.append(f"{display_state_name(slug)} projects ({path}): no columns -> EXCLUDED")
            log(f"{display_state_name(slug):<14}{'ERROR':>10}  (no columns)")
            continue

        label = f"{display_state_name(slug)} projects ({os.path.basename(path)})"
        validated = validate_and_reorder(df, PROJECT_COLUMNS, label, schema_issues)
        if validated is None:
            log(f"{display_state_name(slug):<14}{'ERROR':>10}  (schema mismatch, see issues below)")
            continue

        project_frames.append(validated)
        project_row_counts[slug] = len(validated)
        log(f"{display_state_name(slug):<14}{len(validated):>6} rows")

    log()

    # --- Load & validate PAYMENT EVENTS files ---
    log("PAYMENT EVENT DATA")
    payment_frames = []
    payment_row_counts = {}
    for slug in all_state_slugs:
        path = payments_files.get(slug)
        if path is None:
            continue
        df, err = read_csv_safely(path)
        if err:
            schema_issues.append(f"{display_state_name(slug)} payment_events ({path}): {err} -> EXCLUDED")
            log(f"{display_state_name(slug):<14}{'ERROR':>10}  ({err})")
            continue
        if df is None or len(df.columns) == 0:
            schema_issues.append(f"{display_state_name(slug)} payment_events ({path}): no columns -> EXCLUDED")
            log(f"{display_state_name(slug):<14}{'ERROR':>10}  (no columns)")
            continue

        label = f"{display_state_name(slug)} payment_events ({os.path.basename(path)})"
        validated = validate_and_reorder(df, PAYMENT_COLUMNS, label, schema_issues)
        if validated is None:
            log(f"{display_state_name(slug):<14}{'ERROR':>10}  (schema mismatch, see issues below)")
            continue

        payment_frames.append(validated)
        payment_row_counts[slug] = len(validated)
        log(f"{display_state_name(slug):<14}{len(validated):>6} rows")

    log()

    if schema_issues:
        log("SCHEMA ISSUES")
        for issue in schema_issues:
            log(f"  - {issue}")
        log()

    # --- Concatenate (never merge) ---
    if project_frames:
        all_projects = pd.concat(project_frames, ignore_index=True)
    else:
        all_projects = pd.DataFrame(columns=PROJECT_COLUMNS)

    if payment_frames:
        all_payments = pd.concat(payment_frames, ignore_index=True)
    else:
        all_payments = pd.DataFrame(columns=PAYMENT_COLUMNS)

    # --- Validate final combined datasets ---
    total_project_rows = len(all_projects)
    unique_work_ids = all_projects["work_id"].nunique(dropna=True) if total_project_rows else 0
    dup_mask = all_projects["work_id"].duplicated(keep=False) if total_project_rows else pd.Series(dtype=bool)
    duplicate_work_id_count = (
        all_projects.loc[dup_mask, "work_id"].nunique() if total_project_rows else 0
    )
    duplicate_row_count = int(dup_mask.sum()) if total_project_rows else 0

    total_payment_rows = len(all_payments)

    if total_payment_rows and total_project_rows:
        project_id_set = set(all_projects["work_id"].dropna())
        unmatched_mask = ~all_payments["work_id"].isin(project_id_set)
        unmatched_payment_work_ids = all_payments.loc[unmatched_mask, "work_id"].nunique()
        unmatched_payment_rows = int(unmatched_mask.sum())
    elif total_payment_rows:
        unmatched_payment_work_ids = all_payments["work_id"].nunique(dropna=True)
        unmatched_payment_rows = total_payment_rows
    else:
        unmatched_payment_work_ids = 0
        unmatched_payment_rows = 0

    missing_state_projects = int(
        all_projects["state"].isna().sum() + (all_projects["state"].astype(str).str.strip() == "").sum()
    ) if total_project_rows else 0
    missing_state_payments = int(
        all_payments["state"].isna().sum() + (all_payments["state"].astype(str).str.strip() == "").sum()
    ) if total_payment_rows else 0

    log(f"Total project rows: {total_project_rows}")
    log(f"Unique work IDs: {unique_work_ids}")
    log(f"Duplicate work IDs: {duplicate_work_id_count} (across {duplicate_row_count} rows) — reported, NOT removed")
    if missing_state_projects:
        log(f"WARNING: {missing_state_projects} project row(s) have missing/null 'state'")
    log()

    log(f"Total payment events: {total_payment_rows}")
    log(f"Unmatched payment work IDs: {unmatched_payment_work_ids} (across {unmatched_payment_rows} payment rows)")
    if missing_state_payments:
        log(f"WARNING: {missing_state_payments} payment row(s) have missing/null 'state'")
    log()

    log("QUALITY REPORT NOTES (informational only, not written to output)")
    inspect_quality_reports(quality_files)
    log()

    # --- Save outputs ---
    projects_out_path = os.path.join(output_dir, "all_states_projects.csv")
    payments_out_path = os.path.join(output_dir, "all_states_payment_events.csv")

    all_projects.to_csv(projects_out_path, index=False)
    all_payments.to_csv(payments_out_path, index=False)

    log("=" * 50)
    log("OUTPUT")
    log("=" * 50)
    log()
    log(projects_out_path)
    log(f"  columns: {list(all_projects.columns)}")
    log(f"  rows: {total_project_rows}")
    log()
    log(payments_out_path)
    log(f"  columns: {list(all_payments.columns)}")
    log(f"  rows: {total_payment_rows}")
    log()
    log("Combination completed successfully.")


if __name__ == "__main__":
    main()
