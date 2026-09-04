#!/usr/bin/env python3
"""
combine_mplads.py

Combines MPLADS "Works Sanctioned.csv" and "Works Completed.csv" into a
single "Works_Sanctioned_Combined.csv", matching purely on a cleaned
work_id extracted from the free-text 'Work' column.

Run:
    python combine_mplads.py

Requires: pandas
"""

import re
import sys
import pandas as pd

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SANCTIONED_FILE = "Works Sanctioned.csv"
COMPLETED_FILE = "Works Completed.csv"
OUTPUT_FILE = "Works_Sanctioned_Combined.csv"

# Actual uploaded filenames on this system use underscores instead of
# spaces. Try the "official" names first, then fall back automatically
# so the script still runs without editing.
SANCTIONED_FALLBACK = "Works_Sanctioned.csv"
COMPLETED_FALLBACK = "Works_Completed.csv"

# ----------------------------------------------------------------------
# Final-output-only column rename + reorder (applied to the finished
# DataFrame, after all existing processing/validation is complete --
# does NOT touch any source column name used earlier in the pipeline).
# Key = existing column name produced by the pipeline above.
# Value = desired final output column name.
# ----------------------------------------------------------------------
FINAL_COLUMN_RENAME_MAP = {
    "Sr. No.": "source_serial_number",
    "Work category": "work_category",
    "work_id": "work_id",
    "work_title": "work_title",
    "State": "state",
    "IDA": "ida",
    "Hon'ble Members of Parliament": "mp_name",
    "Constituency": "constituency",
    "Work description": "work_description",
    "Recommended date": "recommended_date",
    "Sanction Date": "sanction_date",
    "expected_completion_date": "expected_completion_date",
    "Sanction Amount ( ₹ )": "sanction_amount",
    "Work Status": "portal_execution_status",
    "work_completion_status": "work_completion_status",
    "amount_disbursed": "total_amount_disbursed",
    "Image": "image",
}

# Exact final column order requested.
FINAL_COLUMN_ORDER = [
    "source_serial_number",
    "work_category",
    "work_id",
    "work_title",
    "state",
    "ida",
    "mp_name",
    "constituency",
    "work_description",
    "recommended_date",
    "sanction_date",
    "expected_completion_date",
    "sanction_amount",
    "portal_execution_status",
    "work_completion_status",
    "total_amount_disbursed",
    "image",
]

# Work ID pattern: WS/MP<digits>/<YYYY-YYYY>/<digits>
WORK_ID_REGEX = re.compile(r"WS/MP\d+/\d{4}-\d{4}/\d+")

# Strict, fully-anchored Work ID validation pattern -- the ENTIRE
# work_id string must match this (not just contain a matching
# substring). Used as a separate validation step after extraction.
WORK_ID_VALID_REGEX = re.compile(r"^WS/MP\d+/\d{4}-\d{4}/\d+$")


def resolve_path(preferred, fallback):
    """Return preferred path if it exists on disk, else the fallback."""
    import os
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    # Neither exists -- return preferred so the resulting error message
    # is meaningful to the user.
    return preferred


def clean_work_value(value):
    """
    Normalize a raw 'Work' cell:
      - Coerce to string, handle NaN/blank safely.
      - Collapse accidental whitespace/tabs that appear right after 'WS/'
        (e.g. 'WS/\\t MP319/...' -> 'WS/MP319/...').
      - Strip leading/trailing whitespace.
    """
    if pd.isna(value):
        return ""
    text = str(value)
    # Collapse any whitespace (space, tab, newline) directly after 'WS/'
    text = re.sub(r"WS/\s+", "WS/", text)
    return text.strip()


def split_work(value):
    """
    Given a cleaned 'Work' string, extract (work_id, work_title) using the
    WORK_ID_REGEX. The work_title is everything after the hyphen that
    immediately follows the matched work_id (title itself may contain
    further hyphens, so we do NOT split blindly on '-').

    Returns (work_id, work_title). Both are "" if the pattern is not found
    or the value is blank/malformed -- this is a soft failure, not a crash.
    """
    if not value:
        return "", ""

    match = WORK_ID_REGEX.search(value)
    if not match:
        return "", ""

    work_id = match.group(0)
    remainder = value[match.end():]

    # The title should follow a hyphen immediately after the ID.
    if remainder.startswith("-"):
        work_title = remainder[1:].strip()
    else:
        # No hyphen immediately after the ID -- malformed title, but we
        # still keep the work_id we found and report an empty title
        # rather than discarding the row.
        work_title = remainder.strip()

    return work_id, work_title


def remove_grand_total_rows(df):
    """
    Detect and remove rows that represent a 'Grand Total' summary row.
    Robust version: checks EVERY column in the row (not just the first
    column or the Work column) for a cell that equals "grand total"
    after trimming whitespace and lower-casing. Returns the cleaned
    dataframe and the count of rows removed.
    """
    normalized = df.apply(lambda col: col.astype(str).str.strip().str.lower())
    mask = (normalized == "grand total").any(axis=1)

    removed = int(mask.sum())
    cleaned = df.loc[~mask].copy()
    return cleaned, removed


def find_duplicate_ids(df, id_col, orig_col, label):
    """
    Report duplicate work_id values (ignoring blank/missing IDs -- blanks
    are never treated as duplicates of one another) using the ORIGINAL
    CSV row numbers (header counted as row 1) stored in `orig_col`.

    For every work_id that occurs more than once, the FIRST occurrence
    (lowest original row number) is marked to KEEP; every later
    occurrence is marked to DELETE.

    Returns:
        duplicates: dict {work_id: {"rows": [...], "kept": row, "deleted": [...]}}
        delete_rows: set of original row numbers to remove from df
    """
    non_blank = df[df[id_col] != ""].copy()

    duplicates = {}
    delete_rows = set()

    if not non_blank.empty:
        grouped = non_blank.groupby(id_col)[orig_col].apply(lambda s: sorted(s.tolist()))
        for work_id, rows in grouped.items():
            if len(rows) > 1:
                kept = rows[0]
                deleted = rows[1:]
                duplicates[work_id] = {"rows": rows, "kept": kept, "deleted": deleted}
                delete_rows.update(deleted)

    if duplicates:
        print(f"  Duplicate work_ids in {label}: {len(duplicates)}")
        for work_id, info in duplicates.items():
            print(f"    {work_id} -> original rows {info['rows']} "
                  f"(KEEP row {info['kept']}, DELETE rows {info['deleted']})")
    else:
        print(f"  Duplicate work_ids in {label}: 0")

    return duplicates, delete_rows


def clean_monetary_value(raw):
    """
    Attempt to safely parse a raw 'amount disbursed' cell into a numeric
    (float) value, without ever silently coercing an invalid value to 0.

    Handles:
      - surrounding whitespace
      - commas used as thousands separators (e.g. Indian "8,12,494" style)
      - currency symbols / words such as '₹', '$', 'Rs.', 'Rs', 'INR'
      - other harmless formatting characters around an otherwise valid
        monetary value

    Returns a (status, result) tuple:
      - ("blank", "")            -- raw was empty/blank; preserved as blank
      - ("valid", float_value)   -- successfully parsed to a float
      - ("invalid", raw)         -- could not be safely parsed; the
                                     original raw value is returned
                                     unchanged so nothing is discarded
    """
    if raw is None:
        return "blank", ""

    text = str(raw).strip()
    if text == "":
        return "blank", ""

    cleaned = text
    # Strip common currency words/symbols (case-insensitive), then any
    # remaining surrounding whitespace.
    cleaned = re.sub(r"(?i)\bRs\.?\b|\bINR\b", "", cleaned)
    cleaned = cleaned.replace("₹", "").replace("$", "")
    cleaned = cleaned.strip()
    # Thousands-separator commas are harmless formatting -- remove them.
    cleaned = cleaned.replace(",", "").strip()

    # Only accept the result if it is now unambiguously a plain number.
    if re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
        try:
            return "valid", float(cleaned)
        except ValueError:
            return "invalid", raw

    return "invalid", raw


def find_sr_no_col(df):
    """Return the 'Sr. No.' column name if present (whitespace/case
    insensitive match), else None."""
    for c in df.columns:
        if c.strip().lower().rstrip(".") in ("sr no", "sr. no"):
            return c
    return None


def validate_work_id_format(df, id_col, orig_row_col, label):
    """
    Enforce the strict, fully-anchored Work ID format:

        ^WS/MP\\d+/\\d{4}-\\d{4}/\\d+$

    This is a separate, stricter check than the extraction regex used
    earlier -- it requires the ENTIRE work_id string to match (MP
    followed by 1+ digits, a 4-digit year, a literal '-', another
    4-digit year, '/', then 1+ digits for the final number).

    Any row with a non-blank work_id that does NOT match this pattern
    is reported (original CSV row number, Sr. No. if the column
    exists, and the invalid work_id value) and then removed entirely.
    Blank/unparseable work_ids (already tracked separately earlier in
    the pipeline) are left untouched here. Valid work_ids -- and their
    rows -- pass through completely unchanged.

    Returns (cleaned_df, removed_count).
    """
    sr_no_col = find_sr_no_col(df)

    non_blank = df[id_col] != ""
    invalid_mask = non_blank & ~df[id_col].apply(
        lambda v: bool(WORK_ID_VALID_REGEX.match(v))
    )

    removed = int(invalid_mask.sum())

    if removed:
        print(f"\nInvalid Work ID format rows in {label} ({removed}):")
        for _, row in df.loc[invalid_mask].iterrows():
            orig_row = row[orig_row_col]
            sr_no = row[sr_no_col] if sr_no_col else "N/A"
            print(f"    Original CSV row {orig_row} | Sr. No. {sr_no} | "
                  f"invalid work_id: {row[id_col]!r}")
    else:
        print(f"\nInvalid Work ID format rows in {label}: 0")

    cleaned = df.loc[~invalid_mask].copy()
    return cleaned, removed


def print_id_list(label, ids, limit=50):
    """Print a (possibly long) list of work IDs, truncated for readability."""
    ids = list(ids)
    print(f"  {label} ({len(ids)}):")
    shown = ids[:limit]
    for work_id in shown:
        print(f"    {work_id}")
    if len(ids) > limit:
        print(f"    ... and {len(ids) - limit} more")


def main(sanctioned_file=None, completed_file=None, output_file=None):
    """
    Run the full pipeline once.

    By default uses the module-level SANCTIONED_FILE / COMPLETED_FILE /
    OUTPUT_FILE constants (unchanged behaviour). Pass explicit paths to
    process a different state's files without editing the script --
    this is what enables looping over multiple states (see the CLI /
    batch-runner instructions).
    """
    sanctioned_path = resolve_path(sanctioned_file or SANCTIONED_FILE, SANCTIONED_FALLBACK)
    completed_path = resolve_path(completed_file or COMPLETED_FILE, COMPLETED_FALLBACK)
    output_path = output_file or OUTPUT_FILE

    print("=" * 70)
    print("MPLADS Works Sanctioned + Completed Combiner")
    print(f"Sanctioned: {sanctioned_path}")
    print(f"Completed : {completed_path}")
    print(f"Output    : {output_path}")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------
    try:
        sanctioned_df = pd.read_csv(sanctioned_path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find sanctioned file at '{sanctioned_path}'")
        sys.exit(1)

    try:
        completed_df = pd.read_csv(completed_path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find completed file at '{completed_path}'")
        sys.exit(1)

    rows_read_sanctioned = len(sanctioned_df)
    rows_read_completed = len(completed_df)

    print(f"\nRows read from Sanctioned file : {rows_read_sanctioned}")
    print(f"Rows read from Completed file  : {rows_read_completed}")

    # Capture ORIGINAL CSV row numbers (header = row 1) before ANY rows
    # are removed, so every report below refers to the untouched source
    # file's row numbering, not a post-cleaning position.
    sanctioned_df["_orig_row"] = sanctioned_df.index + 2
    completed_df["_orig_row"] = completed_df.index + 2

    # --------------------------------------------------------------
    # Identify Work columns (case-sensitive exact names as provided,
    # but fall back to a case-insensitive search just in case).
    # --------------------------------------------------------------
    def find_work_col(df, label):
        if "Work" in df.columns:
            return "Work"
        for c in df.columns:
            if c.strip().lower() == "work":
                return c
        print(f"ERROR: Could not find a 'Work' column in {label} file. "
              f"Columns found: {list(df.columns)}")
        sys.exit(1)

    sanctioned_work_col = find_work_col(sanctioned_df, "Sanctioned")
    completed_work_col = find_work_col(completed_df, "Completed")

    # --------------------------------------------------------------
    # Step 2: Remove Grand Total rows (checks EVERY column in the row)
    # --------------------------------------------------------------
    sanctioned_df, gt_removed_sanctioned = remove_grand_total_rows(sanctioned_df)
    completed_df, gt_removed_completed = remove_grand_total_rows(completed_df)

    print(f"\nGrand Total rows removed (Sanctioned): {gt_removed_sanctioned}")
    print(f"Grand Total rows removed (Completed) : {gt_removed_completed}")

    # --------------------------------------------------------------
    # Step 1: Clean + split Work column in BOTH files
    # --------------------------------------------------------------
    sanctioned_df["_work_clean"] = sanctioned_df[sanctioned_work_col].apply(clean_work_value)
    completed_df["_work_clean"] = completed_df[completed_work_col].apply(clean_work_value)

    sanctioned_split = sanctioned_df["_work_clean"].apply(split_work)
    sanctioned_df["work_id"] = sanctioned_split.apply(lambda t: t[0])
    sanctioned_df["work_title"] = sanctioned_split.apply(lambda t: t[1])

    completed_split = completed_df["_work_clean"].apply(split_work)
    completed_df["work_id"] = completed_split.apply(lambda t: t[0])
    completed_df["work_title"] = completed_split.apply(lambda t: t[1])

    valid_sanctioned_ids = int((sanctioned_df["work_id"] != "").sum())
    failed_sanctioned_ids = int((sanctioned_df["work_id"] == "").sum())
    valid_completed_ids = int((completed_df["work_id"] != "").sum())
    failed_completed_ids = int((completed_df["work_id"] == "").sum())

    print(f"\nValid work IDs in Sanctioned : {valid_sanctioned_ids}")
    print(f"Failed/missing IDs (Sanctioned): {failed_sanctioned_ids}")
    print(f"Valid work IDs in Completed   : {valid_completed_ids}")
    print(f"Failed/missing IDs (Completed): {failed_completed_ids}")

    if failed_sanctioned_ids > 0:
        bad_rows = sanctioned_df.loc[sanctioned_df["work_id"] == "", "_orig_row"].tolist()
        print(f"  Sanctioned original row numbers with unparseable/blank Work value: {bad_rows}")
    if failed_completed_ids > 0:
        bad_rows = completed_df.loc[completed_df["work_id"] == "", "_orig_row"].tolist()
        print(f"  Completed original row numbers with unparseable/blank Work value: {bad_rows}")

    # --------------------------------------------------------------
    # Step 1b: Strict Work ID format validation
    #
    # Separate from extraction above. Enforces the fully-anchored
    # format ^WS/MP\d+/\d{4}-\d{4}/\d+$ against every non-blank
    # work_id. Any work_id that does not match is reported (original
    # CSV row number, Sr. No., invalid value) and its entire row is
    # deleted at this step -- it is never reconstructed, modified, or
    # guessed. Valid work_ids pass through completely unchanged.
    # --------------------------------------------------------------
    sanctioned_df, invalid_format_sanctioned = validate_work_id_format(
        sanctioned_df, "work_id", "_orig_row", "Sanctioned"
    )
    completed_df, invalid_format_completed = validate_work_id_format(
        completed_df, "work_id", "_orig_row", "Completed"
    )

    print(f"\nInvalid Work ID format rows removed (Sanctioned): {invalid_format_sanctioned}")
    print(f"Invalid Work ID format rows removed (Completed)  : {invalid_format_completed}")

    # --------------------------------------------------------------
    # Step 6: Duplicate detection + removal (both files)
    #
    # Rule: if a work_id occurs more than once, KEEP ONLY the first
    # occurrence (lowest original row number) and DELETE every later
    # occurrence. Blank/missing work_ids are never treated as
    # duplicates of one another. This is equivalent to
    # df.drop_duplicates(subset="work_id", keep="first") restricted to
    # rows with a non-blank work_id.
    # --------------------------------------------------------------
    print("\nDuplicate work_id check (Sanctioned):")
    dup_sanctioned, delete_rows_sanctioned = find_duplicate_ids(
        sanctioned_df, "work_id", "_orig_row", "Sanctioned"
    )
    print("\nDuplicate work_id check (Completed):")
    dup_completed, delete_rows_completed = find_duplicate_ids(
        completed_df, "work_id", "_orig_row", "Completed"
    )

    rows_removed_sanctioned_dupes = len(delete_rows_sanctioned)
    rows_removed_completed_dupes = len(delete_rows_completed)

    print(f"\nTotal duplicate sanctioned IDs found : {len(dup_sanctioned)}")
    print(f"Total duplicate completed IDs found  : {len(dup_completed)}")
    print(f"Sanctioned rows removed (duplicates) : {rows_removed_sanctioned_dupes}")
    print(f"Completed rows removed (duplicates)  : {rows_removed_completed_dupes}")

    # Physically remove the later-occurrence duplicate rows. The first
    # occurrence of every work_id, and every row with a blank work_id,
    # is always preserved.
    sanctioned_df = sanctioned_df.loc[
        ~sanctioned_df["_orig_row"].isin(delete_rows_sanctioned)
    ].copy()
    completed_df = completed_df.loc[
        ~completed_df["_orig_row"].isin(delete_rows_completed)
    ].copy()

    # After duplicate removal there is at most one row per non-blank
    # work_id in each dataframe, so the completed lookup is simply the
    # non-blank rows (no further drop_duplicates needed -- kept as a
    # harmless safety net in case of any edge case).
    completed_lookup = completed_df[completed_df["work_id"] != ""].drop_duplicates(
        subset="work_id", keep="first"
    )

    # --------------------------------------------------------------
    # Step 4: expected_completion_date = Sanction Date + 1 year
    # --------------------------------------------------------------
    sanction_date_col = None
    for c in sanctioned_df.columns:
        if c.strip().lower() == "sanction date":
            sanction_date_col = c
            break

    if sanction_date_col is None:
        print("ERROR: Could not find a 'Sanction Date' column in the Sanctioned file. "
              f"Columns found: {list(sanctioned_df.columns)}")
        sys.exit(1)

    # Sanction Date is typically formatted like "29-Nov-24". Try that
    # explicit format first (fast, no warnings); fall back to a generic
    # dayfirst parse for any values that don't match, so odd/mixed
    # formats are still handled instead of crashing.
    parsed_dates = pd.to_datetime(
        sanctioned_df[sanction_date_col], format="%d-%b-%y", errors="coerce"
    )
    still_missing = parsed_dates.isna() & (sanctioned_df[sanction_date_col].str.strip() != "")
    if still_missing.any():
        fallback_parsed = pd.to_datetime(
            sanctioned_df.loc[still_missing, sanction_date_col],
            errors="coerce", dayfirst=True,
        )
        parsed_dates.loc[still_missing] = fallback_parsed
    missing_invalid_dates = int(parsed_dates.isna().sum())

    expected_completion = parsed_dates + pd.DateOffset(years=1)
    sanctioned_df["expected_completion_date"] = expected_completion.dt.strftime("%Y-%m-%d")
    # Where the original date was missing/invalid, leave blank instead of "NaT".
    sanctioned_df.loc[parsed_dates.isna(), "expected_completion_date"] = ""

    print(f"\nMissing/invalid Sanction Date count: {missing_invalid_dates}")

    # --------------------------------------------------------------
    # Step 5: Match completed works purely on work_id
    # --------------------------------------------------------------
    amount_col = None
    for c in completed_lookup.columns:
        if c.strip().lower() == "amount disbursed ( ₹ )":
            amount_col = c
            break
    if amount_col is None:
        # Fall back to a looser match in case of encoding differences.
        for c in completed_lookup.columns:
            if "amount disbursed" in c.strip().lower():
                amount_col = c
                break
    if amount_col is None:
        print("ERROR: Could not find an 'Amount Disbursed ( ₹ )' column in the "
              f"Completed file. Columns found: {list(completed_lookup.columns)}")
        sys.exit(1)

    image_col = None
    for c in completed_lookup.columns:
        if c.strip().lower() == "image":
            image_col = c
            break
    if image_col is None:
        print("ERROR: Could not find an 'Image' column in the Completed file. "
              f"Columns found: {list(completed_lookup.columns)}")
        sys.exit(1)

    completed_map = completed_lookup.set_index("work_id")[[amount_col, image_col]]

    matched_mask = sanctioned_df["work_id"].isin(completed_map.index) & (sanctioned_df["work_id"] != "")

    sanctioned_df["work_completion_status"] = "NOT_COMPLETED"
    sanctioned_df.loc[matched_mask, "work_completion_status"] = "COMPLETED"

    sanctioned_df["amount_disbursed"] = ""
    sanctioned_df["Image"] = ""

    sanctioned_df.loc[matched_mask, "amount_disbursed"] = sanctioned_df.loc[matched_mask, "work_id"].map(
        completed_map[amount_col]
    )
    sanctioned_df.loc[matched_mask, "Image"] = sanctioned_df.loc[matched_mask, "work_id"].map(
        completed_map[image_col]
    )

    found_in_completed = int(matched_mask.sum())
    not_found_in_completed = int(len(sanctioned_df) - found_in_completed)

    print(f"\nSanctioned IDs found in Completed    : {found_in_completed}")
    print(f"Sanctioned IDs NOT found in Completed : {not_found_in_completed}")

    # --------------------------------------------------------------
    # Step 3: Build final output
    # --------------------------------------------------------------
    original_sanctioned_cols = [
        c for c in sanctioned_df.columns
        if c not in (
            sanctioned_work_col,
            "_work_clean",
            "_orig_row",
            "work_id",
            "work_title",
            "expected_completion_date",
            "work_completion_status",
            "amount_disbursed",
            "Image",
        )
    ]

    final_cols = original_sanctioned_cols + [
        "work_id",
        "work_title",
        "expected_completion_date",
        "work_completion_status",
        "amount_disbursed",
        "Image",
    ]

    final_df = sanctioned_df[final_cols].copy()

    # --------------------------------------------------------------
    # Convert amount_disbursed to numeric/decimal.
    #
    # This only rewrites the VALUES already sitting in the finished
    # final_df -- it does not touch matching logic, duplicate handling,
    # date processing, row counts, column names, or column order.
    # Invalid values are never coerced to 0/0.0 and are never
    # discarded: the original raw value is kept as-is and reported
    # below with its original CSV row number. Blank values are left
    # blank per existing behavior.
    # --------------------------------------------------------------
    orig_rows_for_final = sanctioned_df.loc[final_df.index, "_orig_row"]

    valid_amount_count = 0
    blank_amount_count = 0
    invalid_amounts = []  # list of (orig_row, raw_value)
    converted_amounts = []

    for idx in final_df.index:
        raw_value = final_df.at[idx, "amount_disbursed"]
        status, result = clean_monetary_value(raw_value)
        if status == "valid":
            valid_amount_count += 1
        elif status == "blank":
            blank_amount_count += 1
        else:
            invalid_amounts.append((orig_rows_for_final.loc[idx], raw_value))
        converted_amounts.append(result)

    final_df["amount_disbursed"] = converted_amounts

    print("\n" + "-" * 70)
    print("AMOUNT DISBURSED CONVERSION SUMMARY")
    print("-" * 70)
    print(f"Valid amounts converted to numeric : {valid_amount_count}")
    print(f"Blank/missing amounts              : {blank_amount_count}")
    print(f"Invalid/unparseable amounts        : {len(invalid_amounts)}")
    if invalid_amounts:
        print("  Invalid values (original CSV row -> raw value):")
        for orig_row, raw_value in invalid_amounts:
            print(f"    Row {orig_row}: {raw_value!r}")

    # --------------------------------------------------------------
    # Step 7: Validation summary
    # --------------------------------------------------------------
    final_row_count = len(final_df)
    # processed sanctioned row count == sanctioned rows AFTER Grand Total
    # removal AND duplicate removal (this is the base dataset).
    processed_sanctioned_rows = len(sanctioned_df)

    matched_ids = sorted(sanctioned_df.loc[matched_mask, "work_id"].tolist())
    unmatched_ids = sorted(
        sanctioned_df.loc[~matched_mask & (sanctioned_df["work_id"] != ""), "work_id"].tolist()
    )

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Rows read from Sanctioned                     : {rows_read_sanctioned}")
    print(f"Rows read from Completed                      : {rows_read_completed}")
    print(f"Grand Total rows removed (Sanctioned)         : {gt_removed_sanctioned}")
    print(f"Grand Total rows removed (Completed)          : {gt_removed_completed}")
    print(f"Valid work IDs before dup removal (Sanctioned): {valid_sanctioned_ids}")
    print(f"Invalid/missing work IDs (Sanctioned)         : {failed_sanctioned_ids}")
    print(f"Valid work IDs before dup removal (Completed) : {valid_completed_ids}")
    print(f"Invalid/missing work IDs (Completed)          : {failed_completed_ids}")
    print(f"Invalid Work ID FORMAT rows removed (Sanctioned): {invalid_format_sanctioned}")
    print(f"Invalid Work ID FORMAT rows removed (Completed) : {invalid_format_completed}")
    print(f"Total invalid Work ID rows removed (both files) : {invalid_format_sanctioned + invalid_format_completed}")
    print(f"Duplicate work_ids found (Sanctioned)         : {len(dup_sanctioned)}")
    print(f"Duplicate work_ids found (Completed)          : {len(dup_completed)}")
    print(f"Sanctioned rows removed due to duplicates     : {rows_removed_sanctioned_dupes}")
    print(f"Completed rows removed due to duplicates      : {rows_removed_completed_dupes}")
    print(f"Processed sanctioned row count (base dataset) : {processed_sanctioned_rows}")
    print(f"Sanctioned IDs found in Completed              : {found_in_completed}")
    print(f"Sanctioned IDs NOT found in Completed           : {not_found_in_completed}")
    print(f"Missing/invalid Sanction Date count            : {missing_invalid_dates}")
    print(f"Final output row count                         : {final_row_count}")

    print()
    print_id_list("Sanctioned work IDs FOUND in Completed", matched_ids)
    print_id_list("Sanctioned work IDs NOT found in Completed", unmatched_ids)

    # --------------------------------------------------------------
    # Step 13: Final row count safety check
    # --------------------------------------------------------------
    if final_row_count != processed_sanctioned_rows:
        raise RuntimeError(
            "Final row count changed unexpectedly! "
            f"final_df has {final_row_count} rows but the processed "
            f"sanctioned dataset (after Grand Total + duplicate removal) "
            f"has {processed_sanctioned_rows} rows. The Completed file "
            "must never add or remove rows from the final output -- "
            "aborting without writing the output file."
        )
    print("\nOK: final output row count exactly matches the processed "
          "sanctioned row count.")

    # --------------------------------------------------------------
    # FINAL-OUTPUT-ONLY column rename + reorder.
    # This runs strictly AFTER all existing processing, matching,
    # validation, and the row-count safety check above. It only
    # relabels/reorders the already-finished `final_df` -- no source
    # column names, values, or upstream logic are touched.
    # --------------------------------------------------------------
    missing_source_cols = [c for c in FINAL_COLUMN_RENAME_MAP if c not in final_df.columns]
    if missing_source_cols:
        raise RuntimeError(
            "Cannot apply final column rename: the finished DataFrame is "
            f"missing expected source column(s) {missing_source_cols}. "
            f"Columns actually present: {list(final_df.columns)}"
        )

    final_df = final_df.rename(columns=FINAL_COLUMN_RENAME_MAP)
    final_df = final_df[FINAL_COLUMN_ORDER]

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved combined file to: {output_path}")
    return output_path


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Combine MPLADS Works Sanctioned + Works Completed CSVs "
                     "for one state. Run once per state, pointing at that "
                     "state's two input files."
    )
    parser.add_argument("--sanctioned", default=None,
                         help=f"Path to the Sanctioned CSV (default: '{SANCTIONED_FILE}')")
    parser.add_argument("--completed", default=None,
                         help=f"Path to the Completed CSV (default: '{COMPLETED_FILE}')")
    parser.add_argument("--output", default=None,
                         help=f"Path to write the combined CSV (default: '{OUTPUT_FILE}')")
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    main(_args.sanctioned, _args.completed, _args.output)
