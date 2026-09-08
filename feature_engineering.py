"""
NirakshanAI — Feature Engineering Module (Shared Artifact)

This module loads raw MPLADS project and payment CSVs, cleans them, joins payment
aggregates to projects, and computes a standardized feature table.

Output: data/processed/engineered_features.csv

This script is ML-agnostic — it produces a feature table consumed by both:
- The Rule Engine (separate teammate task)
- The ML Anomaly Detection branch (Isolation Forest, this task)

No scikit-learn, no train/test split, no model code here.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Configuration / Constants
# ──────────────────────────────────────────────────────────────────────────────

RAW_PROJECTS_PATH = Path("data/all_states_projects.csv")
RAW_PAYMENTS_PATH = Path("data/all_states_payment_events.csv")
OUTPUT_PATH = Path("data/processed/engineered_features.csv")

# Reference date for "today" — fixed for reproducibility (matches env date)
TODAY = pd.Timestamp("2026-09-06")

# Physical stage proxy mapping (ordinal approximation — NOT a measured percentage)
# Overridden to 100 when work_completion_status == 'COMPLETED'
PHYSICAL_STAGE_MAP = {
    "Sanction": 0,
    "Vendor Identification": 20,
    "Time Estimation": 20,
    "Physical Inspection": 60,
    "Work partially Completed": 75,
    "Work Completed": 100,
}

# Stalled detection thresholds (tuned against actual data distribution)
STALLED_NO_PAYMENT_AGE_DAYS = 180      # sanctioned >180 days ago, zero payments
STALLED_LAST_PAYMENT_DAYS = 180        # days since last payment >180
STALLED_MIN_AGE_DAYS = 180             # minimum age since sanction to consider stalled

# Peer group minimum size for (state, work_title) — fallback to national (work_title) if smaller
PEER_GROUP_MIN_SIZE = 10

# Late-stage statuses for status_payment_mismatch
LATE_STAGE_STATUSES = {"Work partially Completed", "Work Completed"}


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def load_and_clean_projects(path: Path) -> pd.DataFrame:
    """Load projects CSV, parse dates, verify work_id uniqueness."""
    df = pd.read_csv(path)

    # Parse dates — recommended_date and sanction_date are DD-Mon-YY (e.g., 25-Aug-24)
    # expected_completion_date is YYYY-MM-DD
    df["recommended_date"] = pd.to_datetime(df["recommended_date"], format="%d-%b-%y")
    df["sanction_date"] = pd.to_datetime(df["sanction_date"], format="%d-%b-%y")
    df["expected_completion_date"] = pd.to_datetime(df["expected_completion_date"])

    # Verify work_id uniqueness
    dup_count = df["work_id"].duplicated().sum()
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate work_id values in projects file")

    # Sanity: expected_completion_date should be sanction_date + 365 days
    # (This is a verified data property — log a warning if violated)
    expected = df["sanction_date"] + pd.Timedelta(days=365)
    mismatch = (df["expected_completion_date"] != expected).sum()
    if mismatch > 0:
        print(f"[WARNING] {mismatch} rows have expected_completion_date != sanction_date + 365 days")

    return df


def load_and_clean_payments(path: Path) -> pd.DataFrame:
    """Load payments CSV, parse expenditure_date."""
    df = pd.read_csv(path)
    df["expenditure_date"] = pd.to_datetime(df["expenditure_date"])
    return df


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Compute per-work_id payment aggregates."""
    agg = payments.groupby("work_id").agg(
        total_disbursed=("reported_fund_disbursed_amount", "sum"),
        payment_count=("reported_fund_disbursed_amount", "count"),
        max_payment=("reported_fund_disbursed_amount", "max"),
        last_payment_date=("expenditure_date", "max"),
        avg_payment=("reported_fund_disbursed_amount", "mean"),
    ).reset_index()
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Feature Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Category A — Financial features."""
    df = df.copy()

    # total_disbursed: already joined from payments (0 for no payments)
    # utilization_pct = total_disbursed / sanction_amount * 100
    df["utilization_pct"] = np.where(
        df["sanction_amount"] > 0,
        df["total_disbursed"] / df["sanction_amount"] * 100,
        0.0
    )

    # unutilized_amount = sanction_amount - total_disbursed
    df["unutilized_amount"] = df["sanction_amount"] - df["total_disbursed"]

    return df


def compute_physical_stage_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Category B — Physical stage proxy from portal_execution_status.

    Mapping (assumption/proxy, documented here):
      Sanction=0, Vendor Identification=20, Time Estimation=20,
      Physical Inspection=60, Work partially Completed=75, Work Completed=100

    Overridden to 100 whenever work_completion_status == 'COMPLETED'
    (since 92% of completed works sit at 'Physical Inspection', not 'Work Completed').

    This is a ROUGH ORDINAL PROXY, not a measured progress percentage.
    """
    df = df.copy()

    df["physical_stage_proxy"] = df["portal_execution_status"].map(PHYSICAL_STAGE_MAP)

    # Override for completed works
    completed_mask = df["work_completion_status"] == "COMPLETED"
    df.loc[completed_mask, "physical_stage_proxy"] = 100

    # Any unmapped status gets NaN (should not happen with current data)
    if df["physical_stage_proxy"].isna().any():
        unmapped = df.loc[df["physical_stage_proxy"].isna(), "portal_execution_status"].unique()
        print(f"[WARNING] Unmapped portal_execution_status values: {unmapped}")
        df["physical_stage_proxy"] = df["physical_stage_proxy"].fillna(0)

    return df


def compute_progress_gap_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Category B — Progress gap proxy: utilization_pct - physical_stage_proxy.

    Positive gap = financial progress ahead of reported physical stage
    (could indicate overbilling, or legitimate upfront material purchase, or portal lag).
    """
    df = df.copy()
    df["progress_gap_proxy"] = df["utilization_pct"] - df["physical_stage_proxy"]
    return df


def compute_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Category D — Delay features."""
    df = df.copy()

    # overdue_days = max(0, today - expected_completion_date) if NOT_COMPLETED, else 0
    mask = (df["work_completion_status"] == "NOT_COMPLETED") & (df["expected_completion_date"] < TODAY)
    df["overdue_days"] = 0
    df.loc[mask, "overdue_days"] = (TODAY - df.loc[mask, "expected_completion_date"]).dt.days

    # approval_lag_days = sanction_date - recommended_date (in days)
    df["approval_lag_days"] = (df["sanction_date"] - df["recommended_date"]).dt.days

    return df


def compute_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Category F — Payment-derived features (some already in joined aggregates)."""
    df = df.copy()

    # days_since_last_payment = today - MAX(expenditure_date) per work_id
    # Null if no payments — fill with sentinel (large value) and add indicator
    df["days_since_last_payment"] = np.where(
        df["last_payment_date"].notna(),
        (TODAY - df["last_payment_date"]).dt.days,
        np.nan  # will be handled in ML preprocessing; kept as NaN here for transparency
    )

    # payment_count already from aggregate (0 for no payments)

    # max_single_payment_ratio = max_payment / total_disbursed (guard div by zero)
    df["max_single_payment_ratio"] = np.where(
        df["total_disbursed"] > 0,
        df["max_payment"] / df["total_disbursed"],
        0.0
    )

    # avg_payment_per_event (useful for payment_amount_zscore)
    df["avg_payment_per_event"] = np.where(
        df["payment_count"] > 0,
        df["total_disbursed"] / df["payment_count"],
        0.0
    )

    return df


def compute_stalled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Category E — Stalled project detection."""
    df = df.copy()

    # Age since sanction
    df["age_days"] = (TODAY - df["sanction_date"]).dt.days

    # stalled_flag: NOT_COMPLETED AND (payment_count == 0 OR days_since_last_payment > 180)
    # AND age_days > 180
    cond_no_pay = (df["work_completion_status"] == "NOT_COMPLETED") & (df["payment_count"] == 0) & (df["age_days"] > STALLED_NO_PAYMENT_AGE_DAYS)
    cond_old_pay = (
        (df["work_completion_status"] == "NOT_COMPLETED") &
        (df["days_since_last_payment"] > STALLED_LAST_PAYMENT_DAYS) &
        (df["age_days"] > STALLED_MIN_AGE_DAYS)
    )
    df["stalled_flag"] = (cond_no_pay | cond_old_pay).astype(int)

    return df


def compute_status_payment_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Status/payment mismatch: late-stage status but zero payment events."""
    df = df.copy()
    df["status_payment_mismatch"] = (
        df["portal_execution_status"].isin(LATE_STAGE_STATUSES) &
        (df["payment_count"] == 0)
    ).astype(int)
    return df


def compute_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Category C/H — Peer-relative features using median/MAD (robust to skew).

    peer_cost_zscore: (sanction_amount - peer_median) / peer_MAD
    Peer group = (state, work_title) if group size >= PEER_GROUP_MIN_SIZE,
                 else fallback to (work_title) nationally.

    payment_count_zscore, payment_amount_zscore: peer group = work_title (national)
    """
    df = df.copy()

    # --- peer_cost_zscore ---
    # Compute (state, work_title) group stats
    state_title_stats = df.groupby(["state", "work_title"])["sanction_amount"].agg(
        peer_median="median",
        peer_mad=lambda x: np.median(np.abs(x - np.median(x))),
        peer_count="count"
    ).reset_index()

    # Compute national (work_title) fallback stats
    title_stats = df.groupby("work_title")["sanction_amount"].agg(
        peer_median_national="median",
        peer_mad_national=lambda x: np.median(np.abs(x - np.median(x))),
        peer_count_national="count"
    ).reset_index()

    # Merge both
    df = df.merge(state_title_stats, on=["state", "work_title"], how="left")
    df = df.merge(title_stats, on="work_title", how="left")

    # Choose peer stats: use state-level if count >= min_size, else national
    use_state = df["peer_count"] >= PEER_GROUP_MIN_SIZE
    df["peer_median_used"] = np.where(use_state, df["peer_median"], df["peer_median_national"])
    df["peer_mad_used"] = np.where(use_state, df["peer_mad"], df["peer_mad_national"])

    # Guard against MAD = 0 (all same value in group) — fallback to small epsilon
    df["peer_mad_used"] = df["peer_mad_used"].replace(0, 1.0)

    df["peer_cost_zscore"] = (df["sanction_amount"] - df["peer_median_used"]) / df["peer_mad_used"]

    # Clean up intermediate columns
    df.drop(columns=[
        "peer_median", "peer_mad", "peer_count",
        "peer_median_national", "peer_mad_national", "peer_count_national",
        "peer_median_used", "peer_mad_used"
    ], inplace=True, errors="ignore")

    # --- payment_count_zscore & payment_amount_zscore (peer group = work_title nationally) ---
    pay_stats = df.groupby("work_title").agg(
        pay_count_median=("payment_count", "median"),
        pay_count_mad=("payment_count", lambda x: np.median(np.abs(x - np.median(x)))),
        pay_amt_median=("avg_payment_per_event", "median"),
        pay_amt_mad=("avg_payment_per_event", lambda x: np.median(np.abs(x - np.median(x)))),
    ).reset_index()

    # Guard MAD = 0
    pay_stats["pay_count_mad"] = pay_stats["pay_count_mad"].replace(0, 1.0)
    pay_stats["pay_amt_mad"] = pay_stats["pay_amt_mad"].replace(0, 1.0)

    df = df.merge(pay_stats, on="work_title", how="left")

    df["payment_count_zscore"] = (df["payment_count"] - df["pay_count_median"]) / df["pay_count_mad"]
    df["payment_amount_zscore"] = (df["avg_payment_per_event"] - df["pay_amt_median"]) / df["pay_amt_mad"]

    df.drop(columns=[
        "pay_count_median", "pay_count_mad", "pay_amt_median", "pay_amt_mad"
    ], inplace=True, errors="ignore")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_table() -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    print("[1/7] Loading and cleaning projects...")
    projects = load_and_clean_projects(RAW_PROJECTS_PATH)

    print("[2/7] Loading and cleaning payments...")
    payments = load_and_clean_payments(RAW_PAYMENTS_PATH)

    print("[3/7] Aggregating payments per work_id...")
    pay_agg = aggregate_payments(payments)

    print("[4/7] Joining payments to projects...")
    df = projects.merge(pay_agg, on="work_id", how="left")

    # Fill payment aggregates for works with no payment events
    df["total_disbursed"] = df["total_disbursed"].fillna(0.0)
    df["payment_count"] = df["payment_count"].fillna(0).astype(int)
    df["max_payment"] = df["max_payment"].fillna(0.0)
    df["avg_payment"] = df["avg_payment"].fillna(0.0)
    df["last_payment_date"] = df["last_payment_date"]  # keep NaT for no payments

    print("[5/7] Computing financial features...")
    df = compute_financial_features(df)

    print("[6/7] Computing progress, delay, payment, stalled, peer features...")
    df = compute_physical_stage_proxy(df)
    df = compute_progress_gap_proxy(df)
    df = compute_delay_features(df)
    df = compute_payment_features(df)
    df = compute_stalled_features(df)
    df = compute_status_payment_mismatch(df)
    df = compute_peer_features(df)

    print("[7/7] Finalizing output columns...")

    # ──────────────────────────────────────────────────────────────────────────
    # Define output column groups (documentation for consumers)
    # ──────────────────────────────────────────────────────────────────────────

    # Context/passthrough columns — useful for human review, NOT model inputs
    context_cols = [
        "work_id",
        "source_serial_number",
        "work_category",
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
        "total_amount_disbursed",  # original field (only for COMPLETED)
        "image",
    ]

    # Engineered features — actual model/rule inputs
    feature_cols = [
        # Financial (Category A)
        "total_disbursed",
        "utilization_pct",
        "unutilized_amount",
        # Progress (Category B)
        "physical_stage_proxy",
        "progress_gap_proxy",
        # Delay (Category D)
        "overdue_days",
        "approval_lag_days",
        # Payment (Category F)
        "days_since_last_payment",
        "payment_count",
        "max_single_payment_ratio",
        "avg_payment_per_event",
        # Stalled (Category E)
        "stalled_flag",
        "age_days",
        # Status mismatch
        "status_payment_mismatch",
        # Peer-relative (Category C/H)
        "peer_cost_zscore",
        "payment_count_zscore",
        "payment_amount_zscore",
    ]

    # Verify all feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Select and order: context first, then features
    output_cols = context_cols + feature_cols
    df_out = df[output_cols].copy()

    return df_out


def save_feature_table(df: pd.DataFrame, path: Path):
    """Save feature table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved feature table to {path} ({len(df)} rows, {len(df.columns)} columns)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feature_df = build_feature_table()
    save_feature_table(feature_df, OUTPUT_PATH)

    # Print feature summary for verification
    print("\n=== FEATURE SUMMARY ===")
    feature_cols = [c for c in feature_df.columns if c not in [
        "work_id", "source_serial_number", "work_category", "work_title", "state",
        "ida", "mp_name", "constituency", "work_description", "recommended_date",
        "sanction_date", "expected_completion_date", "sanction_amount",
        "portal_execution_status", "work_completion_status", "total_amount_disbursed", "image"
    ]]
    for col in feature_cols:
        dtype = feature_df[col].dtype
        nulls = feature_df[col].isna().sum()
        print(f"  {col}: dtype={dtype}, nulls={nulls}")