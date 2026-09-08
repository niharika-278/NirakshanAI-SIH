"""
config.py
---------
Central place for:
  - Rule weights (must sum to 100, but code doesn't hard-require it —
    risk_engine normalizes by sum of AVAILABLE weights anyway)
  - Risk level bands (LOW / MEDIUM / HIGH / CRITICAL)
  - Small tunable knobs used inside individual rules

Change numbers here instead of hunting through rule files.
"""

# ---------------------------------------------------------------------------
# 1. WEIGHTS — how much each rule influences the FINAL score
# ---------------------------------------------------------------------------
WEIGHTS = {
    "R1":   7,   # Completed but Underfunded
    "R2":   9,   # Reconciliation Mismatch            (trimmed 1 to make room for R3)
    "R3":   5,   # Payment Timing Integrity            (NEW — re-added; low weight, 0 hits so far but a valid control)
    "R4":   7,   # Payment Burst Pattern
    "R5":   10,  # Cost / Sanction Overrun
    "R6":   11,  # Unusually Large Single Payment      (trimmed 1)
    "R7":   7,   # Vendor Concentration & Local Dominance
    "R8":   13,  # Duplicate / Overlapping Project     (trimmed 2 — still the single highest weight)
    "R9":   7,   # Completed Without Evidence
    "R10":  7,   # Completion Delay / Overdue
    "R11A": 5,   # Recommendation-to-Sanction Delay
    "R11B": 12,  # High Sanction Amount vs Peers       (trimmed 1)
}
assert sum(WEIGHTS.values()) == 100, "Weights should sum to 100 (sanity check)"

# ---------------------------------------------------------------------------
# 2. RISK BANDS — final 0-100 score -> human label
# ---------------------------------------------------------------------------
RISK_BANDS = [
    (0, 30,  "LOW"),
    (30, 60, "MEDIUM"),
    (60, 80, "HIGH"),
    (80, 101, "CRITICAL"),   # 101 so that a perfect 100 is included
]


def score_to_band(score: float) -> str:
    """Map a final 0-100 score to a LOW/MEDIUM/HIGH/CRITICAL label."""
    if score is None:
        return "UNKNOWN"
    for low, high, label in RISK_BANDS:
        if low <= score < high:
            return label
    return "CRITICAL"


# ---------------------------------------------------------------------------
# 3. COLUMN NAMES — adjust these if your CSV headers differ
# ---------------------------------------------------------------------------
COLUMNS = {
    "work_id": "work_id",
    "work_title": "work_title",
    "work_description": "work_description",
    "constituency": "constituency",          # or "area" / "location"
    "sanction_amount": "sanction_amount",
    "total_disbursed": "total_amount_disbursed",   # FIXED: real header is total_amount_disbursed
    "status": "work_completion_status",       # e.g. COMPLETED / IN-PROGRESS
    "expected_completion_date": "expected_completion_date",
    "recommendation_date": "recommended_date",      # FIXED: real header is recommended_date
    "sanction_date": "sanction_date",
    "has_image": "image",                     # FIXED: real header is image
    "vendor": "vendor_name",                  # NOTE: projects.csv has NO vendor column at all —
                                               # it must be merged in from payments.csv first (see main.py patch)
}

PAYMENT_COLUMNS = {
    "work_id": "work_id",
    "amount": "reported_fund_disbursed_amount",   # FIXED: real header is reported_fund_disbursed_amount
    "date": "expenditure_date",
}

# ---------------------------------------------------------------------------
# 4. RULE-SPECIFIC THRESHOLDS / KNOBS
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "R4_WINDOW_DAYS": 7,          # rolling window size for payment burst
    "R8_SIMILARITY_HARD_MIN": 0.55,   # below this, don't even consider as duplicate
    "R8_SAME_AREA_BONUS": 0.15,       # similarity bump if same constituency
}
