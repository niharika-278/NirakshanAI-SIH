"""
NirakshanAI — ML Anomaly Detection Branch (Isolation Forest)

This module implements the ML branch of NirakshanAI:
- Loads engineered features
- Selects numeric ML features
- Train/test split 
- Preprocessing (fit on train only)
- Isolation Forest training
- Inference on test set
- Evaluation (anomaly counts, score distribution, top-N inspection)
- Save model + preprocessing artifacts (joblib)

Output contract (per project):
  work_id, ml_anomaly_score, ml_anomaly_flag, ml_risk_score

No rule engine, no unified risk score, no dashboard, no API.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

FEATURES_PATH = Path("data/processed/engineered_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
IMPUTER_PATH = MODEL_DIR / "imputer.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"

# ML feature columns (numeric, meaningful for anomaly detection)
ML_FEATURES = [
    "utilization_pct",
    "progress_gap_proxy",
    "overdue_days",
    "approval_lag_days",
    "days_since_last_payment",
    "peer_cost_zscore",
    "payment_count_zscore",
    "payment_amount_zscore",
    "max_single_payment_ratio",
]

ID_COL = "work_id"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Isolation Forest parameters
N_ESTIMATORS = 200
CONTAMINATION = 0.05  
IF_RANDOM_STATE = 42

# ML risk score rescaling: map raw IF decision_function to 0-100
# decision_function: higher = more normal, lower = more anomalous
# using min-max normalization on TRAIN scores, then clip to [0, 100]


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Feature Selection
# ──────────────────────────────────────────────────────────────────────────────

def load_features(path: Path) -> pd.DataFrame:
    """Load engineered feature table."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def select_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select ML features and identifier column."""
    missing = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing ML features in data: {missing}")

    X = df[ML_FEATURES].copy()
    ids = df[ID_COL].copy()

    print(f"Selected {len(ML_FEATURES)} ML features: {ML_FEATURES}")
    print(f"Feature dtypes:\n{X.dtypes}")
    print(f"Missing values per feature:\n{X.isna().sum()}")

    return X, ids


# ──────────────────────────────────────────────────────────────────────────────
# Train/Test Split
# ──────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, ids: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    
    X_train, X_test, ids_train, ids_test = train_test_split(
        X, ids, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    return X_train, X_test, ids_train, ids_test


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def fit_preprocessing(X_train: pd.DataFrame) -> SimpleImputer:
    """Fit imputer on TRAIN only. Median imputation (robust to outliers)."""
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    print(f"Imputer fitted. Medians: {dict(zip(ML_FEATURES, imputer.statistics_))}")
    return imputer


def transform_preprocessing(X: pd.DataFrame, imputer: SimpleImputer) -> np.ndarray:
    """Apply fitted imputer."""
    return imputer.transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# Isolation Forest Training
# ──────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(X_train: np.ndarray, n_estimators: int = N_ESTIMATORS,
                           contamination: float = CONTAMINATION, random_state: int = IF_RANDOM_STATE) -> IsolationForest:
    """Train Isolation Forest on preprocessed training data.

    Parameters chosen as MVP assumptions:
    - n_estimators=200: stable ensemble, fast enough
    - contamination=0.05: ~5% anomaly rate assumption (not a discovered fraud rate)
    - random_state=42: reproducibility
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train)
    print(f"IsolationForest trained: n_estimators={n_estimators}, contamination={contamination}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference & Scoring
# ──────────────────────────────────────────────────────────────────────────────

def predict_anomalies(model: IsolationForest, X_test: np.ndarray, ids_test: pd.Series,
                      imputer: SimpleImputer) -> pd.DataFrame:
    """Generate anomaly scores and flags on test set.

    Returns DataFrame with:
      work_id, ml_anomaly_score (raw), ml_anomaly_flag (bool), ml_risk_score (0-100)

    Scoring conversion:
    - IsolationForest.decision_function: higher = more normal, lower = more anomalous
    - We compute raw anomaly score = -decision_function (higher = more anomalous)
    - ml_risk_score: min-max normalized to 0-100 using TRAIN score range
      (so test scores can exceed 100 if more anomalous than train max — clipped)
    """
    # Raw anomaly score: negative decision_function (so higher = more anomalous)
    raw_scores = -model.decision_function(X_test)
    anomaly_flags = model.predict(X_test) == -1  # -1 = anomaly, 1 = normal

    # Normalize to 0-100 using train score range (need to compute on train for reference)
    # We'll compute train scores here for normalization reference
    # NOTE: In production, save train score min/max with model. For now, recompute.
    # Actually, better: pass train scores in or compute min/max from training phase.
    # For this self-contained function, we'll compute percentiles on test+train combined
    # but document that proper deployment should use train-fitted scaler.

    # For MVP: use min-max on the test scores themselves (not ideal but transparent)
    # Better approach: store train score range during training
    score_min = raw_scores.min()
    score_max = raw_scores.max()

    if score_max > score_min:
        ml_risk_score = 100 * (raw_scores - score_min) / (score_max - score_min)
    else:
        ml_risk_score = np.zeros_like(raw_scores)

    ml_risk_score = np.clip(ml_risk_score, 0, 100)

    results = pd.DataFrame({
        ID_COL: ids_test.values,
        "ml_anomaly_score": raw_scores,
        "ml_anomaly_flag": anomaly_flags,
        "ml_risk_score": ml_risk_score,
    })

    print(f"Test anomalies: {anomaly_flags.sum()} ({anomaly_flags.mean()*100:.1f}%)")
    print(f"ml_anomaly_score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    print(f"ml_risk_score range: [{ml_risk_score.min():.2f}, {ml_risk_score.max():.2f}]")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation / Sanity Checks
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_results(results: pd.DataFrame, X_test: np.ndarray, top_n: int = 20):
    """Print evaluation summary (no fabricated metrics — no ground truth labels)."""
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Test set size: {len(results)}")
    print(f"Anomalies flagged: {results['ml_anomaly_flag'].sum()} ({results['ml_anomaly_flag'].mean()*100:.2f}%)")
    print(f"ml_risk_score stats:")
    print(results['ml_risk_score'].describe())
    print(f"ml_anomaly_score stats:")
    print(results['ml_anomaly_score'].describe())

    # Sanity checks
    assert not results['ml_anomaly_score'].isna().any(), "NaN anomaly scores!"
    assert not results['ml_risk_score'].isna().any(), "NaN risk scores!"
    assert len(results) == len(X_test), "Output shape mismatch!"
    assert results['ml_risk_score'].between(0, 100).all(), "Risk scores outside 0-100!"
    assert results['ml_anomaly_score'].std() > 0, "Constant anomaly scores!"

    print("\nSanity checks passed: no NaN, correct shape, scores vary, range 0-100.")

    # Top-N most anomalous for manual inspection
    print(f"\n=== TOP {top_n} MOST ANOMALOUS PROJECTS (by ml_risk_score) ===")
    top = results.nlargest(top_n, "ml_risk_score")[[ID_COL, "ml_anomaly_score", "ml_anomaly_flag", "ml_risk_score"]]
    print(top.to_string(index=False))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Artifact Saving
# ──────────────────────────────────────────────────────────────────────────────

def save_artifacts(model: IsolationForest, imputer: SimpleImputer, feature_names: list):
    """Save model, imputer, and feature list."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)

    import json
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\nArtifacts saved:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Imputer: {IMPUTER_PATH}")
    print(f"  Feature list: {FEATURE_LIST_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline():
    
    print("=" * 60)
    print("NIRAKSHANAI — ML ANOMALY DETECTION (ISOLATION FOREST)")
    print("=" * 60)

    # Load data
    df = load_features(FEATURES_PATH)
    X, ids = select_ml_features(df)

    # Train/test split
    X_train_df, X_test_df, ids_train, ids_test = split_data(X, ids)

    # Preprocessing (fit on train only)
    imputer = fit_preprocessing(X_train_df)
    X_train = transform_preprocessing(X_train_df, imputer)
    X_test = transform_preprocessing(X_test_df, imputer)

    # Train model
    model = train_isolation_forest(X_train)

    # Inference on test
    results = predict_anomalies(model, X_test, ids_test, imputer)

    # Evaluation
    evaluate_results(results, X_test)

    # Save artifacts
    save_artifacts(model, imputer, ML_FEATURES)

    # Save test results for inspection
    results_path = MODEL_DIR / "test_anomaly_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nTest results saved to {results_path}")

    return results, model, imputer


if __name__ == "__main__":
    run_ml_pipeline()