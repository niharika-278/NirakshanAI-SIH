"""
NirakshanAI — District-wise Risk Heatmap

Generates a static district-level choropleth map of aggregated risk scores.
Reads from data/results.json (produced by risk_aggregation.py) and
risk heatmaps/district-level shapefile/IND_adm2.shp (district boundaries).

Output: risk heatmaps/district_risk_heatmap.png
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Get project root 
PROJECT_ROOT = Path(__file__).parent.parent

RESULTS_PATH = PROJECT_ROOT / "data" / "results.json"
SHAPEFILE_PATH = PROJECT_ROOT / "risk heatmaps" / "shapefiles" / "India-map" / "IND_adm2.shp"
OUTPUT_PATH = PROJECT_ROOT / "risk heatmaps" / "district_risk_heatmap.png"

# Only include the 10 states present in our prototype
TARGET_STATES = [
    "Bihar", "Chandigarh", "Delhi", "Gujarat", "Madhya Pradesh",
    "Maharashtra", "Punjab", "Rajasthan", "Uttarakhand", "Uttar Pradesh"
]

# Normalization for district name matching
def normalize_district(name: str) -> str:
    """Normalize district name for fuzzy matching."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace("-", " ").replace(".", "").replace("  ", " ")


# ──────────────────────────────────────────────────────────────────────────────
# Load and prepare data
# ──────────────────────────────────────────────────────────────────────────────

def load_risk_data() -> pd.DataFrame:
    """Load risk scores from results.json and compute district-level averages."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["projects"])

    # Keep only relevant columns
    df = df[["work_id", "state", "district", "risk_score"]].copy()

    # Exclude rows with missing district or risk_score
    before = len(df)
    df = df.dropna(subset=["district", "risk_score"])
    after = len(df)
    if before != after:
        print(f"Excluded {before - after} rows with missing district or risk_score")

    # Normalize district names for matching
    df["district_norm"] = df["district"].apply(normalize_district)

    # Aggregate: average risk_score per district
    district_risk = df.groupby(["state", "district", "district_norm"], as_index=False).agg(
        avg_risk_score=("risk_score", "mean"),
        project_count=("work_id", "count")
    )

    print(f"District aggregates: {len(district_risk)} districts from {df['work_id'].nunique()} projects")
    return district_risk


def load_boundaries() -> gpd.GeoDataFrame:
    """Load district boundary shapefile and normalize district names."""
    gdf = gpd.read_file(SHAPEFILE_PATH)

    # Filter to target states only
    gdf = gdf[gdf["NAME_1"].isin(TARGET_STATES)].copy()

    # Normalize district names for matching
    gdf["district_norm"] = gdf["NAME_2"].apply(normalize_district)

    print(f"Boundary districts (target states): {len(gdf)}")
    return gdf


# ──────────────────────────────────────────────────────────────────────────────
# Match and merge
# ──────────────────────────────────────────────────────────────────────────────

def match_and_merge(district_risk: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge risk data with boundaries on normalized district name + state."""
    # Merge on state + normalized district
    merged = boundaries.merge(
        district_risk,
        left_on=["NAME_1", "district_norm"],
        right_on=["state", "district_norm"],
        how="left"
    )

    # Report match statistics
    total_boundary = len(merged)
    matched = merged["avg_risk_score"].notna().sum()
    unmatched = total_boundary - matched
    print(f"Boundary districts in target states: {total_boundary}")
    print(f"Matched with risk data: {matched}")
    print(f"Unmatched (no risk data): {unmatched}")

    # Also report data districts that didn't match any boundary
    data_districts = set(district_risk["district_norm"])
    boundary_districts = set(merged["district_norm"].dropna())
    data_unmatched = data_districts - boundary_districts
    print(f"Data districts with no boundary match: {len(data_unmatched)}")
    if data_unmatched:
        print("  Unmatched data districts:", ", ".join(sorted(data_unmatched)))

    # Fill unmatched with NaN for plotting (will appear as no-data)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Plot heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap(merged_gdf: gpd.GeoDataFrame):
    """Generate and save the district risk heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Plot districts with risk scores
    merged_gdf.plot(
        column="avg_risk_score",
        cmap="YlOrRd",
        linewidth=0.3,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "label": "Average Risk Score (0-100)",
            "orientation": "vertical",
            "shrink": 0.7,
            "pad": 0.02,
        },
        missing_kwds={
            "color": "#f0f0f0",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data",
        },
        ax=ax
    )

    # Plot state boundaries on top for clarity
    state_boundaries = merged_gdf.dissolve(by="NAME_1")
    state_boundaries.boundary.plot(
        ax=ax,
        linewidth=0.8,
        edgecolor="black"
    )

    # Title and formatting
    ax.set_title(
        "NirakshanAI — District-wise Average Risk Score (MPLADS)",
        fontsize=16,
        fontweight="bold",
        pad=20
    )
    ax.set_axis_off()

    # Add note
    plt.figtext(
        0.5, 0.02,
        "Risk levels: 0-30 LOW | 31-60 MEDIUM | 61-80 HIGH | 81-100 CRITICAL\n"
        "Based on unified rule+ML risk aggregation across 41,086 projects",
        ha="center",
        fontsize=10,
        color="gray"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to {OUTPUT_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NIRAKSHANAI — DISTRICT RISK HEATMAP GENERATION")
    print("=" * 60)

    # Load risk data
    print("\n[1/4] Loading risk scores...")
    district_risk = load_risk_data()

    # Load boundaries
    print("\n[2/4] Loading district boundaries...")
    boundaries = load_boundaries()

    # Match and merge
    print("\n[3/4] Matching districts...")
    merged = match_and_merge(district_risk, boundaries)

    # Plot
    print("\n[4/4] Generating heatmap...")
    plot_heatmap(merged)

    print("\nDone!")


if __name__ == "__main__":
    main()