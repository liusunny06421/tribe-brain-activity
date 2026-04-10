import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Load and merge data ---
brain = pd.read_csv("data/roi_features.csv")
metrics = pd.read_excel("data/reels_dataset.xlsx")

# Rename to match
video_map = {
    1: "v1_things i wish i avoided",
    2: "v2_ur sign not to be avg",
    3: "v3_first_two_weeks",
    4: "v4_how it feels taking 2h",
    5: "v5_im cooked",
    6: "v6_academic_comeback_yall",
    7: "v7_do it bored",
    8: "v8_last day of summer",
    9: "v9_they_said_i_cant",
    10: "v10_ai_teaches_better",
}
metrics["video"] = metrics["Video"].map(video_map)
merged = brain.merge(metrics, on="video")
print(f"Merged dataset: {merged.shape[0]} videos\n")

# --- Define engagement metrics to correlate ---
engagement_cols = ["Views", "Likes", "Comments", "Shares", "Saves",
                   "Interactions", "New Followers", "Skip Rate (%)", "Avg Watch Time (s)"]

# Filter to only columns that exist
engagement_cols = [c for c in engagement_cols if c in merged.columns]

# --- Get ROI names (mean activation only for main analysis) ---
mean_roi_cols = [c for c in brain.columns if c.startswith("mean_")]
var_roi_cols = [c for c in brain.columns if c.startswith("var_")]
peak_roi_cols = [c for c in brain.columns if c.startswith("peak_") and not c.startswith("peak_latency_")]

# --- Run correlations: mean activation vs each engagement metric ---
print("=" * 80)
print("TOP 10 BRAIN REGIONS CORRELATED WITH EACH METRIC")
print("=" * 80)

all_correlations = []

for metric in engagement_cols:
    correlations = []
    for roi_col in mean_roi_cols:
        roi_name = roi_col.replace("mean_", "")
        r, p = stats.pearsonr(merged[roi_col], merged[metric])
        correlations.append({"roi": roi_name, "r": r, "p": p, "metric": metric})

    corr_df = pd.DataFrame(correlations)
    corr_df["abs_r"] = corr_df["r"].abs()
    corr_df = corr_df.sort_values("abs_r", ascending=False)

    print(f"\n--- {metric} ---")
    print(f"{'ROI':<55} {'r':>8} {'p':>8}")
    print("-" * 73)
    for _, row in corr_df.head(10).iterrows():
        sig = "*" if row["p"] < 0.05 else " "
        print(f"{row['roi']:<55} {row['r']:>8.3f} {row['p']:>7.3f}{sig}")

    all_correlations.extend(correlations)

# --- Save full correlation table ---
all_corr_df = pd.DataFrame(all_correlations)
all_corr_df.to_csv("data/all_correlations.csv", index=False)
print(f"\nSaved full correlation table: data/all_correlations.csv")

# --- Global features vs engagement ---
print("\n" + "=" * 80)
print("GLOBAL BRAIN FEATURES VS ENGAGEMENT")
print("=" * 80)

global_cols = ["global_mean", "global_var", "global_peak"]
for metric in engagement_cols:
    print(f"\n--- {metric} ---")
    for gcol in global_cols:
        r, p = stats.pearsonr(merged[gcol], merged[metric])
        sig = "*" if p < 0.05 else " "
        print(f"  {gcol:<20} r={r:>7.3f}  p={p:>6.3f}{sig}")

# --- Summary: which ROIs appear in top 10 most often ---
print("\n" + "=" * 80)
print("MOST FREQUENTLY APPEARING ROIS ACROSS ALL METRICS")
print("=" * 80)

all_corr_df["abs_r"] = all_corr_df["r"].abs()
top_rois_per_metric = all_corr_df.groupby("metric").apply(
    lambda x: x.nlargest(10, "abs_r")["roi"].tolist()
)

from collections import Counter
roi_counts = Counter()
for rois in top_rois_per_metric:
    roi_counts.update(rois)

print(f"\n{'ROI':<55} {'Count':>6}")
print("-" * 63)
for roi, count in roi_counts.most_common(20):
    print(f"{roi:<55} {count:>6}")

# --- Network-level analysis ---
print("\n" + "=" * 80)
print("NETWORK-LEVEL CORRELATION WITH VIEWS")
print("=" * 80)

networks = {}
for roi_col in mean_roi_cols:
    roi_name = roi_col.replace("mean_", "")
    parts = roi_name.split("_")
    # Schaefer naming: 7Networks_LH_Vis_1 -> network is "Vis"
    for i, part in enumerate(parts):
        if part in ["LH", "RH"] and i + 1 < len(parts):
            network = parts[i + 1]
            if network not in networks:
                networks[network] = []
            networks[network].append(roi_col)
            break

for network, cols in sorted(networks.items()):
    network_mean = merged[cols].mean(axis=1)
    r, p = stats.pearsonr(network_mean, merged["Views"])
    sig = "*" if p < 0.05 else " "
    print(f"  {network:<20} ({len(cols):>3} ROIs)  r={r:>7.3f}  p={p:>6.3f}{sig}")
