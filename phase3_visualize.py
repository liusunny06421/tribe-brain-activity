import numpy as np
import pandas as pd
from scipy import stats
from nilearn import datasets, plotting
from nilearn.surface import vol_to_surf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- Load data ---
brain = pd.read_csv("data/roi_features.csv")
metrics = pd.read_excel("data/reels_dataset.xlsx")
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

# --- Setup atlas ---
parcellation = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
fsaverage5 = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

labels_img = parcellation.maps
labels_lh = vol_to_surf(labels_img, fsaverage5.pial_left, interpolation='nearest_most_frequent')
labels_rh = vol_to_surf(labels_img, fsaverage5.pial_right, interpolation='nearest_most_frequent')
labels = np.concatenate([labels_lh, labels_rh])
labels = np.round(labels).astype(int)

roi_ids = sorted([r for r in np.unique(labels) if r != 0])
roi_label_names = parcellation.labels[1:]
mean_roi_cols = [c for c in brain.columns if c.startswith("mean_")]

# --- Compute correlation of each ROI with Views ---
target_metric = "Views"
corr_map = np.zeros(len(labels))

for i, roi_id in enumerate(roi_ids):
    roi_col = mean_roi_cols[i]
    r, p = stats.pearsonr(merged[roi_col], merged[target_metric])
    mask = labels == roi_id
    corr_map[mask] = r

# Split into hemispheres
n_lh = len(labels_lh)
corr_lh = corr_map[:n_lh]
corr_rh = corr_map[n_lh:]

# --- Plot brain surface ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={'projection': '3d'})
fig.suptitle(f"Brain ROI Correlation with {target_metric}", fontsize=16, fontweight='bold')

views = [('lateral', 'left'), ('medial', 'left'), ('lateral', 'right'), ('medial', 'right')]
titles = ['Left Lateral', 'Left Medial', 'Right Lateral', 'Right Medial']

for idx, (view, hemi) in enumerate(views):
    ax = axes[idx // 2][idx % 2]
    surf_data = corr_lh if hemi == 'left' else corr_rh
    surf_mesh = fsaverage5.pial_left if hemi == 'left' else fsaverage5.pial_right

    plotting.plot_surf_stat_map(
        surf_mesh,
        stat_map=surf_data,
        hemi=hemi,
        view=view,
        colorbar=idx == 1,
        cmap='RdBu_r',
        vmax=1.0,
        threshold=0.3,
        title=titles[idx],
        axes=ax,
        figure=fig
    )

plt.tight_layout()
plt.savefig("data/brain_correlation_views.png", dpi=150, bbox_inches='tight')
print("Saved: data/brain_correlation_views.png")

# --- Bar chart: top 20 ROIs correlated with Views ---
correlations = []
for i, roi_id in enumerate(roi_ids):
    roi_col = mean_roi_cols[i]
    roi_name = roi_label_names[i].decode() if isinstance(roi_label_names[i], bytes) else str(roi_label_names[i])
    r, p = stats.pearsonr(merged[roi_col], merged[target_metric])
    correlations.append({"roi": roi_name, "r": r, "p": p})

corr_df = pd.DataFrame(correlations)
corr_df["abs_r"] = corr_df["r"].abs()
top20 = corr_df.nlargest(20, "abs_r")

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#e74c3c' if r > 0 else '#3498db' for r in top20["r"]]
ax.barh(range(len(top20)), top20["r"].values, color=colors)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["roi"].values, fontsize=9)
ax.set_xlabel("Pearson r", fontsize=12)
ax.set_title(f"Top 20 Brain Regions Correlated with {target_metric}", fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.invert_yaxis()

for i, (r, p) in enumerate(zip(top20["r"], top20["p"])):
    sig = "*" if p < 0.05 else ""
    ax.text(r + 0.02 if r > 0 else r - 0.02, i, f"{r:.2f}{sig}",
            va='center', ha='left' if r > 0 else 'right', fontsize=8)

plt.tight_layout()
plt.savefig("data/top20_rois_views.png", dpi=150, bbox_inches='tight')
print("Saved: data/top20_rois_views.png")

# --- Scatter: global mean activation vs views ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, gcol in enumerate(["global_mean", "global_var", "global_peak"]):
    ax = axes[idx]
    ax.scatter(merged[gcol], merged["Views"], s=80, alpha=0.7, edgecolors='black')
    r, p = stats.pearsonr(merged[gcol], merged["Views"])
    z = np.polyfit(merged[gcol], merged["Views"], 1)
    x_line = np.linspace(merged[gcol].min(), merged[gcol].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5)
    ax.set_xlabel(gcol, fontsize=11)
    ax.set_ylabel("Views", fontsize=11)
    ax.set_title(f"{gcol} vs Views\nr={r:.3f}, p={p:.3f}", fontsize=11)

    for _, row in merged.iterrows():
        ax.annotate(row["video"].replace("v", "").split("_")[0],
                    (row[gcol], row["Views"]), fontsize=7, alpha=0.7)

plt.tight_layout()
plt.savefig("data/global_vs_views.png", dpi=150, bbox_inches='tight')
print("Saved: data/global_vs_views.png")

plt.show()
