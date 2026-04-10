import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.surface import vol_to_surf
import os
import glob

# --- Setup atlas ---
parcellation = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
fsaverage5 = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

labels_img = parcellation.maps
labels_lh = vol_to_surf(labels_img, fsaverage5.pial_left, interpolation='nearest_most_frequent')
labels_rh = vol_to_surf(labels_img, fsaverage5.pial_right, interpolation='nearest_most_frequent')
labels = np.concatenate([labels_lh, labels_rh])
labels = np.round(labels).astype(int)

roi_ids = sorted([r for r in np.unique(labels) if r != 0])
roi_label_names = parcellation.labels[1:]  # skip background

print(f"Atlas loaded: {len(roi_ids)} ROIs")

# --- Process all videos ---
video_files = sorted(glob.glob("data/*.npy"))
print(f"Found {len(video_files)} video files\n")

results = []

for filepath in video_files:
    name = os.path.basename(filepath).replace(".npy", "")
    preds = np.load(filepath)
    print(f"Processing: {name} | shape: {preds.shape}")

    # Extract per-ROI timeseries
    roi_timeseries = []
    for roi_id in roi_ids:
        mask = labels == roi_id
        roi_timeseries.append(preds[:, mask].mean(axis=1))
    roi_matrix = np.array(roi_timeseries)  # (400, n_timesteps)

    # Extract features per ROI
    mean_activation = roi_matrix.mean(axis=1)        # average over time
    temporal_variance = roi_matrix.var(axis=1)        # how much it fluctuates
    peak_activation = roi_matrix.max(axis=1)          # max activation
    peak_latency = roi_matrix.argmax(axis=1)          # when peak occurs (in seconds)

    # Store per-video summary
    video_result = {
        "video": name,
        "n_timesteps": preds.shape[0],
        "global_mean": mean_activation.mean(),
        "global_var": temporal_variance.mean(),
        "global_peak": peak_activation.max(),
    }

    # Add per-ROI mean activation
    for i, roi_id in enumerate(roi_ids):
        roi_name = roi_label_names[i].decode() if isinstance(roi_label_names[i], bytes) else str(roi_label_names[i])
        video_result[f"mean_{roi_name}"] = mean_activation[i]
        video_result[f"var_{roi_name}"] = temporal_variance[i]
        video_result[f"peak_{roi_name}"] = peak_activation[i]
        video_result[f"peak_latency_{roi_name}"] = peak_latency[i]

    results.append(video_result)

# --- Save results ---
df = pd.DataFrame(results)
df.to_csv("data/roi_features.csv", index=False)
print(f"\nSaved roi_features.csv with shape: {df.shape}")
print(f"Columns: {len(df.columns)} ({len(roi_ids)} ROIs × 4 features + 5 global)")
print(f"\nGlobal summary per video:")
print(df[["video", "n_timesteps", "global_mean", "global_var", "global_peak"]].to_string(index=False))
