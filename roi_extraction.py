import numpy as np
from nilearn import datasets

# Load surface-based Schaefer parcellation directly for fsaverage5
parcellation = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
fsaverage5 = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

# Use nilearn's surface parcellation
from nilearn.surface import vol_to_surf

labels_img = parcellation.maps
labels_lh = vol_to_surf(labels_img, fsaverage5.pial_left, interpolation='nearest_most_frequent')
labels_rh = vol_to_surf(labels_img, fsaverage5.pial_right, interpolation='nearest_most_frequent')
labels = np.concatenate([labels_lh, labels_rh])
labels = np.round(labels).astype(int)

preds = np.load("data/v3_first_two_weeks.npy")
print(f"Preds shape: {preds.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique ROIs: {len(np.unique(labels))}")

# Extract mean activation per ROI
roi_means = []
roi_names = []
for roi_id in np.unique(labels):
    if roi_id == 0:
        continue
    mask = labels == roi_id
    roi_means.append(preds[:, mask].mean(axis=1))
    roi_names.append(roi_id)

roi_matrix = np.array(roi_means).T
print(f"ROI matrix shape: {roi_matrix.shape}")
print(f"Sample ROI names: {roi_names[:5]}")
