"""Script to compare two segmentation NIfTI files.

Can be utilized in testing.
"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.metrics import confusion_matrix

segmentations_dir = Path(
    "/u/69/taivait1/unix/workstation_home/cerebellum_data/data/final_segmentations"
)
file1 = segmentations_dir / "main.nii.gz"
file2 = segmentations_dir / "module.nii.gz"

# %%
img1 = nib.Nifti1Image.from_filename(file1)
img2 = nib.Nifti1Image.from_filename(file2)

assert img1.affine is not None, "Image 1 does not have an affine matrix."
assert img2.affine is not None, "Image 2 does not have an affine matrix."

# Verify shapes match
if img1.shape != img2.shape:
    print("Warning: Image dimensions do not match!")

# Verify physical space (affine matrices) match
if not np.allclose(img1.affine, img2.affine):
    print("Warning: Images are not in the same physical space!")

# %% Extract segmentation data
seg1 = np.asanyarray(img1.dataobj)
seg2 = np.asanyarray(img2.dataobj)

# %% Do basic comparison checks
if np.array_equal(seg1, seg2):
    print("The two segmentations are identical.")

# Create a boolean mask of where the arrays differ
mismatch_mask = seg1 != seg2

# Calculate totals
num_mismatches = np.sum(mismatch_mask)
total_voxels = seg1.size
percent_diff = (num_mismatches / total_voxels) * 100

print(f"Total mismatched voxels: {num_mismatches} out of {total_voxels}")
print(f"Percentage of volume that differs: {percent_diff:.4f}%")

# %% Per-label Dice coefficient calculation

# Find all unique labels across both images (ignoring 0/background if you want)
all_labels = np.unique(np.concatenate((seg1, seg2)))

print(
    f"{'Label':<10} | {'Dice Score':<12} | {'Vol 1 (voxels)':<15} | "
    f"{'Vol 2 (voxels)':<15}"
)
print("-" * 60)

for label in all_labels:
    # Isolate the current label
    mask1 = seg1 == label
    mask2 = seg2 == label

    intersection = np.logical_and(mask1, mask2).sum()
    vol1 = mask1.sum()
    vol2 = mask2.sum()

    if vol1 + vol2 == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection / (vol1 + vol2)

    print(f"{label:<10} | {dice:<12.4f} | {vol1:<15} | {vol2:<15}")

# %% Compute confusion matrix for the foreground labels

# Flatten the 3D arrays to 1D lists for the confusion matrix
seg1_flat = seg1.ravel()
seg2_flat = seg2.ravel()

# To avoid a massive table heavily skewed by the background, only look at
# voxels that are labeled as foreground in AT LEAST one of the images.
fg_mask = (seg1_flat != 0) | (seg2_flat != 0)

# Compute confusion matrix
labels_present = np.unique(np.concatenate((seg1_flat[fg_mask], seg2_flat[fg_mask])))
cm = confusion_matrix(seg1_flat[fg_mask], seg2_flat[fg_mask], labels=labels_present)
