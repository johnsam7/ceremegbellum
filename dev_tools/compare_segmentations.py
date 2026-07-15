import nibabel as nib
import numpy as np

file1 = "segmentation_pipeline_A.nii.gz"
file2 = "segmentation_pipeline_B.nii.gz"

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


def calculate_dice(img1_data, img2_data):
    # Ensure both arrays are boolean/binary masks
    mask1 = img1_data > 0
    mask2 = img2_data > 0

    # Calculate intersection and total volumes
    intersection = np.logical_and(mask1, mask2)
    volume1 = mask1.sum()
    volume2 = mask2.sum()

    # Compute Dice Similarity Coefficient
    if volume1 + volume2 == 0:
        return 1.0  # Both masks are completely empty

    dice_score = 2.0 * intersection.sum() / (volume1 + volume2)
    return dice_score


data1 = img1.get_fdata()
data2 = img2.get_fdata()

print(f"Dice Similarity Coefficient: {calculate_dice(data1, data2):.4f}")
