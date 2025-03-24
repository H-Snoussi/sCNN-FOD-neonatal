import healpy as hp
import nibabel as nib
import numpy as np
import torch
from scnn.sh import spherical_harmonic
import os
import random

BATCH_SIZE = int(1e4)

#############################
####### TARGET DATA #########
#############################
def extract_odfs(subject_range, output_path):
    """Extract and save ODFs for a given subject range, ensuring alignment with DWI signals."""
    all_odfs = []

    for i in subject_range:
        sub_id = f"sub-{i:02}"

        # Load ODFs
        odfs_path = f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/wmfod.nii.gz"
        if not os.path.exists(odfs_path):
            print(f"Skipping {sub_id}: No wmfod.nii.gz found")
            continue
        odfs_sh = nib.load(odfs_path).get_fdata()

        # Get the mask for valid ODFs (exclude zero or NaN ODFs)
        odfs_mask = (np.any(odfs_sh, axis=-1) & np.all(np.isfinite(odfs_sh), axis=-1) & (odfs_sh[..., 0] > 0))

        # Apply mask
        odfs_sh = odfs_sh[odfs_mask]

        # Normalize ODFs
        # odfs_sh = (odfs_sh / odfs_sh[:, 0:1]) / (4 * np.pi * spherical_harmonic(0, 0, [0], [0]).item())

        # Process in batches
        for j in range(0, len(odfs_sh), BATCH_SIZE):
            idx = np.arange(j, min(j + BATCH_SIZE, len(odfs_sh)))
            batch_odfs_sh = odfs_sh[idx]

            all_odfs.append(batch_odfs_sh)

    # Concatenate all collected ODFs and save
    all_odfs = np.concatenate(all_odfs, axis=0)
    print(f"Saved {len(all_odfs)} ODFs to {output_path}")
    torch.save(torch.Tensor(all_odfs).float(), output_path)


# Set a fixed seed to ensure reproducibility
RANDOM_SEED = 117
random.seed(RANDOM_SEED)  

# Define all subjects
all_subjects = list(range(1, 44))  # Subjects 1 to 43

# Shuffle randomly
random.shuffle(all_subjects)

print(all_subjects)



train_subjects = all_subjects[:35]
test_subjects = all_subjects[35:39]
val_subjects = all_subjects[39:43]

# train_subjects = all_subjects[:3]
# test_subjects = all_subjects[40:41]
# val_subjects = all_subjects[16:17]

print(f"Train: {train_subjects}")
print(f"Validation: {val_subjects}")
print(f"Test: {test_subjects}")


# Generate datasets
extract_odfs(train_subjects, "/local/projects/scnn-investigation_haykel/genfiles/training_odfs_sh.pt")
extract_odfs(test_subjects, "/local/projects/scnn-investigation_haykel/genfiles/test_odfs_sh.pt")
extract_odfs(val_subjects, "/local/projects/scnn-investigation_haykel/genfiles/validation_odfs_sh.pt")

# extract_odfs(range(1, 36), "/local/projects/scnn-investigation_haykel/genfiles/training_odfs_sh.pt")
# extract_odfs(range(36, 40), "/local/projects/scnn-investigation_haykel/genfiles/test_odfs_sh.pt")
# extract_odfs(range(40, 44), "/local/projects/scnn-investigation_haykel/genfiles/validation_odfs_sh.pt")


# Train: [37, 7, 18, 29, 17, 28, 4, 9, 35, 43, 31, 30, 10, 24, 38, 19, 32, 1, 20, 33, 13, 25, 6, 3, 40, 42, 5, 2, 21, 39, 15, 23, 22, 34, 8]
# Validation: [14, 11, 12, 16]
# Test: [27, 36, 41, 26]
