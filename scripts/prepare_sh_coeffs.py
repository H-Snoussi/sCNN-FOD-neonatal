import nibabel as nib
import numpy as np
import torch
import os
import random
import scipy.special
from dipy.reconst.shm import sph_harm_ind_list, real_sph_harm
from scnn.sh import spherical_harmonic
BATCH_SIZE = int(1e4)


###############################
####### TRAINING DATA #########
###############################

def compute_sh_basis(bvecs, l_max=8):
    """Compute real symmetric SH basis with proper coefficient ordering."""
    thetas = np.arccos(np.clip(bvecs[:, 2], -1.0, 1.0))
    phis = np.mod(np.arctan2(bvecs[:, 1], bvecs[:, 0]) + 2 * np.pi, 2 * np.pi)

    # Calculate number of coefficients per degree
    degrees = list(range(0, l_max + 1, 2))
    n_coeffs = sum(2 * l + 1 for l in degrees)
    basis = np.zeros((len(bvecs), n_coeffs), dtype=np.float32)

    # Create index mapping
    idx = 0
    for l in degrees:
        for m in range(-l, l + 1):
            basis[:, idx] = spherical_harmonic(l, m, thetas, phis).numpy()
            idx += 1

    return basis


def extract_signals(subject_range, output_path, masking=True):
    """Extract SH coefficients for each shell (400, 1000, 2600) and save as 3 channels."""
    all_sh_coeffs = []

    for i in subject_range:
        sub_id = f"sub-{i:02}"

        # Load white matter mask
        wm_mask_path = f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/wmfod.nii.gz"
        if not os.path.exists(wm_mask_path):
            print(f"Skipping {sub_id}: No white matter FODs found")
            continue
        wm_fod = nib.load(wm_mask_path).get_fdata()
        wm_mask = (np.any(wm_fod, axis=-1) & np.all(np.isfinite(wm_fod), axis=-1) & (wm_fod[..., 0] > 0)).astype(bool)


        type_of_data="part"

        if type_of_data == "full":
            dwi_file="dwi_norm.nii.gz"
            bvecs_file="dwi.bvec"
            bvals_file="dwi.bval"

        elif type_of_data == "part":
            dwi_file="dwi_norm_selected.nii.gz"
            bvecs_file="selected_bvecs.txt"
            bvals_file="selected_bvals.txt"



        # Load DWI data
        dwi_path = f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/{dwi_file}"
        if not os.path.exists(dwi_path):
            print(f"Skipping {sub_id}: No DWI data found")
            continue
        dwi_data = nib.load(dwi_path).get_fdata()

        # Load bvals and bvecs : bvecs and bvals are similar accorss all subjects
        bvals = np.loadtxt(f"/local/projects/scnn-investigation_haykel/protocols/sub-07/{bvals_file}")[:dwi_data.shape[3]]
        bvecs = np.loadtxt(f"/local/projects/scnn-investigation_haykel/protocols/sub-07/{bvecs_file}")[:, :dwi_data.shape[3]]
        bvecs = bvecs.T  # Shape: [n_volumes, 3]
        # bvecs[:, :2] *= -1 


        # Target shells (400, 1000, 2600)
        shells = [400, 1000, 2600]
        tolerance = 50.0
        shell_indices = {shell: np.where(np.isclose(bvals, shell, atol=tolerance))[0] for shell in shells}

        # Check if all shells have data
        missing_shells = [shell for shell in shells if len(shell_indices[shell]) == 0]
        if missing_shells:
            print(f"Skipping {sub_id}: Missing shells {missing_shells}")
            continue

        # Apply WM mask to DWI data
        if masking:
            dwi_wm = dwi_data[wm_mask]  # Shape: [n_voxels, n_volumes]
            
        else:
            dwi_wm = dwi_data.reshape(-1, dwi_data.shape[3])

        print("dwi_wm.shape",dwi_wm.shape)
        # Initialize SH coefficients tensor (n_voxels, 3 shells, 45 coefficients)
        n_voxels = dwi_wm.shape[0]
        sh_coeffs = np.zeros((n_voxels, 3, 45))

        for shell_idx, shell in enumerate(shells):
            idx = shell_indices[shell]
            shell_bvecs = bvecs[idx, :]  # Shape: [n_directions, 3]
            shell_signals = dwi_wm[:, idx]  # Shape: [n_voxels, n_directions]

            # Compute SH basis matrix for this shell
            B = compute_sh_basis(shell_bvecs, l_max=8)  # Shape: [n_directions, 45]

            # Fit SH coefficients using least squares (vectorized)
            try:
                C = np.linalg.lstsq(B, shell_signals.T, rcond=None)[0].T  # Shape: [n_voxels, 45]
                sh_coeffs[:, shell_idx, :] = C
            except np.linalg.LinAlgError:
                print(f"Error fitting SH coefficients for {sub_id}, shell {shell}")
                continue

            
        # for shell_idx in range(3):
            # Step 1: Shell-wise Normalization (Normalize Each Shell Independently)
            # sh_coeffs[:, shell_idx, :] /= np.linalg.norm(sh_coeffs[:, shell_idx, :], axis=-1, keepdims=True) + 1e-8

        # Normalize SH coefficients across all shells together (not separately)
        # sh_coeffs /= np.linalg.norm(sh_coeffs, axis=(-1, -2), keepdims=True) + 1e-8


            # # Step 2: Normalize the First SH Coefficient (Intensity Normalization)
            # sh_coeffs[..., 0] /= sh_coeffs[..., 0].max()
            # # Normalize the first SH coefficient by a robust percentile instead of max()
            # sh_coeffs[..., 0] /= np.percentile(sh_coeffs[..., 0], 99)


        all_sh_coeffs.append(sh_coeffs)

    # Concatenate and save
    if all_sh_coeffs:
        all_sh_coeffs = np.concatenate(all_sh_coeffs, axis=0)
        print(f"Final saved SH coefficients: {all_sh_coeffs.shape[0]} voxels")
        torch.save(torch.Tensor(all_sh_coeffs).float(), output_path)
    else:
        print(f"No valid SH coefficients found for {output_path}, skipping save.")



# Rest of the code (seed, subject splitting, etc.) remains unchanged
random.seed(117)
all_subjects = list(range(1, 44))
random.shuffle(all_subjects)

train_subjects = all_subjects[:35]
test_subjects = all_subjects[35:39]
val_subjects = all_subjects[39:43]

test_sub1 = all_subjects[1:2]
test_sub2 = all_subjects[2:3]
test_sub3 = all_subjects[3:4]
test_sub4 = all_subjects[4:5]

test_sub35 = all_subjects[35:36]
test_sub36 = all_subjects[36:37]
test_sub37 = all_subjects[37:38]
test_sub38 = all_subjects[38:39]

# train_subjects = all_subjects[:3]
# test_subjects = all_subjects[40:41]
# val_subjects = all_subjects[16:17]

# extract_signals(train_subjects, "/local/projects/scnn-investigation_haykel/genfiles/training_sh_for_sCNN_30percent.pt")
# extract_signals(test_subjects, "/local/projects/scnn-investigation_haykel/genfiles/test_sh_for_sCNN_30percent.pt")
# extract_signals(val_subjects, "/local/projects/scnn-investigation_haykel/genfiles/validation_sh_for_sCNN_30percent.pt")

extract_signals(test_sub1, "/local/projects/scnn-investigation_haykel/genfiles/test_sh_for_sCNN_sub-01_30percent_data.pt",masking=False)
extract_signals(test_sub2, "/local/projects/scnn-investigation_haykel/genfiles/test_sh_for_sCNN_sub-02_30percent_data.pt",masking=False)
extract_signals(test_sub3, "/local/projects/scnn-investigation_haykel/genfiles/test_sh_for_sCNN_sub-03_30percent_data.pt",masking=False)
extract_signals(test_sub4, "/local/projects/scnn-investigation_haykel/genfiles/test_sh_for_sCNN_sub-04_30percent_data.pt",masking=False)