import nibabel as nib
import numpy as np
import os
import scipy.special
from dipy.reconst.shm import sph_harm_ind_list, real_sph_harm
from scnn.sh import spherical_harmonic

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

def convert_dwi_to_sh(sub_id, output_nifti):
    """Convert diffusion MRI to spherical harmonic (SH) coefficients and save as NIfTI."""

    type_of_data="full"

    if type_of_data == "full":

        dwi_path = f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/dwi_norm.nii.gz"
        bvals = np.loadtxt(f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/dwi.bval")
        bvecs = np.loadtxt(f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/dwi.bvec").T  # Shape: (n_volumes, 3)

    elif type_of_data == "half":
        dwi_path = f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/dwi_norm_selected.nii.gz"
        bvals = np.loadtxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_bvals.txt")
        bvecs = np.loadtxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_bvecs.txt").T

    bvecs[:, 1] *= -1
    bvecs[:, 2] *= -1


    # bvecs[:, 0] *= -1
    # bvecs[:, 2] *= -1


    # doesnt work
    # bvecs[:, :2] *= -1
    # bvecs[:, 2] *= -1

    # Load DWI data
    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()
    affine = dwi_img.affine  # Preserve original spatial mapping

    # Define shells (b-values) of interest
    shells = [400, 1000, 2600]
    tolerance = 50.0
    shell_indices = {shell: np.where(np.isclose(bvals, shell, atol=tolerance))[0] for shell in shells}

    # Get spatial shape
    X, Y, Z, _ = dwi_data.shape

    # Prepare SH coefficient storage
    sh_coeffs = np.zeros((X, Y, Z, 3, 45), dtype=np.float32)  # (X, Y, Z, Shells, SH)

    # Process each shell
    for shell_idx, shell in enumerate(shells):
        indices = shell_indices[shell]
        shell_bvecs = bvecs[indices, :]  # (N_directions, 3)
        shell_dwi = dwi_data[..., indices]  # (X, Y, Z, N_directions)

        # Compute SH basis
        B = compute_sh_basis(shell_bvecs, l_max=8)  # (N_directions, 45)

        # Reshape for least squares fitting
        shell_dwi_flat = shell_dwi.reshape(-1, shell_dwi.shape[-1])  # (N_voxels, N_directions)

        # Fit SH coefficients using least squares (vectorized)
        C = np.linalg.lstsq(B, shell_dwi_flat.T, rcond=None)[0].T  # (N_voxels, 45)

        # Reshape back into original spatial shape
        sh_coeffs[..., shell_idx, :] = C.reshape(X, Y, Z, 45)  # (X, Y, Z, 3, 45)

    # Save as NIfTI
    sh_nifti = nib.Nifti1Image(sh_coeffs, affine)
    nib.save(sh_nifti, output_nifti)
    print(f"Saved SH coefficients for {sub_id} as {output_nifti}")

# Process **ALL SUBJECTS**
all_subjects = [f"sub-{i:02}" for i in range(1, 44)]  # Subjects 01 to 43
for sub_id in all_subjects:
    convert_dwi_to_sh(sub_id, f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/sh_coeffs_newcon.nii.gz")

