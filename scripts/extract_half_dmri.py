import numpy as np
import nibabel as nib


for i in range(1,44):
    sub_id = f"sub-{i:02}"

    # Input file paths
    input_dwi = f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/dwi_norm.nii.gz"
    # the following is correct
    indices_file = f"/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_indices.txt"

    # Output file path
    output_dwi = f"/local/projects/scnn-investigation_haykel/protocols/{sub_id}/dwi_norm_selected.nii.gz"

    # Load selected indices
    selected_indices = np.loadtxt(indices_file, dtype=int)

    # Load the dMRI data
    dwi_img = nib.load(input_dwi)
    dwi_data = dwi_img.get_fdata()

    # Ensure the selected indices are within range
    if np.any(selected_indices >= dwi_data.shape[3]):
        raise ValueError("Error: Some selected indices exceed the number of available volumes.")

    # Extract selected volumes
    selected_dwi_data = dwi_data[..., selected_indices]

    # Save the new dMRI file
    new_img = nib.Nifti1Image(selected_dwi_data, affine=dwi_img.affine, header=dwi_img.header)
    nib.save(new_img, output_dwi)

    print(f"New dMRI file saved: {output_dwi}")
