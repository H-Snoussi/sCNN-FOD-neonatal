import nibabel as nib
import numpy as np
import torch
import os
from collections import OrderedDict
# from scnn.models import AdvancedMultiShellSCNN
from scnn.models import SCNNModel_3shells_with_attention as AdvancedMultiShellSCNN

# Paths
MODEL_PATH = "/local/projects/scnn-investigation_haykel/notebooks/best_model_march11.pth"
# INPUT_PATH = "/local/projects/scnn-investigation_haykel/genfiles/test_sh_for_sCNN_sub-04.pt"
INPUT_PATH = "/local/projects/scnn-investigation_haykel/genfiles/test_sh_for_sCNN_sub-04_30percent_data.pt"

AFFINE_PATH = "/local/projects/scnn-investigation_haykel/protocols/sub-04/wmfod.nii.gz"  # Adjust path if needed
MASK_PATH = "/local/projects/scnn-investigation_haykel/protocols/sub-04/brain_mask.nii.gz"
# OUTPUT_DIR = "/local/projects/scnn-investigation_haykel/fod_predictions/"
OUTPUT_DIR = "/local/projects/scnn-investigation_haykel/protocols/sub-04/"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = AdvancedMultiShellSCNN(c_in=3, n_out=45).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)


for key, value in state_dict.items():
    print(f"{key}: {value.shape}")

# Handle DataParallel-trained models
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # Remove DataParallel wrapping
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict, strict=True)

model.eval()
print(" Model loaded successfully.")

print(model)
# Load SH coefficients (Test data) for sub-04
sh_data = torch.load(INPUT_PATH).to(device)  # Shape: (N_voxels, 3, 45)
n_voxels = sh_data.shape[0]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run inference in batches
BATCH_SIZE = 1044
fod_predictions = torch.zeros((n_voxels, 45), dtype=torch.float32, device=device)




with torch.no_grad():
    for j in range(0, n_voxels, BATCH_SIZE):
        batch = sh_data[j:j + BATCH_SIZE].to(device)
        # Ensure batch has correct dimensions [B, 3, 45]
        if batch.dim() == 2:
            batch = batch.unsqueeze(1)
        batch_output = model(batch)
        # print(f"Debug: Model output shape: {batch_output.shape}")  # Should be [B, 45] but appears to be [B, 90]
        # fod_predictions[j:j + BATCH_SIZE] = model(batch)
        fod_predictions[j:j + BATCH_SIZE] = model(batch)[:, :45]  # Keep only the first 45 coefficients


# Move predictions to CPU and convert to NumPy
fod_predictions = fod_predictions.cpu().numpy()

# Load affine from corresponding NIfTI file
sh_nifti = nib.load(AFFINE_PATH)  # Load reference affine
mask=nib.load(MASK_PATH).get_fdata()

print(f"🔍 Debug: fod_predictions shape before reshape: {fod_predictions.shape}")
print(f"🔍 Debug: Expected shape based on reference image: {sh_nifti.shape[:3] + (45,)}")

# Ensure voxel count matches
if np.prod(sh_nifti.shape[:3]) == n_voxels:
    # Reshape back to 4D (X, Y, Z, 45)
    fod_predictions = fod_predictions.reshape(sh_nifti.shape[:3] + (45,))
else:
    print(f" Warning: Voxel mismatch! Expected {np.prod(sh_nifti.shape[:3])}, but got {n_voxels}.")


mask = mask.astype(bool)

fod_predictions *= mask[..., np.newaxis]
# Save as NIfTI
output_path = os.path.join(OUTPUT_DIR, "sub-04_fod_sCNN_model_march11_30percent_data.nii.gz")
nifti_output = nib.Nifti1Image(fod_predictions, sh_nifti.affine, header=sh_nifti.header)
nib.save(nifti_output, output_path)

print(f" Saved FOD prediction for sub-04 -> {output_path}")
print(" Inference complete for sub-04!")


