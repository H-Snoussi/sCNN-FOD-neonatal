#/bin/bash

conda activate scnn_env

cd /local/projects/scnn-investigation_haykel

bash scripts/estimate_fod_coefficients.sh
# ==> dwi.nii.gz, dwi.bval, dwi.bvec, brain_mask.nii.gz
# ==> dwi_norm.nii.gz
# ==> wm_response.txt; wmfod.nii.gz
# ==> gm_response.txt; gm.nii.gz 
# ==> csf_response.txt; csf.nii.gz


# NO LONGER REQUIRED for sCNN, but for MLP
# python scripts/no_longer_required/prepare_dwi_signals.py
# ==> training_signals.pt
# ==> validation_signals.pt
# ==> test_signals.pt



# Target
python scripts/prepare_fod_coefficients.py
# ==> training_odfs_sh.pt
# ==> test_odfs_sh.pt
# ==> validation_odfs_sh.pt

# Training
python scripts/prepare_sh_coeffs.py
# training_sh_for_sCNN.pt
# test_sh_for_sCNN.pt
# validation_sh_for_sCNN.pt







# preparing data for testing
python scripts/prepare_sh_coeffs.py


# probably no longer useful
python scripts/prepare_dwi_for_testing.py
