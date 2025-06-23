#/bin/bash

OUT_DIR="/local/projects/scnn-investigation_haykel/protocols/sub-36/"
MASK_PATH="/local/projects/scnn-investigation_haykel/protocols/sub-36/brain_mask.nii.gz"
FIVE_TT_MIF="/local/projects/scnn-investigation_haykel/protocols/sub-36/t2w/5tt_from_drawem2.nii.gz"


# MSMT-CSD
FOD_PATH="/local/projects/scnn-investigation_haykel/protocols/sub-36/wmfod2.nii.gz"
tckgen "$FOD_PATH" \
     "${OUT_DIR}/wmfod_tracks_june20_5tt.tck" \
     -algorithm iFOD2 \
     -seed_dynamic "$FOD_PATH" \
     -mask "$MASK_PATH" \
     -select 100000 \
     -cutoff 0.001 \
     -step 0.4 \
     -angle 20 \
     -minlength 5 \
     -maxlength 300 \
     -act "$FIVE_TT_MIF" \
     -backtrack \
     -crop_at_gmwmi \
     -force


# sCNN
FOD_PATH="/local/projects/scnn-investigation_haykel/protocols/sub-36/sub-36_fod_sCNN_model_june19_30percent_data.nii.gz"
tckgen "$FOD_PATH" \
     "${OUT_DIR}/scnn_tracks_june20_5tt.tck" \
     -algorithm iFOD2 \
     -seed_dynamic "$FOD_PATH" \
     -mask "$MASK_PATH" \
     -select 100000 \
     -cutoff 0.001 \
     -step 0.4 \
     -angle 20 \
     -minlength 5 \
     -maxlength 300 \
     -act "$FIVE_TT_MIF" \
     -backtrack \
     -crop_at_gmwmi \
     -force


# MLP
FOD_PATH="/local/projects/scnn-investigation_haykel/protocols/sub-36/sub-36_mlp_june20_fod_abs.nii.gz"
tckgen "$FOD_PATH" \
     "${OUT_DIR}/mlp_tracks_june20_5tt.tck" \
     -algorithm iFOD2 \
     -seed_dynamic "$FOD_PATH" \
     -mask "$MASK_PATH" \
     -select 100000 \
     -cutoff 0.001 \
     -step 0.4 \
     -angle 20 \
     -minlength 5 \
     -maxlength 300 \
     -act "$FIVE_TT_MIF" \
     -backtrack \
     -crop_at_gmwmi \
     -force
