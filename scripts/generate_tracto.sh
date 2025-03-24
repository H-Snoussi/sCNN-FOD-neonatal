#/bin/bash

for ID in 36; do
for FILE in "wmfod" \
              "sub-${ID}_mlp_fod" \
              "sub-${ID}_fod_sCNN_model_march11_30percent_data"; do

    FOD_PATH="/local/projects/scnn-investigation_haykel/protocols/sub-${ID}/${FILE}.nii.gz"
    OUT_DIR="/local/projects/scnn-investigation_haykel/protocols/sub-${ID}/"
    MASK_PATH="/local/projects/scnn-investigation_haykel/protocols/sub-${ID}/brain_mask.nii.gz"

    tckgen "$FOD_PATH" \
           "${OUT_DIR}/${FILE}_tracks01.tck" \
           -algorithm iFOD2 \
           -seed_dynamic "$FOD_PATH" \
           -mask "$MASK_PATH" \
           -select 100000 \
           -cutoff 0.01 \
           -step 0.5 \
           -angle 20 \
           -minlength 5 \
           -maxlength 100 \
           -force



  done
done