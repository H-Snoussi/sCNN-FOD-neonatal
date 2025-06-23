#/bin/bash

MAINDHCPDIR="/mnt/HardDisk8TB/projects/neonatal_dhcp_ieee_access/data"
PROTOCOLS="/local/projects/scnn-investigation_haykel/protocols"
counter=1

for SUBJECT in ${MAINDHCPDIR}/*; do
    for SESSION in ${SUBJECT}/ses-*; do
        SUBJECID=${SUBJECT#${MAINDHCPDIR}/}
        SESSIONID=${SESSION#${SUBJECT}/}

        BVAL="${SESSION}/dwi/dwi.bval"
        BVEC="${SESSION}/dwi/dwi.bvec"
        DMRI="${SESSION}/dwi/dwi.nii.gz"
        MASK="${SESSION}/dwi/brain_mask.nii.gz"

        if [ "$counter" -lt 10 ]; then
            OUTPUTDIR="${PROTOCOLS}/sub-0${counter}"
        else
            OUTPUTDIR="${PROTOCOLS}/sub-${counter}"
        fi

        mkdir -p "${OUTPUTDIR}"

        cp $BVAL ${OUTPUTDIR}/dwi.bval 
        cp $BVEC ${OUTPUTDIR}/dwi.bvec
        cp $MASK ${OUTPUTDIR}/brain_mask.nii.gz 
        cp $DMRI ${OUTPUTDIR}/dwi.nii.gz 

        echo "Processing ${SUBJECID} ${SESSIONID}..."
        echo "${OUTPUTDIR}"
        # Remove negative values
        echo "Removing negative values..."
        fslmaths $DMRI -thr 0 ${OUTPUTDIR}/dwi_no_neg.nii.gz
        
        # threshold to 55
        fslmaths ${OUTPUTDIR}/dwi_no_neg.nii.gz -uthr 55 ${OUTPUTDIR}/dwi_norm.nii.gz



        #  Use normalized dMRI for FOD estimation
        dwi2response dhollander -fslgrad ${BVEC} ${BVAL} ${OUTPUTDIR}/dwi_norm.nii.gz \
            ${OUTPUTDIR}/wm_response.txt ${OUTPUTDIR}/gm_response.txt ${OUTPUTDIR}/csf_response.txt

        FODTYPE="REVISION"
        if [ "$FODTYPE" == "FIRST_SUBMISSION" ]; then
            dwi2response dhollander -fslgrad ${BVEC} ${BVAL} ${OUTPUTDIR}/dwi_norm.nii.gz \
                ${OUTPUTDIR}/wm_response.txt ${OUTPUTDIR}/gm_response.txt ${OUTPUTDIR}/csf_response.txt 

        elif [ "$FODTYPE" == "REVISION" ]; then
            dwi2response dhollander -fslgrad ${BVEC} ${BVAL} ${OUTPUTDIR}/dwi_norm.nii.gz \
                ${OUTPUTDIR}/wm_response.txt ${OUTPUTDIR}/gm_response.txt ${OUTPUTDIR}/csf_response.txt -wm_algo tournier -fa 0.5 -force

        fi


        dwi2fod msmt_csd ${OUTPUTDIR}/dwi_norm.nii.gz -fslgrad ${BVEC} ${BVAL} -mask ${MASK} \
            ${OUTPUTDIR}/wm_response.txt ${OUTPUTDIR}/wmfod.nii.gz \
            ${OUTPUTDIR}/gm_response.txt ${OUTPUTDIR}/gm.nii.gz \
            ${OUTPUTDIR}/csf_response.txt ${OUTPUTDIR}/csf.nii.gz

        echo "Completed processing ${SUBJECID} ${SESSIONID}"
        counter=$((counter + 1))

    done 
done


