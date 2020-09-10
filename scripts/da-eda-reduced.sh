#!/bin/bash

DATA_DIR="../data"
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
SECTIONS=("20" "50" "80")
NUM_AUGS=(16 8 4)
ALPHAS=(0.05 0.05 0.1)
FULL_VALIDATION_SIZE="full_valid"
RED_VALIDATION_SIZE="red_valid"
TMP_EDA="/tmp/eda.tsv"
TMP_UNIQ="/tmp/eda_uniq.tsv"

for SECTION in ${SECTIONS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        CURRENT_DATA_DIR_ORGINAL="${DATA_DIR}/${DATASET}/${SECTION}"
        INPUT_FULL="${CURRENT_DATA_DIR_ORGINAL}/train_${FULL_VALIDATION_SIZE}.tsv"
        INPUT_RED="${CURRENT_DATA_DIR_ORGINAL}/train_${RED_VALIDATION_SIZE}.tsv"
        CURRENT_DATA_DIR_EDA="${CURRENT_DATA_DIR_ORGINAL}/EDA"
        rm -rf ${CURRENT_DATA_DIR_EDA}
        mkdir -p ${CURRENT_DATA_DIR_EDA}
        for ID_CONF in ${!ALPHAS[@]}
        do
            NUM_AUG="${NUM_AUGS[ID_CONF]}"
            ALPHA="${ALPHAS[ID_CONF]}"
            OUTPUT_FULL="${CURRENT_DATA_DIR_EDA}/train_${FULL_VALIDATION_SIZE}_conf${ID_CONF}.tsv"
            OUTPUT_RED="${CURRENT_DATA_DIR_EDA}/train_${RED_VALIDATION_SIZE}_conf${ID_CONF}.tsv"
            python3 ../da_methods/eda_nlp/code/augment.py --input ${INPUT_FULL} --output ${TMP_EDA} --num_aug ${NUM_AUG} --alpha ${ALPHA}
            # Add header
            head -n 1 ${TMP_EDA} > ${OUTPUT_FULL}
            cat ${OUTPUT_FULL} > ${OUTPUT_RED}
            # Remove duplicates
            sed '1d' ${TMP_EDA} | awk -F '\t' '$3 == "False" { print $0 }' | sort | uniq > ${TMP_UNIQ}
            cat ${TMP_UNIQ} >> ${OUTPUT_FULL}
            cat ${TMP_UNIQ} >> ${OUTPUT_RED}
            # Use two different validation set
            awk -F '\t' '$3 == "True" { print $0 }' ${INPUT_FULL} >> ${OUTPUT_FULL}
            awk -F '\t' '$3 == "True" { print $0 }' ${INPUT_RED} >> ${OUTPUT_RED}
        done
    done
done