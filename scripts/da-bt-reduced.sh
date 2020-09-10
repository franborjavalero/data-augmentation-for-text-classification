#!/bin/bash


DATA_DIR="../data"
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
SECTIONS=("20" "50" "80")
BATCH_SIZE=32

VALIDATION_COL="is_valid"
FULL_VALIDATION_SIZE="full_valid"

for SECTION in ${SECTIONS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/"
        INPUT_FNAME_FULL="${CURRENT_DATA_DIR}/train_${FULL_VALIDATION_SIZE}.tsv"
        PATH_DIR_TRAIN_BT="${CURRENT_DATA_DIR}/backtranslation"
        rm -rf ${PATH_DIR_TRAIN_BT}
        mkdir -p ${PATH_DIR_TRAIN_BT}
        text_col=$(head -n 1 ${INPUT_FNAME_FULL} | awk -F "\t" '{print $1}')
        label_col=$(head -n 1 ${INPUT_FNAME_FULL} | awk -F "\t" '{print $2}')
        python3 -W ignore ../da_methods/backtranslation.py \
        --input_file ${INPUT_FNAME_FULL} \
        --output_dir ${PATH_DIR_TRAIN_BT} \
        --text_col ${text_col} \
        --label_col ${label_col} \
        --batch_size ${BATCH_SIZE} \
        --validation_col ${VALIDATION_COL}
    done
done