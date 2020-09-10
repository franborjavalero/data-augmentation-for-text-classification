#!/bin/bash

if [ $# == 1 ]
then
    if [ $1 == "full" ]
    then
        VALIDATION_SIZE=$1
        SECTIONS=("full" "20" "50" "80")
    elif [ $1 == "red" ]
    then
        VALIDATION_SIZE=$1
        SECTIONS=("20" "50" "80")
    else
        echo "Incorrect VALIDATION-SIZE, tha valid values are 'full' and 'red'"
        exit
    fi
else
    echo "./launch-baselines-roberta.sh VALIDATION-SIZE"
    exit
fi

DATA_DIR="../data"
PRETRAINED_MODEL_TYPE="roberta"
PRETRAINED_MODEL_NAME="roberta-base"
SEED=42
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
EXPORT_DIR="./EXPORT/RoBERTa/${VALIDATION_SIZE}"
CACHE_DIR="./CACHE"
WANDB_DIR="./wandb"

for SECTION in ${SECTIONS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}"
        CURRENT_CACHE_DIR="${CACHE_DIR}/${DATASET}/${SECTION}"
        PROJECT_NAME_WB="${DATASET}-${SECTION}-BASELINE-VALIDATION-${VALIDATION_SIZE}"
        CURRENT_EXPORT_DIR="${EXPORT_DIR}/${PROJECT_NAME_WB}"
        if [ $SECTION == "full" ]
        then
            FILE_TRAIN="train.tsv"
            PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        else
            FILE_TRAIN="train_${VALIDATION_SIZE}_valid.tsv"
            PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        fi
        text_col=$(head -n 1 ${PATH_FILE_TRAIN} | awk -F "\t" '{print $1}')
        label_col=$(head -n 1 ${PATH_FILE_TRAIN} | awk -F "\t" '{print $2}')
        LABELS_FILE="${DATA_DIR}/${DATASET}/labels.json"
        if [ "${DATASET}" == "RAIC" ] || [ "${DATASET}" == "RAIT" ]
        then
            python3 ../models/script_roberta.py \
                --pretrained_model_type ${PRETRAINED_MODEL_TYPE} \
                --pretrained_model_name ${PRETRAINED_MODEL_NAME} \
                --data_dir ${CURRENT_DATA_DIR} \
                --export_dir ${CURRENT_EXPORT_DIR} \
                --cache_dir ${CURRENT_CACHE_DIR} \
                --training_file ${FILE_TRAIN} \
                --text_col ${text_col} \
                --label_col ${label_col} \
                --use_custom_split \
                --random_seed ${SEED} \
                --track_with_wb \
                --project_name_wb ${PROJECT_NAME_WB} \
                --anonymized_tokens \
                --labels_file ${LABELS_FILE}
        else
            python3 ../models/script_roberta.py \
                --pretrained_model_type ${PRETRAINED_MODEL_TYPE} \
                --pretrained_model_name ${PRETRAINED_MODEL_NAME} \
                --data_dir ${CURRENT_DATA_DIR} \
                --export_dir ${CURRENT_EXPORT_DIR} \
                --training_file ${FILE_TRAIN} \
                --cache_dir ${CURRENT_CACHE_DIR} \
                --text_col ${text_col} \
                --label_col ${label_col} \
                --use_custom_split \
                --random_seed ${SEED} \
                --track_with_wb \
                --project_name_wb ${PROJECT_NAME_WB} \
                --labels_file ${LABELS_FILE}
        fi
        cp "${CURRENT_EXPORT_DIR}/${PROJECT_NAME_WB}.log" ${EXPORT_DIR}
        rm -rf ${CACHE_DIR} ${CURRENT_EXPORT_DIR} ${WANDB_DIR}
    done
done