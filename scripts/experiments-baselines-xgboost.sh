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
SEED=42
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
EXPORT_DIR="./EXPORT/XGBoost-fastText/${VALIDATION_SIZE}"
NTHREAD=6

for SECTION in ${SECTIONS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/"
        LOG_NAME="${DATASET}-${SECTION}-BASELINE"
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
        python3 ../models/script_xgboost.py \
        --data_dir ${CURRENT_DATA_DIR} \
        --export_dir ${EXPORT_DIR} \
        --training_file ${FILE_TRAIN} \
        --text_col ${text_col} \
        --label_col ${label_col} \
        --random_seed ${SEED} \
        --nthread ${NTHREAD} \
        --log_name ${LOG_NAME}
    done
done