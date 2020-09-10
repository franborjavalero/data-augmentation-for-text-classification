#!/bin/bash

SECTION="full"
DATA_DIR="../data"
SEED=42
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
EXPORT_DIR="./EXPORT/TFM/XGBoost-fastText/"
NTHREAD=6
EDA_CONFS=(0 1 2)
CAUG_NS=(10 20 30)
BT_LANGS=("bg" "cs" "ca" "et" "fr" "ru" "fr_ca_ru_cs_et_bg")

function train_xgboost {
    # $1 => CURRENT_DATA_DIR
    # $2 => EXPORT_DIR
    # $3 => FILE_TRAIN
    # $4 => LOG_NAME
    # $5 => PATH_FILE_TRAIN
    # train_xgboost ${CURRENT_DATA_DIR} ${EXPORT_DIR} ${FILE_TRAIN} ${LOG_NAME} ${PATH_FILE_TRAIN}

    text_col=$(head -n 1 $5 | awk -F "\t" '{print $1}')
    label_col=$(head -n 1 $5 | awk -F "\t" '{print $2}')
    echo $4
    python3 ../models/script_xgboost.py \
        --data_dir $1 \
        --export_dir $2 \
        --training_file $3 \
        --text_col ${text_col} \
        --label_col ${label_col} \
        --random_seed ${SEED} \
        --nthread ${NTHREAD} \
        --log_name $4
}


for DATASET in ${DATASETS[@]}
do
    
    # Easy Data Augmentation
    for EDA_CONF in ${EDA_CONFS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/EDA"
        LOG_NAME="${DATASET}-${SECTION}-EDA-CONF${EDA_CONF}"
        FILE_TRAIN="train_conf${EDA_CONF}.tsv"
        PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
        train_xgboost ${CURRENT_DATA_DIR} ${EXPORT_DIR} ${FILE_TRAIN} ${LOG_NAME} ${PATH_FILE_TRAIN}
    done
    
    # Backtranslation
    for BT_LANG in ${BT_LANGS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/backtranslation"
        LOG_NAME="${DATASET}-${SECTION}-backtranslation-lang-${BT_LANG}"
        FILE_TRAIN="train_en_${BT_LANG}.tsv"
        PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
        train_xgboost ${CURRENT_DATA_DIR} ${EXPORT_DIR} ${FILE_TRAIN} ${LOG_NAME} ${PATH_FILE_TRAIN}
    done
    
    # Contextual Augmentation
    for CAUG_N in ${CAUG_NS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/contextual_augmentation"
        LOG_NAME="${DATASET}-${SECTION}-contextual-augmentation-N-${CAUG_N}"
        FILE_TRAIN="train_${CAUG_N}.tsv"
        PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
        train_xgboost ${CURRENT_DATA_DIR} ${EXPORT_DIR} ${FILE_TRAIN} ${LOG_NAME} ${PATH_FILE_TRAIN}
    done
done