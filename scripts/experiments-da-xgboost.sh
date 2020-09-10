#!/bin/bash

if [ $# == 1 ]
then
    if [ $1 == "full" ] || [ $1 == "red" ]
    then
        VALIDATION_SIZE=$1
    else
        echo "Incorrect VALIDATION-SIZE, tha valid values are 'full' and 'red'"
        exit
    fi
else
    echo "./launch-da-roberta.sh VALIDATION-SIZE"
    exit
fi

DATA_DIR="../data"
SEED=42
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
EXPORT_DIR="./EXPORT/XGBoost-fastText/${VALIDATION_SIZE}"
NTHREAD=6
SECTIONS=("20" "50" "80")
EDA_CONFS=(0 1 2)
CAUG_NS=(10 20 30)
BT_LANGS=("bg" "cs_et_nl_id_bg" "fr" "it" "ca_fr_de_it" "cs" "id" "nl" "ca_fr" "de" "it_fr_de_ca_ru_cs_et_nl_id_bg" "ru" "ca" "et" "it_fr_de_ca_ru")

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

for SECTION in ${SECTIONS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        # Easy Data Augmentation
        for EDA_CONF in ${EDA_CONFS[@]}
        do
            CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/EDA"
            LOG_NAME="${DATASET}-${SECTION}-EDA-CONF${EDA_CONF}"
            FILE_TRAIN="train_${VALIDATION_SIZE}_valid_conf${EDA_CONF}.tsv"
            PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
            cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
            train_xgboost ${CURRENT_DATA_DIR} ${EXPORT_DIR} ${FILE_TRAIN} ${LOG_NAME} ${PATH_FILE_TRAIN}
        done
        # Backtranslation
        for BT_LANG in ${BT_LANGS[@]}
        do
            CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/backtranslation"
            LOG_NAME="${DATASET}-${SECTION}-backtranslation-lang-${BT_LANG}"
            # FILE_TRAIN="train_${VALIDATION_SIZE}_valid_en_${BT_LANG}.tsv"
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
            FILE_TRAIN="train_${VALIDATION_SIZE}_valid_${CAUG_N}.tsv"
            PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
            cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
            train_xgboost ${CURRENT_DATA_DIR} ${EXPORT_DIR} ${FILE_TRAIN} ${LOG_NAME} ${PATH_FILE_TRAIN}
        done
    done
done