#!/bin/bash

SECTION="full"
DATA_DIR="../data"
PRETRAINED_MODEL_TYPE="roberta"
PRETRAINED_MODEL_NAME="roberta-base"
SEED=42
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
EXPORT_DIR="./EXPORT/TFM/RoBERTa"
CACHE_DIR="./CACHE"
WANDB_DIR="./wandb"
EDA_CONFS=(0 1 2)
CAUG_NS=(10 20 30)
BT_LANGS=("bg" "cs" "ca" "et" "fr" "ru" "fr_ca_ru_cs_et_bg")

function train_roberta {
    # $1 => DATASET
    # $2 => CURRENT_DATA_DIR
    # $3 => CURRENT_EXPORT_DIR
    # $4 => CURRENT_CACHE_DIR
    # $5 => FILE_TRAIN
    # $6 => PROJECT_NAME_WB
    # $7 => LABELS_FILE
    # $8 => PATH_FILE_TRAIN
    # train_roberta ${DATASET} ${CURRENT_DATA_DIR} ${CURRENT_EXPORT_DIR} ${CURRENT_CACHE_DIR} ${FILE_TRAIN} ${PROJECT_NAME_WB} ${LABELS_FILE} ${PATH_FILE_TRAIN}

    text_col=$(head -n 1 $8 | awk -F "\t" '{print $1}')
    label_col=$(head -n 1 $8 | awk -F "\t" '{print $2}')
    
    if [ "$1" == "RAIC" ] || [ "$1" == "RAIT" ]
    then
        python3 ../models/script_roberta.py \
            --pretrained_model_type ${PRETRAINED_MODEL_TYPE} \
            --pretrained_model_name ${PRETRAINED_MODEL_NAME} \
            --data_dir $2 \
            --export_dir $3 \
            --cache_dir $4 \
            --training_file $5 \
            --text_col ${text_col} \
            --label_col ${label_col} \
            --use_custom_split \
            --random_seed ${SEED} \
            --track_with_wb \
            --project_name_wb $6 \
            --anonymized_tokens \
            --labels_file $7
    else
        python3 ../models/script_roberta.py \
            --pretrained_model_type ${PRETRAINED_MODEL_TYPE} \
            --pretrained_model_name ${PRETRAINED_MODEL_NAME} \
            --data_dir $2 \
            --export_dir $3 \
            --cache_dir $4 \
            --training_file $5 \
            --text_col ${text_col} \
            --label_col ${label_col} \
            --use_custom_split \
            --random_seed ${SEED} \
            --track_with_wb \
            --project_name_wb $6 \
            --labels_file $7
    fi
    cp "$3/$6.log" ${EXPORT_DIR}
    rm -rf ${CACHE_DIR} $3 ${WANDB_DIR}
}


for DATASET in ${DATASETS[@]}
do
    
    LABELS_FILE="${DATA_DIR}/${DATASET}/labels.json"
    
    # Easy Data Augmentation
    for EDA_CONF in ${EDA_CONFS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/EDA"
        CURRENT_CACHE_DIR="${CACHE_DIR}/${DATASET}/${SECTION}/EDA"
        PROJECT_NAME_WB="${DATASET}-${SECTION}-EDA-CONF${EDA_CONF}"
        CURRENT_EXPORT_DIR="${EXPORT_DIR}/${PROJECT_NAME_WB}"
        FILE_TRAIN="train_conf${EDA_CONF}.tsv"
        PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
        train_roberta ${DATASET} ${CURRENT_DATA_DIR} ${CURRENT_EXPORT_DIR} ${CURRENT_CACHE_DIR} ${FILE_TRAIN} ${PROJECT_NAME_WB} ${LABELS_FILE} ${PATH_FILE_TRAIN}
    done
    
    # Backtranslation
    for BT_LANG in ${BT_LANGS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/backtranslation"
        CURRENT_CACHE_DIR="${CACHE_DIR}/${DATASET}/${SECTION}/backtranslation"
        PROJECT_NAME_WB="${DATASET}-${SECTION}-backtranslation-lang-${BT_LANG}"
        CURRENT_EXPORT_DIR="${EXPORT_DIR}/${PROJECT_NAME_WB}"
        FILE_TRAIN="train_en_${BT_LANG}.tsv"
        PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
        train_roberta ${DATASET} ${CURRENT_DATA_DIR} ${CURRENT_EXPORT_DIR} ${CURRENT_CACHE_DIR} ${FILE_TRAIN} ${PROJECT_NAME_WB} ${LABELS_FILE} ${PATH_FILE_TRAIN}
    done
    
    # Contextual Augmentation
    for CAUG_N in ${CAUG_NS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/contextual_augmentation"
        CURRENT_CACHE_DIR="${CACHE_DIR}/${DATASET}/${SECTION}/contextual_augmentation"
        PROJECT_NAME_WB="${DATASET}-${SECTION}-contextual-augmentation-N-${CAUG_N}"
        CURRENT_EXPORT_DIR="${EXPORT_DIR}/${PROJECT_NAME_WB}"
        FILE_TRAIN="train_${CAUG_N}.tsv"
        PATH_FILE_TRAIN="${CURRENT_DATA_DIR}/${FILE_TRAIN}"
        cp "${DATA_DIR}/${DATASET}/${SECTION}/test.tsv" ${CURRENT_DATA_DIR}
        train_roberta ${DATASET} ${CURRENT_DATA_DIR} ${CURRENT_EXPORT_DIR} ${CURRENT_CACHE_DIR} ${FILE_TRAIN} ${PROJECT_NAME_WB} ${LABELS_FILE} ${PATH_FILE_TRAIN}
    done
done