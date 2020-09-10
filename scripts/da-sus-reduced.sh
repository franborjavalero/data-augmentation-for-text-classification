#!/bin/bash

# Download GloVe
GLOVE_DIR="./glove"
GLOVE_CONF="glove.42B.300d"
GLOVE_EMB="./${GLOVE_DIR}/${GLOVE_CONF}.txt"
if [ ! -d ${GLOVE_DIR} ]
then 
    mkdir -p ${GLOVE_DIR}
    compresed_file="${GLOVE_DIR}/${GLOVE_CONF}.zip"
    wget "http://nlp.stanford.edu/data/${GLOVE_CONF}.zip" -O ${compresed_file}
    unzip ${compresed_file} -d ${GLOVE_DIR}
    rm -rf ${compresed_file}
fi

# Download BERT
BERT_DIR="bert-base-uncased"
if [ ! -d ${BERT_DIR} ]
then
    mkdir -p ${BERT_DIR}
    wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt" -O "${BERT_DIR}/vocab.txt"
    compresed_file="${BERT_DIR}/bert-base-uncased.tar.gz"
    wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz" -O ${compresed_file}
    tar zxvf ${compresed_file} -C ${BERT_DIR}
    mv "${BERT_DIR}/bert_config.json" "${BERT_DIR}/config.json"
    rm -rf ${compresed_file}
fi

# Data augmetation
DATA_DIR="../data"
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")
SECTIONS=("20" "50" "80")
NS=("10" "20" "30")
FULL_VALIDATION_SIZE="full_valid"
RED_VALIDATION_SIZE="red_valid"
INPUT_TRAIN_TMP="/tmp/ca-train"
OUPUT_TMP="/tmp/ca-aug"
UNIQ_TMP="/tmp/ca-aug-uniq.tsv"
INPUT_FULL_VAL_TMP="/tmp/ca-fval.tsv"
INPUT_RED_VAL_TMP="/tmp/ca-rval.tsv"

for SECTION in ${SECTIONS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/${SECTION}/"
        INPUT_FNAME_FULL="${CURRENT_DATA_DIR}/train_${FULL_VALIDATION_SIZE}.tsv"
        INPUT_FNAME_RED="${CURRENT_DATA_DIR}/train_${RED_VALIDATION_SIZE}.tsv"
        PATH_AUG="${CURRENT_DATA_DIR}/contextual_augmentation"
        head -n 1 ${INPUT_FNAME_FULL} > "${INPUT_TRAIN_TMP}.tsv"
        awk -F '\t' '$3 == "False" { print $0 }' ${INPUT_FNAME_FULL} >> "${INPUT_TRAIN_TMP}.tsv"
        awk -F '\t' '$3 == "True" { print $0 }' ${INPUT_FNAME_FULL} > ${INPUT_FULL_VAL_TMP}
        awk -F '\t' '$3 == "True" { print $0 }' ${INPUT_FNAME_RED} > ${INPUT_RED_VAL_TMP}
        rm -rf ${PATH_AUG}
        mkdir -p ${PATH_AUG}
        for N in ${NS[@]}
        do  
            OUTPUT_FNAME_FULL="${PATH_AUG}/train_${FULL_VALIDATION_SIZE}_${N}.tsv"
            OUTPUT_FNAME_RED="${PATH_AUG}/train_${RED_VALIDATION_SIZE}_${N}.tsv"
            python3 ../da_methods/Pretrained-Language-Model/TinyBERT/data_augmentation.py \
            --pretrained_bert_model ${BERT_DIR} \
            --glove_embs ${GLOVE_EMB} \
            --task_name ${DATASET} \
            --input_fname ${INPUT_TRAIN_TMP} \
            --output_fname ${OUPUT_TMP}  \
            --N ${N}
            # Add header
            head -n 1 ${OUPUT_TMP}.tsv > ${UNIQ_TMP}
            # Remove duplicates
            sed '1d' ${OUPUT_TMP}.tsv | sort | uniq >> ${UNIQ_TMP}
            # Use two different validation set
            cat ${UNIQ_TMP} ${INPUT_FULL_VAL_TMP} >> ${OUTPUT_FNAME_FULL}
            cat ${UNIQ_TMP} ${INPUT_RED_VAL_TMP} >> ${OUTPUT_FNAME_RED}
        done
    done
done