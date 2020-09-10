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

DATA_DIR="../data"
DATASETS=("SST-2" "SNIPS" "TREC" "RAIC" "RAIT" "Question-Topic")

NUM_AUGS=(16 8 4)
ALPHAS=(0.05 0.05 0.1)

NS=("10" "20" "30")

TMP_EDA="/tmp/eda.tsv"
TMP_UNIQ="/tmp/eda_uniq.tsv"

INPUT_TRAIN_TMP="/tmp/ca-train"
OUPUT_TMP="/tmp/ca-aug"
UNIQ_TMP="/tmp/ca-aug-uniq.tsv"
INPUT_FULL_VAL_TMP="/tmp/ca-fval.tsv"
INPUT_RED_VAL_TMP="/tmp/ca-rval.tsv"

BATCH_SIZE=16
VALIDATION_COL="is_valid"

for DATASET in ${DATASETS[@]}
do
    
    # EDA
    CURRENT_DATA_DIR="${DATA_DIR}/${DATASET}/full"
    INPUT="${CURRENT_DATA_DIR}/train.tsv"
    CURRENT_DATA_DIR_EDA="${CURRENT_DATA_DIR}/EDA"
    rm -rf ${CURRENT_DATA_DIR_EDA}
    mkdir -p ${CURRENT_DATA_DIR_EDA}
    for ID_CONF in ${!ALPHAS[@]}
    do
        NUM_AUG="${NUM_AUGS[ID_CONF]}"
        ALPHA="${ALPHAS[ID_CONF]}"
        OUTPUT="${CURRENT_DATA_DIR_EDA}/train_conf${ID_CONF}.tsv"
        python3 ../da_methods/eda_nlp/code/augment.py --input ${INPUT} --output ${TMP_EDA} --num_aug ${NUM_AUG} --alpha ${ALPHA}
        # Add header
        head -n 1 ${TMP_EDA} > ${OUTPUT}
        # Remove duplicates
        sed '1d' ${TMP_EDA} | awk -F '\t' '$3 == "False" { print $0 }' | sort | uniq > ${TMP_UNIQ}
        cat ${TMP_UNIQ} >> ${OUTPUT}
        # Add validation set
        awk -F '\t' '$3 == "True" { print $0 }' ${INPUT} >> ${OUTPUT}
    done
    
    # BACK-TRANSLATION
    PATH_DIR_TRAIN_BT="${CURRENT_DATA_DIR}/backtranslation"
    rm -rf ${PATH_DIR_TRAIN_BT}
    mkdir -p ${PATH_DIR_TRAIN_BT}
    text_col=$(head -n 1 ${INPUT} | awk -F "\t" '{print $1}')
    label_col=$(head -n 1 ${INPUT} | awk -F "\t" '{print $2}')
    python3 -W ignore ../da_methods/backtranslation.py \
    --input_file ${INPUT} \
    --output_dir ${PATH_DIR_TRAIN_BT} \
    --text_col ${text_col} \
    --label_col ${label_col} \
    --batch_size ${BATCH_SIZE} \
    --validation_col ${VALIDATION_COL}
    
    # CONTEXTUAL-AUGMETATION
    if [ $DATASET == "Question-Topic" ]
    then
        PATH_DIR_TRAIN_CAUG="${CURRENT_DATA_DIR}/contextual_augmentation"
        head -n 1 ${INPUT} > "${INPUT_TRAIN_TMP}.tsv"
        awk -F '\t' '$3 == "False" { print $0 }' ${INPUT} >> "${INPUT_TRAIN_TMP}.tsv"
        awk -F '\t' '$3 == "True" { print $0 }' ${INPUT} > ${INPUT_FULL_VAL_TMP}
        rm -rf ${PATH_DIR_TRAIN_CAUG}
        mkdir -p ${PATH_DIR_TRAIN_CAUG}
        for N in ${NS[@]}
        do  
            OUTPUT="${PATH_DIR_TRAIN_CAUG}/train_${N}.tsv"
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
            cat ${UNIQ_TMP} ${INPUT_FULL_VAL_TMP} >> ${OUTPUT}
        done
    fi

done