#!/bin/bash

SEED=42
DIMENSIONS_TRAIN="20 50 80"
DIMENSIONS_VALID="5 12 20"

# SST-2
corpus="SST-2"
data_dir="../corpus/${corpus}/preprocessed"
file_train="${data_dir}/train.tsv"
file_test="${data_dir}/test.tsv"
file_val="${data_dir}/dev.tsv"
file_labels="${data_dir}/labels.json"
python3 generate_full_datasets_for_experimenting.py ${corpus} ${file_train} ${file_test} --VALIDATION_FILE ${file_val}
python3 generate_reduced_datasets_for_experimenting.py ${corpus} --N_train ${DIMENSIONS_TRAIN} --N_valid ${DIMENSIONS_VALID} --SEED ${SEED}
cp ${file_labels} ${corpus}

# TREC-6
corpus="TREC"
data_dir="../corpus/${corpus}/preprocessed/"
file_train="${data_dir}/train.tsv"
file_test="${data_dir}/test.tsv"
file_labels="${data_dir}/labels.json"
VALIDATION_SPLIT_SIZE=0.1
python3 generate_full_datasets_for_experimenting.py ${corpus} ${file_train} ${file_test} --SEED ${SEED} --VALIDATION_SPLIT_SIZE ${VALIDATION_SPLIT_SIZE}
python3 generate_reduced_datasets_for_experimenting.py ${corpus} --N_train ${DIMENSIONS_TRAIN} --N_valid ${DIMENSIONS_VALID} --SEED ${SEED}
cp ${file_labels} ${corpus}

# SNIPS
corpus="SNIPS"
data_dir="../corpus/${corpus}/preprocessed"
file_train="${data_dir}/train_full.tsv"
file_test="${data_dir}/validate.tsv"
file_labels="${data_dir}/labels.json"
N_VALIDATION_CLASS=100
python3 generate_full_datasets_for_experimenting.py ${corpus} ${file_train} ${file_test} --SEED ${SEED} --N_VALIDATION_CLASS ${N_VALIDATION_CLASS}
python3 generate_reduced_datasets_for_experimenting.py ${corpus} --N_train ${DIMENSIONS_TRAIN} --N_valid ${DIMENSIONS_VALID} --SEED ${SEED}
cp ${file_labels} ${corpus}

# RAIC and RAIT
LABEL_FOLDERS=("category" "topic")
CORPUS_NAMES=("RAIC" "RAIT")
for ID_LABELS in ${!LABEL_FOLDERS[@]}
do
    corpus="${CORPUS_NAMES[ID_LABELS]}"
    data_dir="../corpus/REPLYDOTAI/preprocessed/${LABEL_FOLDERS[ID_LABELS]}"
    file_train="${data_dir}/train.tsv"
    file_test="${data_dir}/validate.tsv"
    file_statistics="${data_dir}/statistics.json"
    file_labels="${corpus}/labels.json"
    file_labels_sraw="${data_dir}/labels.json"
    file_labels_draw="${corpus}/labels_raw.json"
    VALIDATION_SPLIT_SIZE=0.1
    THRESHOLD=70
    mkdir -p ${corpus}
    python3 remove_limited_classes.py ${file_statistics} ${file_labels} ${THRESHOLD} --VALIDATION_SPLIT_SIZE ${VALIDATION_SPLIT_SIZE}
    cp ${file_labels_sraw} ${file_labels_draw}
    python3 generate_full_datasets_for_experimenting.py ${corpus} ${file_train} ${file_test} --SEED ${SEED} --VALIDATION_SPLIT_SIZE ${VALIDATION_SPLIT_SIZE} \
    --FILE_ID_CLASSES_RAW ${file_labels_draw} --FILE_ID_CLASSES_FILTERED ${file_labels}
    python3 generate_reduced_datasets_for_experimenting.py ${corpus} --N_train ${DIMENSIONS_TRAIN} --N_valid ${DIMENSIONS_VALID} --SEED ${SEED}
done

# Question-Topic
corpus="Question-Topic"
data_dir="../corpus/${corpus}/preprocessed"
file_train="${data_dir}/train.tsv"
file_test="${data_dir}/validate.tsv"
file_labels="${data_dir}/labels.json"
VALIDATION_SPLIT_SIZE=0.1
python3 generate_full_datasets_for_experimenting.py ${corpus} ${file_train} ${file_test} --SEED ${SEED} --VALIDATION_SPLIT_SIZE ${VALIDATION_SPLIT_SIZE}
python3 generate_reduced_datasets_for_experimenting.py ${corpus} --N_train ${DIMENSIONS_TRAIN} --N_valid ${DIMENSIONS_VALID} --SEED ${SEED}
cp ${file_labels} ${corpus}