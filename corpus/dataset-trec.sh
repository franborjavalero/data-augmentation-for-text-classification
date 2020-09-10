#!/bin/bash

data_dir="./TREC"
raw_dir="${data_dir}/raw"

if [ ! -d ${raw_dir} ]
then
    
    file_names=("train_5500.label" "TREC_10.label")
    
    rm -rf "${data_dir}"
    mkdir -p "${data_dir}"

    # Training and Test sets

    mkdir -p ${raw_dir}

    for file_name in "${file_names[@]}"
    do
        url="https://cogcomp.seas.upenn.edu/Data/QA/QC/${file_name}"
        wget -P ${raw_dir} $url
    done

fi


# Preprocess

preprocessed_dir="${data_dir}/preprocessed"
rm -rf ${preprocessed_dir}
mkdir -p ${preprocessed_dir}

python3 preprocess_trec.py ${raw_dir} ${preprocessed_dir}