#!/bin/bash

data_dir="./SST-2"
raw_dir="${data_dir}/raw"

if [ ! -d ${raw_dir} ]
then
    base_name="sentiment_dataset"
    git clone "https://github.com/AcademiaSinicaNLPLab/${base_name}.git"
    mkdir -p ${raw_dir}
    sections=("train" "test" "dev")
    for section in "${sections[@]}"
    do
        cp ${base_name}/data/stsa.binary.${section} ${raw_dir}/${section}.txt
    done
    rm -rf ${base_name}
fi

# Preprocess

preprocessed_dir="${data_dir}/preprocessed"
rm -rf ${preprocessed_dir}
mkdir -p ${preprocessed_dir}

python3 preprocess_sst_2.py ${raw_dir} ${preprocessed_dir}