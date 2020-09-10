#!/bin/bash

data_dir="./SNIPS"
raw_dir="${data_dir}/raw"

if [ ! -d ${raw_dir} ]
then

    if [ ! -d "nlu-benchmark" ]
    then

        git clone https://github.com/snipsco/nlu-benchmark
    fi
    
    labels=("AddToPlaylist" "GetWeather" "RateBook" "SearchCreativeWork" "BookRestaurant" "PlayMusic" "SearchScreeningEvent")

    # Training and Test sets

    mkdir -p ${raw_dir}

    for label in "${labels[@]}"
    do
        train_file_base="train_${label}_full.json"
        validate_file_base="validate_${label}.json"
        train_file="./nlu-benchmark/2017-06-custom-intent-engines/${label}/${train_file_base}"
        validate_file="./nlu-benchmark/2017-06-custom-intent-engines/${label}/${validate_file_base}"
        cp ${train_file} "${raw_dir}/${train_file_base}"
        cp ${validate_file} "${raw_dir}/${validate_file_base}"
    done

fi


# Preprocess

preprocessed_dir="${data_dir}/preprocessed"
rm -rf ${preprocessed_dir}
mkdir -p ${preprocessed_dir}

python3 preprocess_snips.py ${raw_dir} ${preprocessed_dir}

