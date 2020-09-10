#!/bin/bash

data_dir="./Question-Topic"
raw_dir="${data_dir}/raw"
input_file="${raw_dir}/question_topic.csv"

if [ ! -d ${raw_dir} ]
then

    mkdir -p ${raw_dir}

    wget "https://raw.githubusercontent.com/sambit9238/Machine-Learning/master/question_topic.csv" -O ${input_file}

fi

# Preprocess

preprocessed_dir="${data_dir}/preprocessed"
rm -rf ${preprocessed_dir}
mkdir -p ${preprocessed_dir}

python3 preprocess_question_topic.py ${input_file} ${preprocessed_dir}
