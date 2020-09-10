#!/bin/bash

data_dir="./REPLYDOTAI"
raw_dir="${data_dir}/raw"
input_file="${raw_dir}/user-queries-topics-en_v3.4.csv"

if [ ! -d ${raw_dir} ]
then

    mkdir -p ${raw_dir}

    wget "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/862d2148-68f1-4f88-8a97-60b8122e067c/user-queries-topics-en_v3.4.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200619%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200619T163222Z&X-Amz-Expires=86400&X-Amz-Signature=8e477930beaf95b943d7329a2e72a5fe39cb096bc354a374b351d60090e02262&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22user-queries-topics-en_v3.4.csv%22" -O ${input_file}

fi

# Preprocess

preprocessed_dir="${data_dir}/preprocessed"
rm -rf ${preprocessed_dir}
mkdir -p ${preprocessed_dir}

python3 preprocess_replydotai.py ${input_file} ${preprocessed_dir}
