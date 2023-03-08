#!/usr/bin/env bash

data=`pwd`/data
local=`pwd`/local

train_csv_file=/home/yehor/ukr/train.csv
test_csv_file=/home/yehor/ukr/test.csv

python3 ${local}/create_scp_text.py train ${train_csv_file} ${data}/train

python3 ${local}/create_scp_text.py test ${test_csv_file} ${data}/test
