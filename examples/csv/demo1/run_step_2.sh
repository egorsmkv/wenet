#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0

# path to save preproecssed data
# export data=data
. ./path.sh
. ./tools/parse_options.sh || exit 1

nj=16

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=
nbpe=5000

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=20
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
# decode_modes="attention attention_rescoring"


bpemode=unigram
dict=data/lang_char_/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char_/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"

### Task dependent. You have to check non-linguistic symbols used in the corpus.
echo "stage 2: Dictionary and Json Data Preparation"
mkdir -p data/lang_char_/
echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
echo "<unk> 1" >> ${dict} # <unk> must be 1

# we borrowed these code and scripts which are related bpe from ESPnet.
cut -f 2- -d" " data/${train_set}/text > data/lang_char_/input.txt
tools/spm_train --input=data/lang_char_/input.txt --vocab_size=${nbpe} \
    --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
tools/spm_encode --model=${bpemodel}.model --output_format=piece \
    < data/lang_char_/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
num_token=$(cat $dict | wc -l)
echo "<sos/eos> $num_token" >> $dict # <eos>
wc -l ${dict}
