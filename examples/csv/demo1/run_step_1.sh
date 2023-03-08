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
# data
csv_data=/root/autodl-tmp/cv-corpus-8.0-2022-01-19
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
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_conformer_no_pos.yaml: Conformer without relative positional encoding
# 6. conf/train_u2++_conformer.yaml: U2++ conformer
# 7. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=conf/train_u2pp_conformer.yaml
cmvn=true
dir=exp/u2pp_conformer_uk
checkpoint=
nbpe=5000

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=20
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
# decode_modes="attention attention_rescoring"

echo "stage 1: compute global cmvn"
# compute cmvn
python tools/compute_cmvn_stats.py --num_workers 1 --train_config $train_config \
        --in_scp data/${train_set}/wav.scp \
        --out_cmvn data/${train_set}/global_cmvn
echo "Finish stage 1"
