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
dir=exp/u2pp_conformer
checkpoint=
nbpe=5000

# use average_checkpoint will get better result
average_checkpoint=true
# decode_checkpoint=$dir/final.pt
decode_checkpoint=$dir/159.pt
average_num=150
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
# decode_modes="attention attention_rescoring"

bpemode=unigram
dict=data/lang_char_/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char_/${train_set}_${bpemode}${nbpe}


if true; then
  # Test model, please specify the model you want to test by --checkpoint
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
  # TODO, Add model average here
  mkdir -p $dir/test
  if [ ${average_checkpoint} == true ]; then
      decode_checkpoint=$dir/avg_${average_num}.pt
      echo "do model average and final checkpoint is $decode_checkpoint"
      python wenet/bin/average_model.py \
          --dst_model $decode_checkpoint \
          --src_path $dir  \
          --num ${average_num} \
          --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  # Polling GPU id begin with index 0
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  idx=0
  for mode in ${decode_modes}; do
    {
      {
        test_dir=$dir/test_${mode}
        mkdir -p $test_dir
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
        python wenet/bin/recognize.py --gpu 0 \
          --mode $mode \
          --config $dir/train.yaml \
          --data_type "raw" \
          --bpe_model $bpemodel.model \
          --test_data data/test/data.list \
          --checkpoint $decode_checkpoint \
          --beam_size 20 \
          --batch_size 1 \
          --penalty 0.0 \
          --dict $dict \
          --result_file $test_dir/text_bpe \
          --ctc_weight $ctc_weight \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

        cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
        cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp

         tools/spm_decode --model=${bpemodel}.model --input_format=piece \
           < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value
        #sed -e "s/▁/ /g" $test_dir/text_bpe_value_tmp > $test_dir/text_value
        paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value > $test_dir/text
        # a raw version wer without refining processs
        python tools/compute-wer.py --char=1 --v=1 \
          data/test/text $test_dir/text > $test_dir/wer
      } &

      ((idx+=1))
      if [ $idx -eq $num_gpus ]; then
        idx=0
      fi
    }
    done

  wait
fi
