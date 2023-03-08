#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from uuid import uuid4

if __name__ == '__main__':
    subset = sys.argv[1]
    filename = sys.argv[2]
    output_dir = sys.argv[3]

    scp_file = open(output_dir + "/wav.scp", "w")
    text_file = open(output_dir + "/text", "w")
    utt2spk = open(output_dir + "/utt2spk", "w")

    with open(filename) as f:
        for idx, row in enumerate(f):
            if idx == 0:
                continue
            parts = row.strip().split(',')
            if len(parts) != 2:
                continue
    
            wav_file = parts[0]
            now_sentence = parts[1]
            client_id = str(uuid4()).replace('-','')

            temple_str = wav_file.split('/')[-1].split('.')[0]
        
            scp_file.writelines(temple_str + " " + wav_file + "\n")
            text_file.writelines(temple_str + " " + now_sentence + "\n")
            utt2spk.writelines(temple_str + " " + client_id + "\n")

    scp_file.close()
    text_file.close()
    utt2spk.close()
