#!/bin/sh

MODEL="llava_onevision"
TASK="DocHaystack-100"
OUTPUT="./output/zero-shot"

python model/main.py --model_name $MODEL \
                     --no_patch \
                     --debug \
                     --dataset $TASK \
                     --anns_path ./data/test_docVQA.json \
                     --image_path ./data/Test \
                     --pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
                     --outpath $OUTPUT