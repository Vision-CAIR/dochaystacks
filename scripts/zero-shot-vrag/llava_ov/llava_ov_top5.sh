#!/bin/sh

MODEL="llava_onevision"
OUTPUT="./output/zero-shot-vrag"

TASK=$1
TOPK=$2
RETRIVAL=$3
ANN=$4

python model/main.py --model_name $MODEL \
                     --no_patch \
                     --v_RAG \
                     --topk $TOPK \
                     --dataset $TASK \
                     --anns_path ./data/${ANN}.json \
                     --image_path ./data/Test \
                     --retrival_path $RETRIVAL \
                     --pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
                     --outpath $OUTPUT

python eval/gpt_eval.py --api_key gpt-4o-mini \
                        --outpath ${OUTPUT}/${TASK}_${MODEL}_top_${TOPK}.json \