#!/bin/sh

MODEL="qwen2_vl"
TASK="DocHaystack-200"
OUTPUT="./output/zero-shot"

python model/main.py --model_name $MODEL \
                     --low_res \
                     --scale_factor 6 \
                     --dataset $TASK \
                     --anns_path ./data/test_docVQA.json \
                     --image_path ./data/Test \
                     --pretrained Qwen/Qwen2-VL-7B-Instruct \
                     --outpath $OUTPUT

python eval/gpt_eval.py --api_key gpt-4o-mini \
                        --outpath ${OUTPUT}/${TASK}_${MODEL}.json \