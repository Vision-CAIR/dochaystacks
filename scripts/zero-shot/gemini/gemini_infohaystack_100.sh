#!/bin/sh

MODEL="gemini"
TASK="InfoHaystack-100"
OUTPUT="./output/zero-shot"

python model/main.py --model_name $MODEL \
                     --upload \
                     --dataset $TASK \
                     --anns_path ./data/test_infoVQA.json \
                     --image_path ./data/Test \
                     --outpath $OUTPUT

python eval/gpt_eval.py --api_key gpt-4o-mini \
                        --outpath ${OUTPUT}/${TASK}_${MODEL}.json \