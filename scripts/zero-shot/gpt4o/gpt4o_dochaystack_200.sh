#!/bin/sh

MODEL="gpt4o"
TASK="DocHaystack-200"
OUTPUT="./output/zero-shot"

python model/main.py --model_name $MODEL \
                     --low_res \
                     --dataset $TASK \
                     --anns_path ./data/test_docVQA.json \
                     --image_path ./data/Test \
                     --outpath $OUTPUT

python eval/gpt_eval.py --api_key gpt-4o-mini \
                        --outpath ${OUTPUT}/${TASK}_${MODEL}.json \