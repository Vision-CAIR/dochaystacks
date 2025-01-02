#!/bin/bash

DATASET_FOLDER="./data"
DATASET_FILE="$DATASET_FOLDER/test_infoVQA.json"
IMAGE_ROOT="$DATASET_FOLDER/Test"
IMAGE_DIR="InfoHaystack_200"

OUTPUT_DIR="./output/infovqa_200"


python model/VRAG_retrieval.py --dataset_file $DATASET_FILE --image_root $IMAGE_ROOT --image_dir $IMAGE_DIR --output_dir $OUTPUT_DIR --use_question_query
