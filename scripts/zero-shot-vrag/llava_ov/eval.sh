#!/bin/sh

sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh DocHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh DocHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh DocHaystack-1000 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh InfoHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh InfoHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh InfoHaystack-1000 5 ./output/retrieval/ test_infoVQA

# run llava_ov.sh for top 1 and top 3 as llava one-vision can accept 3 images without reasonbale prediction.
