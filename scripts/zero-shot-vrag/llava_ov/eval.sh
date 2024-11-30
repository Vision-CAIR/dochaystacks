#!/bin/sh

# top 5
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh DocHaystack-100 5 ./output/retrieval/docvqa_100 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh DocHaystack-200 5 ./output/retrieval/docvqa_200 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh DocHaystack-1000 5 ./output/retrieval/docvqa_1000 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh InfoHaystack-100 5 ./output/retrieval/infovqa_100 test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh InfoHaystack-200 5 ./output/retrieval/infovqa_200 test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov_top5.sh InfoHaystack-1000 5 ./output/retrieval/infovqa_1000 test_infoVQA

# run llava_ov.sh for top 1 and top 3 as llava one-vision can accept 3 images without reasonbale prediction.

# # top 3
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh DocHaystack-100 3 ./output/retrieval/docvqa_100 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh DocHaystack-200 3 ./output/retrieval/docvqa_200 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh DocHaystack-1000 3 ./output/retrieval/docvqa_1000 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh InfoHaystack-100 3 ./output/retrieval/infovqa_100 test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh InfoHaystack-200 3 ./output/retrieval/infovqa_200 test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh InfoHaystack-1000 3 ./output/retrieval/infovqa_1000 test_infoVQA

# top 1
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh DocHaystack-100 1 ./output/retrieval/docvqa_100 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh DocHaystack-200 1 ./output/retrieval/docvqa_200 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh DocHaystack-1000 1 ./output/retrieval/docvqa_1000 test_docVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh InfoHaystack-100 1 ./output/retrieval/infovqa_100 test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh InfoHaystack-200 1 ./output/retrieval/infovqa_200 test_infoVQA
sh scripts/zero-shot-vrag/llava_ov/llava_ov.sh InfoHaystack-1000 1 ./output/retrieval/infovqa_1000 test_infoVQA