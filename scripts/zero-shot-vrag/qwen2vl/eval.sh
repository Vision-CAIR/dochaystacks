#!/bin/sh

sh scripts/zero-shot-vrag/qwen2vl/qwen2vl.sh DocHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/qwen2vl/qwen2vl.sh DocHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/qwen2vl/qwen2vl.sh DocHaystack-1000 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/qwen2vl/qwen2vl.sh InfoHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/qwen2vl/qwen2vl.sh InfoHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/qwen2vl/qwen2vl.sh InfoHaystack-1000 5 ./output/retrieval/ test_infoVQA
