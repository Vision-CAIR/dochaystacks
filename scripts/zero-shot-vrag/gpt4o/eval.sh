#!/bin/sh

sh scripts/zero-shot-vrag/gpt4o/gpt4o.sh DocHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gpt4o/gpt4o.sh DocHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gpt4o/gpt4o.sh DocHaystack-1000 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gpt4o/gpt4o.sh InfoHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gpt4o/gpt4o.sh InfoHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gpt4o/gpt4o.sh InfoHaystack-1000 5 ./output/retrieval/ test_infoVQA
