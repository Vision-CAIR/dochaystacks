#!/bin/sh

sh scripts/zero-shot-vrag/gemini/gemini.sh DocHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gemini/gemini.sh DocHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gemini/gemini.sh DocHaystack-1000 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gemini/gemini.sh InfoHaystack-100 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gemini/gemini.sh InfoHaystack-200 5 ./output/retrieval/ test_infoVQA
sh scripts/zero-shot-vrag/gemini/gemini.sh InfoHaystack-1000 5 ./output/retrieval/ test_infoVQA
