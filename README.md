## Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents

**Authors**: [Jun Chen](https://junchen14.github.io/), Dannong Xu, [Junjie Fei](https://feielysia.github.io/), [Chun-Mei Feng](https://scholar.google.com.hk/citations?user=g2nqHBcAAAAJ&hl=zh-CN), [Mohamed Elhoseiny](https://scholar.google.com/citations?user=iRBUTOAAAAAJ&hl=en)

The official implementation of our paper: [*Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents*](https://arxiv.org/pdf/2411.16740).

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2307.16525-b31b1b.svg)](https://arxiv.org/abs/2411.16740)
[![benchmark](https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-DocHaystack-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/)

</div>

## Catalogue:
* <a href='#introduction'>Introdction</a>
* <a href='#citation'>Citation</a>
* <a href='#data_preparation'>Data Preparation</a>
* <a href='#evaluation'>Evaluation</a>
    * <a href='#vc_retrieval'>Visual-Centric Retrieval</a>
    * <a href='#aug_reasoning'>Augmented Multi-Image Reasoning</a>
* <a href='#finetune'>Fine-Tuning</a>
* <a href='#acknowledgments'>Acknowledgments</a>
* <a href='#contact'>Contact</a>

<span id = 'introduction'/>

***

## Introduction

While large multimodal models (LMMs) have achieved impressive progress in vision-language understanding, they fall short in reasoning over a large number of images, a complex but common real-world application. Existing benchmarks for Multi-Image Question Answering fail to comprehensively evaluate this capability of LMMs. To bridge this gap, we introduce two document haystack benchmarks, DocHaystack and InfoHaystack, designed to evaluate LMMs' performance on large-scale visual document retrieval and understanding. Unlike previous benchmarks, DocHaystack and InfoHaystack map each question to a substantially larger document collection, scaling up to 1,000 visual documents. This expanded scope more accurately represents large-scale document retrieval scenarios and offers a greater challenge in retrieval accuracy and visual question answering. Additionally, we propose V-RAG, a novel, vision-centric retrieval-augmented generation (RAG) framework enabling efficiently question answering across thousands of images, setting a new standard on our DocHaystack and InfoHaystack benchmarks.

<div align = center>
<img src="./assets/benchmark.png" width = 100% heigth = 100%>
</div>



***

<span id = 'data_preparation'/>

## Data Preparation

First, we should download the DocHaystack and InfoHaystack Benchmarks from [Huggingface 🤗](https://huggingface.co/), respectively. Then, place the downloaded benchmarks into the `data/*` directory. The data should be organized in the following format:

```
├── dochaystacks
│   ├── data
│   │   ├── Train
│   │   │   ├── infographicsvqa_images
│   │   │   ├── spdocvqa_images
│   │   ├── Test
│   │   │   ├── DocHaystack_100
│   │   │   ├── DocHaystack_200
│   │   │   ├── DocHaystack_1000
│   │   │   ├── InfoHaystack_100
│   │   │   ├── InfoHaystack_200
│   │   │   ├── InfoHaystack_1000
│   │   ├── test_docVQA.json
│   │   ├── test_infoVQA.json
│   │   ├── train_specific.json
```

***

<span id = 'evaluation'/>

## Evaluation

To evaluate the performance of LMMs on DocHaystack and InfoHaystack, execute the scripts provided in the `scripts/*` directory.

By running the following commands, you can obtain the results of current LMMs on large-scale visual document understanding without any additional processing. For Qwen2-VL, we reduce the input image resolution using the `--low_res` and `--scale_factor` options to ensure all inputs fit on a single A100 GPU (80G). LLaVA-OneVision, however, cannot process large-scale visual documents, even when attempting to handle multiple input images as a video using the `--no_patch` option. For this reason, we only provide a script demonstrating how to run LLaVA-OneVision on our benchmarks.

Note: Due to API calls, there may be variations in the results. However, the overall conclusions drawn should remain consistent.

```bash
sh scripts/zero-shot/qwen2vl/*.sh
sh scripts/zero-shot/llava_ov/*.sh
sh scripts/zero-shot/gpt4o/*.sh
sh scripts/zero-shot/gemini/*.sh
```

***

<span id = 'vc_retrieval'/>

## Visual-Centric Retrieval

sh scripts/retrieval/*.sh

***

<span id = 'aug_reasoning'/>

### Augmented Multi-Image Reasoning 

We enhance the large-scale visual document understanding capabilities of existing LMMs through vision-centric retrieval-augmented generation (V-RAG). To evaluate the performance of V-RAG, you first need to obtain the visual-centric retrieval results and save them in the `/output/retrieval/*` directory. Once the retrieved results are available, augmenting any LMM is straightforward: simply feed the retrieved top k images into the model. By running the following commands, you can easily evaluate the performance of LMMs augmented by visual-centric retrieval on DocHaystack and InfoHaystack.

Note: For LLaVA-OneVision, we observed that the model collapses when handling multiple images directly (without video-like processing) with top_k = 5.

```bash
sh scripts/zero-shot-vrag/qwen2vl/eval.sh
sh scripts/zero-shot-vrag/llava_ov/eval.sh
sh scripts/zero-shot-vrag/gpt4o/eval.sh
sh scripts/zero-shot-vrag/gemini/eval.sh
```



<div align = center>

Model          |DocHaystack-100|DocHaystack-200|DocHaystack-1000|InfoHaystack-100|InfoHaystack-200|InfoHaystack-1000
-----|------|------|------|-----|------|------
LLaVA-OV+V-RAG|69.72|65.14|55.05|43.22|41.94|36.77
Gemini+V-RAG|73.39|65.14|58.72|57.42|57.42|47.10
GPT-4o+V-RAG|81.65|72.48|66.97|65.16|63.23|56.77
Qwen2-VL+V-RAG|82.57|74.31|66.06|65.81|65.81|60.00

</div>

***

<span id = 'finetune'/>

## Fine-Tuning

We fine-tune Qwen2-VL on our curated dataset using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which makes the implementation straightforward by following their instructions. To maintain balance during fine-tuning, we ensure that the number of samples from infographicsvqa (899) matches the number of docvqa samples (899).

***

<span id = 'acknowledgments'/>

## Acknowledgments

Our repository builds on [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), GPT-4o, Gemini. Thanks for them!

***

<span id = 'contact'/>

***

<span id = 'citation'/>

## Citation

If you find our paper and code helpful, we would greatly appreciate it if you could leave a star and cite our work. Thanks!

```bibtex
@article{chen2024document,
  title={Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents},
  author={Chen, Jun and Xu, Dannong and Fei, Junjie and Feng, Chun-Mei and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2411.16740},
  year={2024}
}
```

## Contact

If you have any questions, please feel free to contact us.