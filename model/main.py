import argparse
import os
import json
from gpt4o import EvalGPT4O
from llava_onevision import EvalLLaVAOV
from qwen2_vl import EvalQwen2VL
from gemini import EvalGemini
from tqdm import tqdm

def load_dataset(anns_path, v_RAG = False, retrival_path = ''):
    
    sources = json.load(open(anns_path, 'r'))
    preprocesses = []
    for source in sources:
        
        retrieval = json.load(open(os.path.join(retrival_path, f"{source['id']}.json"), 'r')) if v_RAG else None

        if v_RAG:
            assert '.'.join(source["pos_image"][0].split('.')[:-1]) == retrieval["real_positive_image"], f"paired image should be the same for the same question, but given {source['pos_image'][0]} and {retrieval['real_positive_image']}"
            preprocesses.append({
                "question": source["conversations"][0]["value"],
                "ground_truth": source["conversations"][1]["value"],
                "imageId": source["pos_image"][0],
                "retrieved_image": retrieval["top_10_images"],
                "id": source["id"]
            })
        else:       
            preprocesses.append({
                "question": source["conversations"][0]["value"],
                "ground_truth": source["conversations"][1]["value"],
                "imageId": source["pos_image"][0],
                "id": source["id"]
            })
            
    return preprocesses
    
def loop_whole_images(dataset, image_path):
    
    ann_path = os.path.join(os.path.dirname(image_path), "image_id_classify.json")
    with open(ann_path, 'r') as fp:
        anns = json.load(fp)

    images = [os.path.join(image_path, ann["image_id"]) for ann in anns if dataset in ann["folders"]]

    return images

def set_seed(seed):
    """Set the seed for reproducibility."""

    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args):
    
    # seed everything
    set_seed(args.seed)
    
    image_path = os.path.join(args.image_path, args.dataset.replace('-', '_'))

    # load dataset
    sources = load_dataset(args.anns_path, args.v_RAG, args.retrival_path)
    targets = []
    
    # load model
    if args.model_name == "gpt4o":
        model = EvalGPT4O(args.model_name, args.low_res, args.prompt)
    elif args.model_name == "llava_onevision":
        model = EvalLLaVAOV(args.model_name, args.pretrained, args.low_res, args.scale_factor, args.no_patch, args.prompt)
    elif args.model_name == "qwen2_vl":
        model = EvalQwen2VL(args.model_name, args.pretrained, args.low_res, args.scale_factor, args.prompt)
    elif args.model_name == "gemini":
        model = EvalGemini(args.model_name, args.low_res, args.upload, args.prompt)
    else:
        raise NotImplementedError
    
    if args.v_RAG:
        
        # response per question
        for source in tqdm(sources):
        
            if args.debug and len(targets) >= 3:
                break

            # process image, every question retrieve different image
            imgs_path = [os.path.join(image_path, img_path) for img_path in source["retrieved_image"][:args.topk]]
            images = model.batch_image(imgs_path)
            assert len(images) <= args.topk, f"images should be {args.topk}, but given {len(images)}"
            response = model.generate(source["question"], images)
            
            targets.append({
                "id": source["id"],
                "imageId": source["imageId"],
                "retrieved_image": [img_path for img_path in source["retrieved_image"][:args.topk]],
                "question": source["question"],
                "ground_truth": source["ground_truth"],
                "response": response
            })
        
        save_path = os.path.join(args.outpath, f"{args.dataset}_{args.model_name}_top_{args.topk}.json") if not args.debug else os.path.join(args.outpath, f"{args.dataset}_{args.model_name}_top_{args.topk}_debug.json")
        
    else:

        # process image first, because the images is the same (the whole dataset) for all questions
        # image_path = loop_whole_images(args.dataset, args.image_path)
        image_path = [os.path.join(image_path, img) for img in os.listdir(image_path)]
        images = model.batch_image(image_path)

        # response per question
        for source in tqdm(sources):
        
            if args.debug and len(targets) >= 3:
                break
            
            response = model.generate(source["question"], images)
            
            targets.append({
                "id": source["id"],
                "imageId": source["imageId"],
                "question": source["question"],
                "ground_truth": source["ground_truth"],
                "response": response
            })
    
        save_path = os.path.join(args.outpath, f"{args.dataset}_{args.model_name}.json") if not args.debug else os.path.join(args.outpath, f"{args.dataset}_{args.model_name}_debug.json")
    
    os.makedirs(args.outpath, exist_ok = True)
    with open(save_path, 'w') as fp:
        json.dump(targets, fp, indent = 2)


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type = str, default = "gpt4o")
    args.add_argument("--low_res", action = "store_true", help = "using low resolution image")
    args.add_argument("--scale_factor", type = int, default = 4, help = "scale factor for low resolution, working when low_res is True")
    args.add_argument("--no_patch", action = "store_true", help = "not using patches in llava onevision (to include more images)")
    args.add_argument("--upload", action = "store_true", help = "upload the images to the server for gemini when the input images is too much")
    args.add_argument("--v_RAG", action = "store_true", help = "vision-centric retrieval augmented VQA")
    args.add_argument("--topk", type = int, default = 5, help = "answer with the top k images")
    args.add_argument("--prompt", type = str, default = None, help = "if you want to format the output, set it as: Answer the question using a single word or phrase.")
    args.add_argument("--debug", action = "store_true")
    args.add_argument("--seed", type = int, default = 42, help = "set seed for reproducibility")
    args.add_argument("--anns_path", type = str, default = "./data/test_docVQA.json")
    args.add_argument("--image_path", type = str, default = "./data/Test")
    args.add_argument("--retrival_path", type = str, default = None, help = "path to the retrival images")
    args.add_argument("--pretrained", type = str, default = None, help = "path to the pretrained model if needed")
    args.add_argument("--outpath", type = str, default = "./output")
    args.add_argument(
        "--dataset", 
        choices = ["DocHaystack-100", "DocHaystack-200", "DocHaystack-1000", "InfoHaystack-100", "InfoHaystack-200", "InfoHaystack-1000"],
        required = True,
        help = "choose benchmarks"
    )

    args = args.parse_args()
    main(args)