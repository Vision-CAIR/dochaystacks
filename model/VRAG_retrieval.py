import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
import numpy as np
import json
import argparse
import time
import logging
import open_clip

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

import copy

pretrained = "lmms-lab/llava-onevision-qwen2-7b-si"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
model.eval()

def llava_single_image_inference(image_path, question):
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_size = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_size,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]

class CombinedSolver:
    def __init__(self, clip_model_name='openai/clip-vit-large-patch14', siglip_model_name='google/siglip-so400m-patch14-384',
                 openclip_model_name='convnext_xxlarge', image_root='dataset', image_dir="image/infor_100",use_question_query=False):
        
        self.image_root = image_root
        self.image_dir = os.path.join(self.image_root, image_dir)
        
        self.clip_model = AutoModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = AutoProcessor.from_pretrained(clip_model_name)
        
        self.siglip_model = AutoModel.from_pretrained(siglip_model_name).to(device)
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
        
        self.openclip_model, _, self.openclip_preprocess = open_clip.create_model_and_transforms(openclip_model_name, pretrained='laion2b_s34b_b82k_augreg_soup')
        self.openclip_model.eval().to(device)
        self.openclip_tokenizer = open_clip.get_tokenizer(openclip_model_name)

        self.use_question_query = use_question_query
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def filter_with_llava(self, top_k_images, question):
        start_time = time.time()
        filtered_images = []
        for img_path in top_k_images:
            llava_prompt = question + " Can this image provide the answer for this question? only answer yes or no."
            answer = llava_single_image_inference(img_path, llava_prompt)
            answer = answer.replace("only answer","")            
            if "yes" in answer.lower():
                filtered_images.append(img_path)
        
        if len(filtered_images) == 0:
            print("No images passed the LLaVA filter. Keeping the top-5 images.")
            filtered_images = top_k_images

        elapsed_time = time.time() - start_time
        return filtered_images, elapsed_time

    def get_combined_top_k_images(self, needle_word, pos_image, k=10, batch_size=100):
        start_time = time.time()
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png',".jpeg"))]
        image_paths = [os.path.join(self.image_dir, img) for img in image_files]

        text_inputs_clip = self.clip_processor(text=[needle_word], return_tensors="pt").to(self.device)
        text_inputs_siglip = self.siglip_processor(text=[needle_word], return_tensors="pt").to(self.device)
        text_inputs_openclip = self.openclip_tokenizer([needle_word]).to(self.device)

        with torch.no_grad():
            text_embeds_clip = self.clip_model.get_text_features(**text_inputs_clip).squeeze(0)
            text_embeds_clip /= text_embeds_clip.norm()

            text_embeds_siglip = self.siglip_model.get_text_features(**text_inputs_siglip).squeeze(0)
            text_embeds_siglip /= text_embeds_siglip.norm()

            text_embeds_openclip = self.openclip_model.encode_text(text_inputs_openclip)
            text_embeds_openclip /= text_embeds_openclip.norm(dim=-1, keepdim=True)

            combined_scores = {}

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                images_clip = [Image.open(img).convert("RGB") for img in batch_paths]
                image_inputs_clip = self.clip_processor(images=images_clip, return_tensors="pt").to(self.device)

                images_siglip = [Image.open(img).convert("RGB") for img in batch_paths]
                image_inputs_siglip = self.siglip_processor(images=images_siglip, return_tensors="pt").to(self.device)

                images_openclip = [self.openclip_preprocess(Image.open(img).convert("RGB")).to(self.device) for img in batch_paths]
                image_inputs_openclip = torch.stack(images_openclip)

                image_embeds_clip = self.clip_model.get_image_features(**image_inputs_clip)
                image_embeds_clip /= image_embeds_clip.norm(dim=1, keepdim=True)

                image_embeds_siglip = self.siglip_model.get_image_features(**image_inputs_siglip)
                image_embeds_siglip /= image_embeds_siglip.norm(dim=1, keepdim=True)

                image_embeds_openclip = self.openclip_model.encode_image(image_inputs_openclip)
                image_embeds_openclip /= image_embeds_openclip.norm(dim=-1, keepdim=True)

                cosine_similarities_clip = torch.matmul(image_embeds_clip, text_embeds_clip).cpu().numpy().flatten()
                cosine_similarities_siglip = torch.matmul(image_embeds_siglip, text_embeds_siglip).cpu().numpy().flatten()
                cosine_similarities_openclip = (image_embeds_openclip @ text_embeds_openclip.T).squeeze().cpu().numpy().flatten()

                for idx, img_path in enumerate(batch_paths):
                    combined_score = (cosine_similarities_clip[idx] + cosine_similarities_siglip[idx] +
                                      cosine_similarities_openclip[idx])
                    combined_scores[img_path] = combined_score

                del images_clip, image_inputs_clip, images_siglip, image_inputs_siglip, images_openclip, image_inputs_openclip
                torch.cuda.empty_cache()

            top_k_images = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
            top_k_includes_pos = [int(pos_image in top_k_images[:i + 1]) for i in range(k)]

        elapsed_time = time.time() - start_time
        return top_k_images, top_k_includes_pos, elapsed_time

    def process_dataset(self, dataset_file, output_dir, top_k_filter=20):


        with open(dataset_file, 'r') as f:
            data = json.load(f)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_entries = len(data)
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        top10_correct = 0

        total_combined_time = 0
        total_llava_filtering_time = 0

        for idx, entry in enumerate(data):
            question = entry["conversations"][0]["value"]

            id = entry["id"]


            if self.use_question_query:
                needle_word = question
            else:
                needle_word = entry["needle"]


            pos_image = os.path.join(self.image_dir, entry["pos_image"][0])

            top_k_images, top_k_includes_pos, combined_time = self.get_combined_top_k_images(needle_word, pos_image, k=10)
            total_combined_time += combined_time

            filtered_images, llava_filtering_time = self.filter_with_llava(top_k_images, question)
            total_llava_filtering_time += llava_filtering_time


            filtered_images = [img.split("/")[-1] for img in filtered_images]

            pos_image = pos_image.split("/")[-1]

            top_k_includes_pos_filtered = [int(pos_image in filtered_images[:i + 1]) for i in range(len(filtered_images))]

            if top_k_includes_pos_filtered[0]:
                top1_correct += 1
            if any(top_k_includes_pos_filtered[:3]):
                top3_correct += 1
            if any(top_k_includes_pos_filtered[:5]):
                top5_correct += 1
            if any(top_k_includes_pos_filtered[:10]):
                top10_correct += 1


            file_name = id+".json"
            output_file = os.path.join(output_dir, f"{file_name}")

            filtered_images = [img.split("/")[-1] for img in filtered_images]

            pos_image = pos_image.split(".")[0]
            with open(output_file, "w") as f_out:
                json.dump({
                    "question": question,
                    "needle_word": needle_word,
                    "top_10_images": filtered_images[:10],
                    "top_k_includes_pos": top_k_includes_pos_filtered,
                    "real_positive_image": pos_image
                }, f_out, indent=4)

            print(f"Processed entry {idx + 1}/{total_entries}: Saved top 5 images to {output_file}")

        top1_accuracy = top1_correct / total_entries
        top3_accuracy = top3_correct / total_entries
        top5_accuracy = top5_correct / total_entries
        top10_accuracy = top10_correct / total_entries

        avg_combined_time = total_combined_time / total_entries
        avg_llava_filtering_time = total_llava_filtering_time / total_entries

        log_file = os.path.join(output_dir, 'accuracy.log')

    
        with open(log_file, 'w') as log:
            log.write(f"Total Entries: {total_entries}\n")
            log.write(f"Top-1 Accuracy: {top1_accuracy:.2%}\n")
            log.write(f"Top-3 Accuracy: {top3_accuracy:.2%}\n")
            log.write(f"Top-5 Accuracy: {top5_accuracy:.2%}\n")
            log.write(f"Top-10 Accuracy: {top10_accuracy:.2%}\n")
            log.write(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds\n")
            log.write(f"Average LLaVA Filtering Time: {avg_llava_filtering_time:.4f} seconds\n")

        print(f"Total Entries: {total_entries}")
        print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
        print(f"Top-3 Accuracy: {top3_accuracy:.2%}")
        print(f"Top-5 Accuracy: {top5_accuracy:.2%}")
        print(f"Top-10 Accuracy: {top10_accuracy:.2%}")
        print(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds")
        print(f"Average LLaVA Filtering Time: {avg_llava_filtering_time:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SiGLIP Solver with adjustable dataset and image directory.")
    parser.add_argument('--dataset_file', type=str, default="dataset/VHs_qa/VHs_full/single_needle/infor_100.json", help="Path to the dataset JSON file")
    parser.add_argument('--image_root', type=str, default="dataset", help="Root directory for images")
    parser.add_argument('--image_dir', type=str, default="image/infor_100", help="Directory for specific images")
    parser.add_argument('--top_k_filter', type=int, default=60, help="Output directory for results")
    parser.add_argument('--output_dir', type=str, default="output/siglip_results", help="Output directory for results")
    parser.add_argument('--use_question_query', action='store_true', help="Use question as the query instead of needle word")
    args = parser.parse_args()

    solver = CombinedSolver(clip_model_name='openai/clip-vit-large-patch14', siglip_model_name="google/siglip-so400m-patch14-384", image_root=args.image_root, image_dir=args.image_dir, use_question_query=args.use_question_query)
    solver.process_dataset(args.dataset_file, args.output_dir, top_k_filter=args.top_k_filter)







