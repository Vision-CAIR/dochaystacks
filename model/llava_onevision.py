# code base: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision_Tutorials.ipynb

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import os
import copy
import torch
import warnings
from PIL import Image
from base import BaseModel

class EvalLLaVAOV(BaseModel):

    def __init__(self, model_name = "llava_onevision", pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov", low_res = False, scale_factor = 4, no_patch = False, prompt = None):
        
        """
        Args:
            model_name: str, baseline model to use
            pretrained: str, path to the pretrained model
            low_res: bool, whether to use low resolution images
            scale_factor: int, scale factor to resize the image
            no_patch: bool, whether to use patch for each image
            prompt: str, whether to constraint the model output space
        """

        self.model_name = model_name
        self.low_res = low_res
        self.scale_factor = scale_factor
        self.no_patch = no_patch
        self.prompt = prompt
        self.conv_template = "qwen_1_5"

        warnings.filterwarnings("ignore")
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, "llava_qwen", device_map = "auto", **llava_model_args)  # Add any other thing you want to pass in llava_model_args
        self.model.eval()

    def encode_image(self, image_path):
        
        image =  Image.open(image_path).convert("RGB")

        if self.low_res:
            return image.resize((image.size[0]//self.scale_factor, image.size[1]//self.scale_factor))

        return image

    def batch_image(self, image_path) -> list:
        
        assert type(image_path) == list or os.path.isdir(image_path), f"image_path should be a directory or a list of image paths, but given {image_path}"
        
        if type(image_path) == list:
            images = image_path
        else:
            images = [os.path.join(image_path, image) for image in os.listdir(image_path)]

        images = [self.encode_image(image) for image in images]
        self.image_sizes = [image.size for image in images]
        
        if self.no_patch:
            self.image_tensors = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].half().cuda()
        else:
            image_tensors = process_images(images, self.image_processor, self.model.config)
            self.image_tensors = [_image.to(dtype = torch.float16, device = "cuda") for _image in image_tensors]
        
        return self.image_tensors

    @torch.no_grad()
    def generate(self, question, images):
        """
        Args:
            question: str, question to ask
            images: list, list of multiple image features
        """
        if self.prompt is not None:
            question = f"{question}\n{self.prompt}"

        question = DEFAULT_IMAGE_TOKEN * len(images) + f"\n{question}"
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        
        # Generate response
        cont = self.model.generate(
            input_ids,
            images = images,
            image_sizes = self.image_sizes,
            do_sample = False,
            temperature = 0,
            max_new_tokens = 4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens = True)
        
        return text_outputs[0]
        
if __name__ == "__main__":
    
    model = EvalLLaVAOV(no_patch = True)
    images = model.batch_image("./img")
    
    print("\n***************************Instance1*********************************\n")
    q = "Find the dog and tell me what breed is this dog in this set of images"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")

    print("\n***************************Instance2*********************************\n")
    q = "Where does the bear sit in?"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")

    print("\n***************************Instance3*********************************\n")
    q = "Is there a white horse in this set of image?"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")

    print("\n***************************Instance4*********************************\n")
    q = "Is there a black horse in this set of image?"
    resps = model.generate(q, images)
    print(f"Question: {q}\nAnswer: {resps}")
