# code base: https://github.com/QwenLM/Qwen2-VL

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
from PIL import Image
from base import BaseModel

class EvalQwen2VL(BaseModel):

    def __init__(self, model_name = "qwen2_vl", pretrained = "Qwen/Qwen2-VL-7B-Instruct", low_res = False, scale_factor = 4, prompt = None):
        
        """
        Args:
            model_name: str, baseline model to use
            pretrained: str, path to the pretrained model
            low_res: bool, whether to use low resolution images
            scale_factor: int, scale factor to resize the image
            prompt: str, whether to constraint the model output space
        """

        self.model_name = model_name
        self.low_res = low_res
        self.scale_factor = scale_factor
        self.prompt = prompt

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype = torch.bfloat16,
            attn_implementation = "flash_attention_2",
            device_map = "auto",
        )
        self.model.eval()

        self.template = [
            {
                "role": "user",
                "content": [],
            }
        ]

    def encode_image(self, image_path):
        
        return Image.open(image_path).convert("RGB")

    def batch_image(self, image_path) -> list:
        
        # clear cached images
        self.template = [
            {
                "role": "user",
                "content": [],
            }
        ]
        
        assert type(image_path) == list or os.path.isdir(image_path), f"image_path should be a directory or a list of image paths, but given {image_path}"
        
        if type(image_path) == list:
            images = image_path
        else:
            images = [os.path.join(image_path, image) for image in os.listdir(image_path)]

        for image in images:
            image = self.encode_image(image)
            if self.low_res:
                self.template[0]["content"].append(
                    {
                        "type": "image",
                        "image": image,
                        "resized_width": image.size[0] // self.scale_factor,
                        "resized_height": image.size[1] // self.scale_factor,
                    }
                )
            else:
                self.template[0]["content"].append(
                    {
                        "type": "image",
                        "image": image,
                    }
                )

        image_inputs, video_inputs = process_vision_info(self.template)

        return image_inputs

    @torch.no_grad()
    def generate(self, question, images):
        """
        Args:
            question: str, question to ask
            images: list, list of multiple image features
        """
        if self.prompt is not None:
            question = f"{question}\n{self.prompt}"

        self.template[0]["content"].append(
            {
                "type": "text",
                "text": question
            }
        )
        text = self.processor.apply_chat_template(self.template, tokenize = False, add_generation_prompt = True)

        inputs = self.processor(
            text = [text],
            images = images,
            videos = None,
            padding = True,
            return_tensors = "pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens = 128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        self.template[0]["content"].pop()

        return text_outputs[0]
    
if __name__ == "__main__":
    
    model = EvalQwen2VL(low_res = True, scale_factor = 4)
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