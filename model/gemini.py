# code base: https://ai.google.dev/gemini-api/docs/vision?hl=zh-cn&lang=python

import os
import google.generativeai as genai
from base import BaseModel
from PIL import Image

genai.configure(api_key="Your API Key")

class EvalGemini(BaseModel):
    
    def __init__(self, model_name = "gemini", low_res = False, upload = False, prompt = None, api_key = "gemini-1.5-pro"):
        
        """
        Args:
            model_name: str, baseline model to use
            low_res: bool, whether to use low resolution images
            upload: bool, whether to upload the image to server, gemini can only process images from server when input images is too much
            prompt: str, whether to constraint the model output space
            api_key: str, api key of gemini to use
        """
        
        self.model_name = model_name
        self.low_res = low_res
        self.upload = upload
        self.prompt = prompt
        self.api_key = api_key
        self.model = genai.GenerativeModel(model_name = api_key)

        self.exist_images = {}

    def encode_image(self, image_path):
        
        if not self.upload:
            image =  Image.open(image_path)
            if self.low_res:
                return image.resize((image.size[0]//self.scale_factor, image.size[1]//self.scale_factor))

        image_name = os.path.basename(image_path)
        if image_name in self.exist_images:
            image = genai.get_file(name = self.exist_images[image_name].name)
        else:
            image = genai.upload_file(path = image_path, display_name = image_name)
            self.exist_images[image_name] = image
        
        return image

    def batch_image(self, image_path) -> list:
        
        assert type(image_path) == list or os.path.isdir(image_path), f"image_path should be a directory or a list of image paths, but given {image_path}"
        
        if type(image_path) == list:
            images = image_path
        else:
            images = [os.path.join(image_path, image) for image in os.listdir(image_path)]
        
        images = [self.encode_image(image) for image in images]

        return images

    def generate(self, question, images):
        """
        Args:
            question: str, question to ask
            images: list, list of multiple image features
        """
        if self.prompt is not None:
            question = f"{question}\n{self.prompt}"

        inputs = [question] + images

        return self.model.generate_content(inputs).text
    
if __name__ == "__main__":
    
    model = EvalGemini()
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