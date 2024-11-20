import os
from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def encode_image(self, image_path):
        """
        the method to encode image of different models
        """
        pass
    
    @abstractmethod
    def generate(self):
        """
        the method to response/infer of different models
        """
        pass
