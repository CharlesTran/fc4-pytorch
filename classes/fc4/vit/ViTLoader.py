import os
from classes.fc4.vit.ViT import ViT

class ViTLoader:
    def __init__(self, version: float = 1.1):
        self.__version = version
        self.__model = ViT()

    def load(self, pretrained: bool = False) -> ViT:
        return self.__model