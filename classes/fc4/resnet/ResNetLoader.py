import os

from torch.utils import model_zoo

from classes.fc4.resnet.ResNet import ResidualNet

class ResNetLoader:
    def __init__(self, version: int):
        self.__version = version
        self.__model = ResidualNet('CIFAR10', version)

    def load(self, pretrained = False):
        return self.__model