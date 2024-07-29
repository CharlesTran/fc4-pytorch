from classes.fc4.tripletatten.resnet import ResidualNet

class TripletAttentionLoader:
    def __init__(self, version: float = 1.1):
        self.__model = ResidualNet(network_type="ImageNet", depth=50, att_type="TripletAttention")
        
    def load(self, pretrained: bool = False):
        return self.__model