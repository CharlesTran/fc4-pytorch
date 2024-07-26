from classes.fc4.tripletatten.TripletAttention import TripletAttention

class TripletAttentionLoader:
    def __init__(self, version: float = 1.1):
        self.__model = TripletAttention(self.__version)
        
    def load(self, pretrained: bool = False):
        return self.__model