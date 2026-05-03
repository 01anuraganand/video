from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, device):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict_and_annotate(self, frame):
        """
        Takes a BGR image frame (numpy array), runs inference, 
        draws bounding boxes, and returns the annotated frame.
        """
        pass
