from ultralytics import YOLO
from .base_model import BaseModel

class YOLOModel(BaseModel):
    def __init__(self, device, weights_path='yolov8n.pt'):
        super().__init__(device)
        self.weights_path = weights_path

    def load(self):
        print(f"Loading YOLO model from {self.weights_path} onto {self.device}...")
        self.model = YOLO(self.weights_path)
        self.model.to(self.device)

    def predict_and_annotate(self, frame):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # YOLOv8 returns a list of Results objects
        results = self.model(frame, verbose=False, device=self.device)
        annotated_frame = results[0].plot()
        return annotated_frame
