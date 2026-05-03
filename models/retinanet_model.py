import torch
import cv2
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from .base_model import BaseModel

class RetinaNetModel(BaseModel):
    def load(self):
        print(f"Loading RetinaNet onto {self.device}...")
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = retinanet_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def predict_and_annotate(self, frame):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
        
        annotated_frame = frame.copy()
        for i in range(len(prediction['boxes'])):
            if prediction['scores'][i] > 0.5:
                box = prediction['boxes'][i].cpu().numpy().astype(int)
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        return annotated_frame
