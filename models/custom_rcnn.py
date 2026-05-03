import torch
import torch.nn as nn
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from .base_model import BaseModel

class CustomBoxPredictor(nn.Module):
    """
    Custom prediction head to replace the default one in Faster R-CNN.
    This demonstrates the ability to modify network architecture with custom layers.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.cls_score = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Custom added layer
            nn.Linear(512, num_classes)
        )
        self.bbox_pred = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes * 4)
        )

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

class CustomRCNNModel(BaseModel):
    def __init__(self, device):
        super().__init__(device)
        self.num_classes = 91 # Standard COCO classes
        
    def load(self):
        print(f"Loading Custom Faster R-CNN onto {self.device}...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        
        # Modify the architecture
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = CustomBoxPredictor(in_features, self.num_classes)
        
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        print("Custom Faster R-CNN Model loaded with custom Dropout head.")

    def predict_and_annotate(self, frame):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Convert BGR to RGB, then to tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
            
        # Draw boxes manually for RCNN
        annotated_frame = frame.copy()
        for i in range(len(prediction['boxes'])):
            if prediction['scores'][i] > 0.5: # Confidence threshold
                box = prediction['boxes'][i].cpu().numpy().astype(int)
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
        return annotated_frame
