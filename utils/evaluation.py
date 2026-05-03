import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import xml.etree.ElementTree as ET
import os

# Mapping VOC classes to IDs (simplified)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

def parse_voc_xml(node):
    res = []
    for obj in node.findall("object"):
        bndbox = obj.find("bndbox")
        res.append({
            "name": obj.find("name").text,
            "bbox": [
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            ]
        })
    return res

def evaluate_model_on_voc(model_wrapper, dataset_root='./data/voc', num_samples=50):
    print(f"Evaluating {model_wrapper.__class__.__name__} on VOC2007...")
    # Download and load VOC 2007 test set
    dataset = VOCDetection(root=dataset_root, year='2007', image_set='test', download=True)
    
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    device = model_wrapper.device
    model = model_wrapper.model
    model.eval()

    # We evaluate on a subset for speed in this capstone draft
    for i in range(min(num_samples, len(dataset))):
        img, target_dict = dataset[i]
        
        # Parse ground truth
        objects = parse_voc_xml(target_dict['annotation'])
        gt_boxes = []
        gt_labels = []
        for obj in objects:
            if obj['name'] in CLASS_TO_IDX:
                gt_boxes.append(obj['bbox'])
                gt_labels.append(CLASS_TO_IDX[obj['name']])
        
        if not gt_boxes:
            continue
            
        target = [
            dict(
                boxes=torch.tensor(gt_boxes, dtype=torch.float32).to(device),
                labels=torch.tensor(gt_labels, dtype=torch.int64).to(device)
            )
        ]

        # Predict
        img_tensor = F.to_tensor(img).to(device)
        with torch.no_grad():
            if hasattr(model, 'predict'): # YOLO
                results = model(img, verbose=False)
                # YOLO output parsing
                boxes = results[0].boxes.xyxy
                scores = results[0].boxes.conf
                labels = results[0].boxes.cls.to(torch.int64)
                preds = [dict(boxes=boxes.to(device), scores=scores.to(device), labels=labels.to(device))]
            else: # Torchvision
                preds = model([img_tensor])
                # Filter out low confidence
                keep = preds[0]['scores'] > 0.1
                preds = [dict(
                    boxes=preds[0]['boxes'][keep],
                    scores=preds[0]['scores'][keep],
                    labels=preds[0]['labels'][keep]
                )]

        metric.update(preds, target)
        
    results = metric.compute()
    
    # Return key metrics
    return {
        'mAP@50': results['map_50'].item(),
        'mAP@50-95': results['map'].item(),
        'Precision': results['mar_100'].item(), # approximation
        'Recall': results['mar_1'].item() # approximation
    }
