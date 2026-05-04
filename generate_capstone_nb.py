import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# AI/ML Video Analysis Capstone: Advanced Comparative Analysis\nThis notebook evaluates object detection architectures across multiple datasets. It covers data augmentation, architectural introspection, feature analysis, fine-tuning, and final benchmark evaluations on standard detection datasets like VOC2012 and COCO."))

cells.append(nbf.v4.new_code_cell("""import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn as nn

from models.yolo_model import YOLOModel
from models.custom_rcnn import CustomRCNNModel
from models.ssd_model import SSDModel
from models.retinanet_model import RetinaNetModel
from models.fcos_model import FCOSModel
from utils.video_utils import process_video_stream
from utils.evaluation import evaluate_model_on_voc, parse_voc_xml, CLASS_TO_IDX
from data.dataset_loader import DatasetLoader
from data.augmentations import get_train_transforms
from utils.layer_visualizer import LayerVisualizer
from utils.feature_analysis import perform_tsne_pca_analysis
from utils.finetuning import finetune_model

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on: {device}")
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Dataset Integration & Augmentation\nLoading standard benchmark object detection datasets (VOC2012, COCO128, COCO8, Open Images v7 - 500 images) and configuring Albumentations."))
cells.append(nbf.v4.new_code_cell("""loader = DatasetLoader()
print("Initializing dataset downloads and loaders...")

# Load Standard Object Detection datasets
voc2012 = loader.load_voc('2012')
coco128 = loader.load_coco128()
coco8 = loader.load_coco8()

# Large-scale datasets (Open Images v7 - 500 image validation subset)
open_images = loader.load_open_images(max_samples=500)

print(f"\\nDatasets loaded. For example, VOC 2012 has {len(voc2012)} images.")

# Initialize complex data augmentations
transforms = get_train_transforms()
print("Albumentations data augmentation pipeline configured.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Load Models & Plot Architectures\nInitializing models, printing their CNN layer weights, and plotting architecture diagrams."))
cells.append(nbf.v4.new_code_cell("""models = {
    'YOLO11n': YOLOModel(device, 'yolo11n.pt'),
    'YOLOv8n': YOLOModel(device, 'yolov8n.pt'),
    'YOLOv9c': YOLOModel(device, 'yolov9c.pt'),
    'Custom Faster R-CNN': CustomRCNNModel(device),
    'SSD300 VGG16': SSDModel(device),
    'RetinaNet': RetinaNetModel(device),
    'FCOS ResNet50': FCOSModel(device)
}

# Load weights
for name, model in models.items():
    model.load()
print("\\nAll models loaded successfully!")

# Select 5 models for detailed architectural analysis
analysis_models = {
    'Custom Faster R-CNN': models['Custom Faster R-CNN'],
    'SSD300 VGG16': models['SSD300 VGG16'],
    'RetinaNet': models['RetinaNet'],
    'FCOS ResNet50': models['FCOS ResNet50'],
    'YOLOv8n': models['YOLOv8n']
}

for name, model_wrapper in analysis_models.items():
    print(f"\\n{'='*50}\\nAnalyzing Architecture: {name}\\n{'='*50}")
    model_obj = getattr(model_wrapper, 'model', model_wrapper)
    visualizer = LayerVisualizer(model_obj)
    
    # 1. Print actual CNN layer values/statistics
    visualizer.print_layer_values()
    
    # 2. Plot Architecture Diagram using torchviz
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    visualizer.plot_architecture(dummy_input, f"{name.replace(' ', '_')}_arch")
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Real Feature Extraction Analysis (t-SNE & PCA)\nPassing real VOC images through the network to extract bottleneck features and project them into 2D space."))
cells.append(nbf.v4.new_code_cell("""# Extract real features from Custom Faster R-CNN backbone using 50 real VOC images
rcnn_model = models['Custom Faster R-CNN'].model
visualizer = LayerVisualizer(rcnn_model)

# Find a deep CNN layer to hook into
target_layer = 'backbone.body.layer4'
visualizer.register_hooks([target_layer])

extracted_features = []
labels = []
print("Extracting real features from 50 VOC2012 images...")

rcnn_model.eval()
with torch.no_grad():
    for i in range(50):
        img, target_dict = voc2012[i]
        img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
        
        # Forward pass to trigger hook
        try:
             rcnn_model(img_tensor)
             act = visualizer.activations[target_layer]
             # Global Average Pooling to get a 1D feature vector
             feat = act.mean(dim=[2,3]).squeeze(0).cpu().numpy()
             extracted_features.append(feat)
             
             # Extract simple label (first object in image)
             objects = target_dict['annotation'].get('object', [])
             if not isinstance(objects, list):
                 objects = [objects]
             lbl = objects[0]['name'] if objects else 'background'
             labels.append(lbl)
        except Exception as e:
             continue

visualizer.remove_hooks()

if extracted_features:
    extracted_features = np.array(extracted_features)
    # Convert string labels to ints for plotting
    unique_labels = list(set(labels))
    int_labels = [unique_labels.index(l) for l in labels]
    
    print(f"Extracted feature matrix of shape: {extracted_features.shape}")
    perform_tsne_pca_analysis(extracted_features, int_labels)
else:
    print("Failed to extract features.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Fine-Tuning Pipeline\nFine-tuning the network on a targeted subset of real images."))
cells.append(nbf.v4.new_code_cell("""# Create a dataset wrapper to convert VOC format into PyTorch tensor targets
class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, voc_dataset, max_samples=10):
        self.voc = voc_dataset
        self.max_samples = min(max_samples, len(voc_dataset))
        
    def __len__(self):
        return self.max_samples
        
    def __getitem__(self, idx):
        img, target_dict = self.voc[idx]
        img_tensor = F.to_tensor(img)
        
        objects = parse_voc_xml(target_dict['annotation'])
        gt_boxes, gt_labels = [], []
        for obj in objects:
            if obj['name'] in CLASS_TO_IDX:
                gt_boxes.append(obj['bbox'])
                gt_labels.append(CLASS_TO_IDX[obj['name']])
        
        if len(gt_boxes) == 0:
            gt_boxes = [[0, 0, 1, 1]] # Fallback
            gt_labels = [0]
            
        target = {
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32),
            'labels': torch.tensor(gt_labels, dtype=torch.int64)
        }
        return img_tensor, target

ft_dataset = FinetuneDataset(voc2012, max_samples=8)

# Finetuning the Custom Faster R-CNN on the subset
print("Starting finetuning loop on Custom Faster R-CNN...")
finetune_model(models['Custom Faster R-CNN'], ft_dataset, num_epochs=2, batch_size=2)
print("Finetuning pipeline executed successfully.")
"""))


cells.append(nbf.v4.new_markdown_cell("## 5. Model Accuracy Evaluation\nCalculating Mean Average Precision (mAP), Precision, and Recall using `torchmetrics` on the validation subset (VOC2012)."))
cells.append(nbf.v4.new_code_cell("""accuracy_results = {}
for name, model in models.items():
    res = evaluate_model_on_voc(model, num_samples=30)
    accuracy_results[name] = res

df_acc = pd.DataFrame(accuracy_results).T
display(df_acc)
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Inference Speed Benchmark (FPS)\nMeasuring real-time throughput on a live pedestrian video stream."))
cells.append(nbf.v4.new_code_cell("""loader = DatasetLoader()
video_path = loader.download_sample_video()

fps_results = {}
for name, model in models.items():
    out_path = f"output/{name.replace(' ', '_')}_out.mp4"
    fps = process_video_stream(model, video_path, out_path, num_frames=30)
    fps_results[name] = fps

df_fps = pd.DataFrame.from_dict(fps_results, orient='index', columns=['FPS'])
display(df_fps)
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Final Comparative Visualizations\nPlotting accuracy vs throughput."))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy Plot
df_acc['mAP@50'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Accuracy (mAP@50)')
axes[0].set_ylabel('mAP')
axes[0].tick_params(axis='x', rotation=45)

# Speed Plot
df_fps['FPS'].plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title(f'Inference Speed (FPS) on {device.type.upper()}')
axes[1].set_ylabel('Frames Per Second')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""))

nb['cells'] = cells

with open('capstone_evaluation.ipynb', 'w') as f:
    nbf.write(nb, f)
print("capstone_evaluation.ipynb generated.")
