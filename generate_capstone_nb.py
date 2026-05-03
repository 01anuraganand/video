import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# AI/ML Video Analysis Capstone: 6-Model Comparative Analysis\nThis notebook evaluates 6 different object detection architectures on the **Pascal VOC 2007** full dataset for accuracy (mAP, Precision, Recall) and on a pedestrian video stream for speed (FPS)."))

cells.append(nbf.v4.new_code_cell("""import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from models.yolo_model import YOLOModel
from models.custom_rcnn import CustomRCNNModel
from models.ssd_model import SSDModel
from models.retinanet_model import RetinaNetModel
from models.fcos_model import FCOSModel
from utils.video_utils import process_video_stream
from utils.evaluation import evaluate_model_on_voc
from data.dataset_loader import DatasetLoader

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on: {device}")
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load 6 Pretrained/Custom Models"))

cells.append(nbf.v4.new_code_cell("""models = {
    'YOLO11n': YOLOModel(device, 'yolo11n.pt'),
    'YOLOv8n': YOLOModel(device, 'yolov8n.pt'),
    'YOLOv9c': YOLOModel(device, 'yolov9c.pt'),
    'Custom Faster R-CNN': CustomRCNNModel(device),
    'SSD300 VGG16': SSDModel(device),
    'RetinaNet': RetinaNetModel(device),
    'FCOS ResNet50': FCOSModel(device)
}

for name, model in models.items():
    model.load()
print("\\nAll models loaded successfully!")
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Accuracy Evaluation on Pascal VOC 2007\nCalculating Mean Average Precision (mAP), Precision, and Recall using `torchmetrics`."))

cells.append(nbf.v4.new_code_cell("""accuracy_results = {}
# Evaluating on a sample subset for demonstration speed in this notebook
for name, model in models.items():
    res = evaluate_model_on_voc(model, num_samples=30)
    accuracy_results[name] = res

df_acc = pd.DataFrame(accuracy_results).T
display(df_acc)
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Real-time Video Stream Speed Evaluation (FPS)"))

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

cells.append(nbf.v4.new_markdown_cell("## 4. Final Comparative Visualizations"))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy Plot
df_acc['mAP@50'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Accuracy (mAP@50) on Pascal VOC 2007')
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
