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

cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Integration & Augmentation
Loading standard benchmark object detection datasets (VOC2012, COCO128, COCO8, Open Images v7)
and visualising sample images with bounding boxes, segmentation info, and raw annotation values."""))

cells.append(nbf.v4.new_code_cell("""loader = DatasetLoader(data_dir='data')
print("Initializing dataset downloads and loaders...")

voc2012     = loader.load_voc('2012')
coco128_dir = loader.load_coco128()
coco8_dir   = loader.load_coco8()
open_images = loader.load_open_images(max_samples=500)

print(f"\\nVOC 2012 : {len(voc2012)} images")
print(f"COCO128  : {coco128_dir}")
print(f"COCO8    : {coco8_dir}")
print(f"Open Img : {len(open_images) if open_images else 'N/A'} images")
"""))

# ── 1a. Per-dataset visualisation ──────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("### 1a. Dataset Sample Visualisation\nShowing 4 sample images per dataset with bounding boxes overlaid and raw annotation values printed."))

cells.append(nbf.v4.new_code_cell("""import os, glob, cv2, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data.augmentations import COCO_CLASSES, VOC_CLASSES

def draw_boxes(ax, image, boxes, labels, class_names, title=""):
    ax.imshow(image)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')
    colors = plt.cm.Set2.colors
    for (x1, y1, x2, y2), lbl in zip(boxes, labels):
        color = colors[int(lbl) % len(colors)]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        cname = class_names[int(lbl)] if int(lbl) < len(class_names) else str(int(lbl))
        ax.text(x1, y1-4, cname, fontsize=7, color='white',
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'))

# ── VOC 2012 ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("VOC 2012  |  bbox format: [xmin ymin xmax ymax] (absolute px)")
print("=" * 60)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("VOC 2012 — Samples with Bounding Boxes", fontsize=13, fontweight='bold')
for i, ax in enumerate(axes):
    pil_img, ann = voc2012[i]
    img = np.array(pil_img)
    objs = ann['annotation'].get('object', [])
    if not isinstance(objs, list): objs = [objs]
    boxes, labels = [], []
    print(f"\\n  Image {i}: {ann['annotation']['filename']}  size={img.shape[:2]}")
    for obj in objs:
        bb = obj['bndbox']
        x1,y1,x2,y2 = int(float(bb['xmin'])),int(float(bb['ymin'])),int(float(bb['xmax'])),int(float(bb['ymax']))
        lbl = VOC_CLASSES.index(obj['name']) if obj['name'] in VOC_CLASSES else 0
        boxes.append([x1,y1,x2,y2]); labels.append(lbl)
        print(f"    {obj['name']:15s} bbox=[{x1:4d},{y1:4d},{x2:4d},{y2:4d}]  difficult={obj.get('difficult','0')}")
    draw_boxes(ax, img, boxes, labels, VOC_CLASSES, f"VOC #{i}")
plt.tight_layout(); plt.show()

# ── COCO8 ─────────────────────────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("COCO8  |  label format: class cx cy w h (normalised 0-1)")
print("=" * 60)
coco8_imgs = sorted(glob.glob(os.path.join(coco8_dir, 'images', 'train', '*.jpg')))[:4]
fig, axes = plt.subplots(1, len(coco8_imgs), figsize=(18, 5))
fig.suptitle("COCO8 — Samples with Bounding Boxes", fontsize=13, fontweight='bold')
for ax, img_path in zip(axes, coco8_imgs):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(coco8_dir, 'labels', 'train', stem + '.txt')
    boxes, labels = [], []
    print(f"\\n  {os.path.basename(img_path)}  ({w}x{h})")
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                p = line.strip().split()
                cls,cx,cy,bw,bh = int(p[0]),float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h)
                x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)
                boxes.append([x1,y1,x2,y2]); labels.append(cls)
                cname = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
                print(f"    {cname:20s} raw=[{cls} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}]  abs=[{x1},{y1},{x2},{y2}]")
    draw_boxes(ax, img, boxes, labels, COCO_CLASSES, os.path.basename(img_path))
plt.tight_layout(); plt.show()

# ── COCO128 ───────────────────────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("COCO128  |  label format: class cx cy w h (normalised 0-1)")
print("=" * 60)
c128_imgs = sorted(glob.glob(os.path.join(coco128_dir, 'images', 'train2017', '*.jpg')))[:4]
fig, axes = plt.subplots(1, len(c128_imgs), figsize=(18, 5))
fig.suptitle("COCO128 — Samples with Bounding Boxes", fontsize=13, fontweight='bold')
for ax, img_path in zip(axes, c128_imgs):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(coco128_dir, 'labels', 'train2017', stem + '.txt')
    boxes, labels = [], []
    print(f"\\n  {os.path.basename(img_path)}  ({w}x{h})")
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                p = line.strip().split()
                cls,cx,cy,bw,bh = int(p[0]),float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h)
                x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)
                boxes.append([x1,y1,x2,y2]); labels.append(cls)
                cname = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
                print(f"    {cname:20s} raw=[{cls} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}]  abs=[{x1},{y1},{x2},{y2}]")
    draw_boxes(ax, img, boxes, labels, COCO_CLASSES, os.path.basename(img_path))
plt.tight_layout(); plt.show()

# ── Open Images v7 ────────────────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("Open Images v7  |  bbox: [top-left-x top-left-y width height] (normalised)")
print("=" * 60)
if open_images:
    oi_samples = list(open_images.take(4))
    fig, axes = plt.subplots(1, len(oi_samples), figsize=(18, 5))
    fig.suptitle("Open Images v7 — Samples with Bounding Boxes", fontsize=13, fontweight='bold')
    for ax, sample in zip(axes, oi_samples):
        img = cv2.cvtColor(cv2.imread(sample.filepath), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, labels = [], []
        dets = sample.get_field('detections')
        print(f"\\n  {os.path.basename(sample.filepath)}  ({w}x{h})")
        if dets and hasattr(dets, 'detections'):
            for det in dets.detections[:6]:
                bx,by,bw,bh = det.bounding_box
                x1=int(bx*w); y1=int(by*h); x2=int((bx+bw)*w); y2=int((by+bh)*h)
                boxes.append([x1,y1,x2,y2]); labels.append(0)
                print(f"    {det.label:25s} normalised=[{bx:.4f},{by:.4f},{bw:.4f},{bh:.4f}]  abs=[{x1},{y1},{x2},{y2}]")
        draw_boxes(ax, img, boxes, labels, [det.label for det in (dets.detections if dets else [])], os.path.basename(sample.filepath))
    plt.tight_layout(); plt.show()
else:
    print("Open Images not loaded — skipping.")
"""))

# ── 1b. Augmentation visualisation ─────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("### 1b. Augmentation Preview — Before vs After\nShowing the same sample from each dataset before and after the full training augmentation pipeline (resize→flip→colour jitter→noise→normalise)."))

cells.append(nbf.v4.new_code_cell("""import torch
import albumentations as A
from data.augmentations import (get_train_transforms, get_val_transforms,
                                 VOCDatasetWrapper, YOLODatasetWrapper,
                                 OpenImagesDatasetWrapper, COCO_CLASSES, VOC_CLASSES)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_rgb(t):
    \"\"\"Undo ImageNet normalisation for display.\"\"\"
    img = t.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)

train_tf = get_train_transforms()

# Collect one raw sample from each dataset
raw_samples = []

# VOC
pil_img, ann = voc2012[0]
objs = ann['annotation'].get('object', [])
if not isinstance(objs, list): objs = [objs]
voc_boxes, voc_labels = [], []
for obj in objs:
    bb = obj['bndbox']
    x1,y1,x2,y2 = int(float(bb['xmin'])),int(float(bb['ymin'])),int(float(bb['xmax'])),int(float(bb['ymax']))
    lbl = VOC_CLASSES.index(obj['name']) if obj['name'] in VOC_CLASSES else 0
    voc_boxes.append([x1,y1,x2,y2]); voc_labels.append(lbl)
raw_samples.append(("VOC 2012", np.array(pil_img), voc_boxes, voc_labels, VOC_CLASSES))

# COCO8
img_path = sorted(glob.glob(os.path.join(coco8_dir,'images','train','*.jpg')))[0]
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]
lbl_path = os.path.join(coco8_dir,'labels','train', os.path.splitext(os.path.basename(img_path))[0]+'.txt')
c8_boxes, c8_labels = [], []
if os.path.exists(lbl_path):
    for line in open(lbl_path):
        p=line.strip().split(); cls,cx,cy,bw,bh=int(p[0]),float(p[1]),float(p[2]),float(p[3]),float(p[4])
        c8_boxes.append([int((cx-bw/2)*w),int((cy-bh/2)*h),int((cx+bw/2)*w),int((cy+bh/2)*h)]); c8_labels.append(cls)
raw_samples.append(("COCO8", img, c8_boxes, c8_labels, COCO_CLASSES))

# COCO128
img_path = sorted(glob.glob(os.path.join(coco128_dir,'images','train2017','*.jpg')))[0]
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]
lbl_path = os.path.join(coco128_dir,'labels','train2017', os.path.splitext(os.path.basename(img_path))[0]+'.txt')
c12_boxes, c12_labels = [], []
if os.path.exists(lbl_path):
    for line in open(lbl_path):
        p=line.strip().split(); cls,cx,cy,bw,bh=int(p[0]),float(p[1]),float(p[2]),float(p[3]),float(p[4])
        c12_boxes.append([int((cx-bw/2)*w),int((cy-bh/2)*h),int((cx+bw/2)*w),int((cy+bh/2)*h)]); c12_labels.append(cls)
raw_samples.append(("COCO128", img, c12_boxes, c12_labels, COCO_CLASSES))

# Open Images
if open_images:
    s = list(open_images.take(1))[0]
    img = cv2.cvtColor(cv2.imread(s.filepath), cv2.COLOR_BGR2RGB)
    h,w = img.shape[:2]
    oi_boxes, oi_labels = [], []
    dets = s.get_field('detections')
    if dets and hasattr(dets,'detections'):
        for det in dets.detections[:5]:
            bx,by,bw_n,bh_n = det.bounding_box
            oi_boxes.append([int(bx*w),int(by*h),int((bx+bw_n)*w),int((by+bh_n)*h)])
            oi_labels.append(0)
    raw_samples.append(("Open Images v7", img, oi_boxes, oi_labels, COCO_CLASSES))

# ── Plot before / after ──────────────────────────────────────────────────────
n = len(raw_samples)
fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
fig.suptitle("Augmentation Preview — Before vs After (640×640 letterbox)", fontsize=14, fontweight='bold')

for row, (name, raw_img, boxes, labels, cls_names) in enumerate(raw_samples):
    # Before
    ax_before = axes[row, 0]
    ax_before.set_title(f"{name} — Original", fontsize=10)
    draw_boxes(ax_before, raw_img, boxes, labels, cls_names)

    # After augmentation
    boxes_valid = [b for b in boxes if b[2]>b[0] and b[3]>b[1]] or [[0,0,1,1]]
    labels_valid = labels[:len(boxes_valid)] if labels else [0]
    result = train_tf(image=raw_img.copy(), bboxes=boxes_valid, class_labels=labels_valid)
    aug_img  = tensor_to_rgb(result['image'])
    aug_boxes  = [[int(x) for x in b] for b in result['bboxes']]
    aug_labels = list(result['class_labels'])

    ax_after = axes[row, 1]
    ax_after.set_title(f"{name} — Augmented (640×640)", fontsize=10)
    draw_boxes(ax_after, aug_img, aug_boxes, aug_labels, cls_names)

    print(f"[{name}] original={raw_img.shape[:2]}  →  augmented=640×640  boxes kept={len(aug_boxes)}")

plt.tight_layout(); plt.show()
"""))

# ── 1c. Combined dataset + DataLoader ───────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("### 1c. Combined Dataset → DataLoader for Models\nAll four datasets are merged into a single `ConcatDataset` with the full training augmentation pipeline. A `DataLoader` is constructed and used to feed batches to each model."))

cells.append(nbf.v4.new_code_cell("""from torch.utils.data import DataLoader
from data.augmentations import build_combined_dataset
import torch

def collate_fn(batch):
    \"\"\"Custom collate: images stacked, boxes/labels kept as lists.\"\"\"
    images  = torch.stack([b[0] for b in batch])
    boxes   = [b[1] for b in batch]
    labels  = [b[2] for b in batch]
    return images, boxes, labels

print("Building combined dataset from all sources...")
combined_ds = build_combined_dataset(
    voc_dataset        = voc2012,
    coco128_dir        = coco128_dir,
    coco8_dir          = coco8_dir,
    open_images_dataset= open_images,
    mode               = 'train',
)

combined_loader = DataLoader(
    combined_ds,
    batch_size  = 4,
    shuffle     = True,
    num_workers = 2,
    collate_fn  = collate_fn,
)

# Sanity-check: pull one batch and print shapes
images, boxes, labels = next(iter(combined_loader))
print(f"\\nSample batch — images tensor : {images.shape}")
print(f"               boxes (list)  : {[len(b) for b in boxes]} objects per image")
print(f"               labels (list) : {[len(l) for l in labels]} labels  per image")
print("\\n✓ DataLoader ready. Batches flow directly into model forward passes below.")
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
