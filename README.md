# AI/ML Enabled Video Analysis Capstone

This project demonstrates an AI-powered pipeline for real-time video analysis and interpretation. It uses advanced deep learning models to perform object detection on video streams, and it is structured as a modular research project.

## Features
- **7 Pretrained/Custom Architectures**: Includes state-of-the-art models like `YOLO11n`, `YOLOv8n`, `YOLOv9c`, alongside `SSD300 VGG16`, `RetinaNet`, `FCOS ResNet50`, and a custom `Faster R-CNN` (modified with dropout and custom FC heads).
- **Extensive Dataset Integration**: Automated loaders for 7 standard benchmark datasets: *Pascal VOC 2007*, *Pascal VOC 2012*, *COCO128*, *PennFudanPed*, *Oxford-IIIT Pets*, *Caltech101*, and *WIDERFace*.
- **Advanced Data Augmentations**: Uses `Albumentations` for professional-grade spatial transformations, color jittering, and bounding-box aware augmentations.
- **Model Introspection**: Features a `LayerVisualizer` module that hooks into deep convolutional layers to extract and visualize real feature maps, alongside plotting computational graphs using `torchviz`.
- **Feature Analysis**: Includes a `FeatureAnalysis` module that extracts bottleneck embeddings from models and projects them using **t-SNE** and **PCA** scatter plots to analyze class separability.
- **Fine-tuning Pipeline**: A unified PyTorch training loop (`utils/finetuning.py`) that can adapt the custom standard models (like Faster R-CNN) to specific subsets.
- **Automated Performance Evaluation**: Measures Processing Speed (FPS) and Accuracy (mAP@50, Precision, Recall) via `torchmetrics`, visualizing the results using `pandas` and `matplotlib`.
- **Real-Time Webcam Detection**: Includes support for live inference via a local webcam feed.

## Project Structure
- `data/`: Contains `dataset_loader.py` (which handles the 7 datasets) and `augmentations.py` (handling `albumentations` logic).
- `models/`: Contains the core architectures (`base_model.py`, `yolo_model.py`, `custom_rcnn.py`, `ssd_model.py`, `retinanet_model.py`, `fcos_model.py`).
- `utils/`: Contains tools for calculating metrics (`evaluation.py`), processing video (`video_utils.py`), feature analysis (`feature_analysis.py`), intermediate layer hooking (`layer_visualizer.py`), and a fully featured model finetuning loop (`finetuning.py`).
- `generate_capstone_nb.py`: Orchestrator script to generate the fully configured Jupyter Notebook.
- `main.py`: The CLI orchestration script for live webcam/video analysis.
- `output/`: Stores the annotated output videos, feature graphs (PCA/t-SNE), and architectural plots.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/01anuraganand/video.git
   cd video
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: If you have an older NVIDIA GPU driver, you may need to install the CUDA 11.8 specific PyTorch version manually: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)*

## Running the Jupyter Notebook Pipeline (Recommended)

The central piece of this capstone research is the `capstone_evaluation.ipynb` notebook. It tests 5-7 models, plots architectures, evaluates metrics, runs t-SNE analytics, and tests the finetuning loop.

If the notebook doesn't exist or needs refreshing, regenerate it:
```bash
python generate_capstone_nb.py
```
Then start your Jupyter server:
```bash
jupyter notebook
```
Open `capstone_evaluation.ipynb` and run all cells.

## Running the CLI Pipeline

Run the `main.py` orchestrator script for video analysis:
```bash
python main.py
```

### Real-Time Webcam Inference
To run real-time inference using your computer's webcam, use the `--webcam` flag:
```bash
python main.py --webcam
```
*(Press 'q' inside the video window to quit the live feed. If running `both` models, quitting the first feed will automatically open the second.)*

### Options
You can run specific models using the `--model` flag:
- `python main.py --model yolo` (runs only YOLO11)
- `python main.py --model rcnn` (runs only the custom Faster R-CNN)

You can pass a custom video using the `--video` flag:
- `python main.py --video /path/to/your/video.mp4`
