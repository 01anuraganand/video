# AI/ML Enabled Video Analysis Capstone

This project demonstrates an AI-powered pipeline for real-time video analysis and interpretation. It uses advanced deep learning models to perform object detection on video streams, and it is structured as a modular research project.

## Features
- **Dynamic Hardware Acceleration**: Automatically runs on a CUDA-enabled GPU if available, or falls back to CPU.
- **Latest Pretrained Models**: Includes the state-of-the-art **YOLO11** architecture out of the box for high-speed inference.
- **Custom Architectures**: Includes a customized PyTorch **Faster R-CNN** implementation where the classification head has been modified (with additional Dropout and fully connected layers) to demonstrate architectural customization.
- **Automated Dataset Handling**: Automatically downloads a realistic pedestrian/traffic detection video dataset for inference.
- **Real-Time Webcam Detection**: Includes support for live inference via a local webcam feed.
- **Performance Evaluation**: Measures processing speed (FPS) and generates comparative bar charts.

## Project Structure
- `data/`: Contains the dataset loaders (`dataset_loader.py`) and downloaded video data.
- `models/`: Contains the core architectures (`base_model.py`, `yolo_model.py`, `custom_rcnn.py`).
- `utils/`: Contains tools for calculating metrics and processing video frames (`metrics.py`, `video_utils.py`).
- `main.py`: The main orchestration script to run the analysis.
- `output/`: Stores the annotated output videos and performance charts.

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

## Running the Pipeline

Run the `main.py` orchestrator script:
```bash
python main.py
```

By default, this will process a sample video using both YOLO11 and the Custom Faster R-CNN and save outputs to the `output/` folder.

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
