import argparse
import torch
import os
from data.dataset_loader import DatasetLoader
from models.yolo_model import YOLOModel
from models.custom_rcnn import CustomRCNNModel
from utils.video_utils import process_video_stream
from utils.metrics import plot_fps_comparison

def main():
    parser = argparse.ArgumentParser(description="AI/ML Video Analysis Capstone Pipeline")
    parser.add_argument('--model', type=str, choices=['yolo', 'rcnn', 'both'], default='both', help='Model to run')
    parser.add_argument('--video', type=str, default=None, help='Path to input video')
    args = parser.parse_args()

    # Determine device dynamically
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Initialization ---")
    print(f"Device selected: {device}")

    # Prepare data
    loader = DatasetLoader()
    video_path = args.video
    if not video_path:
        video_path = loader.download_sample_video()
    
    os.makedirs('output', exist_ok=True)
    results = {}

    # Run YOLO if selected
    if args.model in ['yolo', 'both']:
        print("\\n--- Running YOLO11 Pipeline ---")
        yolo = YOLOModel(device)
        yolo.load()
        out_path = os.path.join('output', 'yolo_out.mp4')
        fps = process_video_stream(yolo, video_path, out_path, num_frames=50)
        results['YOLO11'] = fps

    # Run Custom R-CNN if selected
    if args.model in ['rcnn', 'both']:
        print("\\n--- Running Custom Faster R-CNN Pipeline ---")
        rcnn = CustomRCNNModel(device)
        rcnn.load()
        out_path = os.path.join('output', 'rcnn_out.mp4')
        fps = process_video_stream(rcnn, video_path, out_path, num_frames=50)
        results['Custom R-CNN'] = fps

    # Comparison
    if len(results) > 1:
        print("\\n--- Performance Comparison ---")
        plot_fps_comparison(results)

if __name__ == "__main__":
    main()
