import cv2
import time

def process_video_stream(model, video_path, out_path, num_frames=30):
    """
    Reads a video, runs inference frame by frame using the provided model,
    and writes the annotated frames to out_path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    start_time = time.time()
    frame_count = 0
    
    print(f"Starting inference on {video_path}...")
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = model.predict_and_annotate(frame)
        out.write(annotated_frame)
        frame_count += 1
        
    end_time = time.time()
    
    cap.release()
    out.release()
    
    fps_processed = frame_count / (end_time - start_time)
    print(f"Processed {frame_count} frames. Average FPS: {fps_processed:.2f}")
    return fps_processed
