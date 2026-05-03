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
    
    if frame_count > 0:
        fps_processed = frame_count / (end_time - start_time)
        print(f"Processed {frame_count} frames. Average FPS: {fps_processed:.2f}")
        return fps_processed
    return 0

def process_webcam_stream(model, camera_id=0):
    """
    Opens the specified webcam, runs inference frame by frame,
    and displays the result in a real-time window.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Error opening webcam with ID: {camera_id}")
        
    print(f"Started webcam stream. Press 'q' to quit (ensure the video window is focused).")
    
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
            
        annotated_frame = model.predict_and_annotate(frame)
        
        cv2.imshow(f'Real-time Detection ({model.__class__.__name__})', annotated_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        
    end_time = time.time()
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        fps_processed = frame_count / (end_time - start_time)
        print(f"Processed {frame_count} frames. Average FPS: {fps_processed:.2f}")
        return fps_processed
    return 0
