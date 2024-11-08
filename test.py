import cv2
from ultralytics import YOLO
import time

# Load YOLO models
model_v8 = YOLO("yolov8m.pt")  # Use smaller model for speed
model_v11 = YOLO("yolo11m.pt")  # Assuming YOLO v11 is available

# Path to the input video
video_path = "C:/Users/DELL/Downloads/Crowded shopping malls during CA Stay at Home Covid Surge.mp4"
cap = cv2.VideoCapture(video_path)

# Font settings for labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)
thickness = 2
position_v8 = (10, 30)  # Position of the YOLO v8 label
position_v11 = (10, 30)  # Position of the YOLO v11 label

# Get the dimensions of the video frame to ensure consistent resizing
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Process every nth frame (for faster processing)
frame_skip = 4  # Process every 2nd frame

frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1

    # Skip frames to speed up the process
    if frame_counter % frame_skip != 0:
        continue

    # Start measuring inference time
    start_time = time.time()

    # Run YOLO v8 tracking on the frame
    results_v8 = model_v8.track(source=frame, tracker="bytetrack.yaml")
    frame_v8 = results_v8[0].plot()  # YOLO v8 processed frame
    
    # Run YOLO v11 tracking on the frame
    results_v11 = model_v11.track(source=frame, tracker="bytetrack.yaml")
    frame_v11 = results_v11[0].plot()  # YOLO v11 processed frame

    # Resize both frames to a fixed resolution for better comparison (e.g., 640x480)
    target_size = (640, 480)  # You can adjust this size if needed
    frame_v8_resized = cv2.resize(frame_v8, target_size)
    frame_v11_resized = cv2.resize(frame_v11, target_size)

    # Add label text to each frame
    cv2.putText(frame_v8_resized, "YOLO v8", position_v8, font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(frame_v11_resized, "YOLO 11", position_v11, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Concatenate frames horizontally for side-by-side display
    combined_frame = cv2.hconcat([frame_v8_resized, frame_v11_resized])

    # Display the combined frame
    cv2.imshow("YOLO v8 vs YOLO 11 Tracking Comparison", combined_frame)

    # Calculate and print inference time for profiling
    inference_time = time.time() - start_time
    print(f"Frame Inference Time: {inference_time:.3f} seconds")

    # To speed up the video, skip frame delay adjustment
    # Display frames at the original FPS
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # Adjust the wait time to match video FPS
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
