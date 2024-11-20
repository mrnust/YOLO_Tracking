from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8m.pt")  # Load the model

# Define vehicle and person classes (COCO class indices)
class_names = {2: "Car", 3: "Motorcycle", 7: "Truck", 0: "Person"}  # Map class IDs to names
vehicle_counts = {name: 0 for name in class_names.values()}  # Initialize counts for each class

# Set to store unique tracking IDs that have already been counted
counted_ids = set()

# Define bounding box colors for each class (using RGB)
class_colors = {
    "Car": (0, 255, 0),        # Green for Cars
    "Motorcycle": (0, 0, 255), # Red for Motorcycles
    "Truck": (255, 0, 0),      # Blue for Trucks
    "Person": (255, 255, 0)    # Yellow for Person
}

# Track the video and display results with vehicle count
results = model.track(
    source="C:/Users/DELL/Downloads/2109463-uhd_3840_2160_30fps.mp4",  # Video file path
    show=False,  # Disable YOLO's built-in visualization
    tracker="bytetrack.yaml",
    conf=0.3,
    iou=0.5,
    stream=True  # Stream results for frame-by-frame processing
)

# OpenCV video writer setup for saving the output video
output_path = "output_with_vehicle_count.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_writer = None

# Resize factor for display
resize_scale = 0.5  # Reduce size by 50%

# Process each frame in the tracking results
for frame_result in results:
    frame = frame_result.orig_img  # Original video frame
    detections = frame_result.boxes  # Detections in the current frame

    # Process detections
    if detections is not None:
        for det in detections:
            tracking_id = int(det.id) if det.id is not None else None  # Get the tracking ID
            class_id = int(det.cls)

            if tracking_id is not None and class_id in class_names:
                vehicle_type = class_names[class_id]

                # Count the vehicle only if it's a new tracking ID
                if tracking_id not in counted_ids:
                    counted_ids.add(tracking_id)
                    vehicle_counts[vehicle_type] += 1

                # Draw the bounding box and label
                box = det.xyxy[0].tolist()  # Bounding box [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box)
                color = class_colors[vehicle_type]  # Get the bounding box color for the current class
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{vehicle_type} ID: {tracking_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Overlay the vehicle counts on the frame
    y_offset = 40
    for vehicle_type, count in vehicle_counts.items():
        count_text = f"{vehicle_type} Count: {count}"
        cv2.putText(
            frame,
            count_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )
        y_offset += 30  # Adjust text spacing

    # Initialize the video writer if not already done
    if output_writer is None:
        height, width, _ = frame.shape
        output_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    # Resize the frame for display
    resized_frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

    # Write the processed frame to the output video
    output_writer.write(frame)

    # Display the resized frame
    cv2.imshow("Vehicle Tracking", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
if output_writer is not None:
    output_writer.release()
cv2.destroyAllWindows()

# Print final counts
print("Final Vehicle Counts:")
for vehicle_type, count in vehicle_counts.items():
    print(f"{vehicle_type}: {count}")
