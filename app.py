# using opencv

# import cv2

# from ultralytics import YOLO

# # Load the YOLO11 model
# model = YOLO("yolo11n.pt")

# # Open the video file
# video_path = "C:/Users/DELL/Downloads/1725911732503.mp4"
# cap = cv2.VideoCapture(video_path)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLO11 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLO11 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()





# from ultralytics import YOLO

# # Configure the tracking parameters and run the tracker
# model = YOLO("yolo11n.pt")
# # model = YOLO("yolo11n-seg.pt") 
# # model = YOLO("yolo11n-pose.pt")
# # results = model.track(source="C:/Users/DELL/Downloads/1725911732503.mp4", conf=0.3, iou=0.5, show=True)
# # results = model.track("C:/Users/DELL/Downloads/1725911732503.mp4", show=True, tracker="bytetrack.yaml")  # -> more better then normal

# results = model.track("C:/Users/DELL/Downloads/1725911732503.mp4", show=True, tracker="botsort.yaml")




from ultralytics import YOLO


model = YOLO("C:/Tracking/best.pt")  # Load a custom trained model

# Perform tracking with the model
# results = model.track("https://www.youtube.com/watch?v=gbxJT-yBdcs", show=True)  # Tracking with default tracker
results = model.track("https://www.youtube.com/watch?v=gbxJT-yBdcs", show=True, tracker="bytetrack.yaml")  # with ByteTrack