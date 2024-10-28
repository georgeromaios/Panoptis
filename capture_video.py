import cv2
import torch
import numpy as np
from mss import mss
from datetime import datetime

# Define screen capture area (adjust as needed to match your screen setup)
# The area should be a dictionary with 'top', 'left', 'width', 'height'
screen_area = {
    "top": 100,       # Y-axis position of top-left corner
    "left": 100,      # X-axis position of top-left corner
    "width": 1280,    # Width of capture area
    "height": 720     # Height of capture area
}

# Initialize screen capture
sct = mss()

# Load YOLO model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Replace 'yolov5s' with a specific model if needed

# Define video output settings
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'captured_video_{timestamp}.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (screen_area['width'], screen_area['height']))

# Capture loop
print("Press 'q' to exit the screen capture.")
while True:
    # Capture screen region
    screen_frame = np.array(sct.grab(screen_area))

    # Convert the image to RGB (mss returns BGRA)
    frame_rgb = cv2.cvtColor(screen_frame, cv2.COLOR_BGRA2RGB)

    # Run YOLO model on the frame
    results = model(frame_rgb)

    # Filter results for 'person' and 'dog' classes
    detected_objects = results.pandas().xyxy[0]  # Get the predictions
    filtered_objects = detected_objects[detected_objects['name'].isin(['person', 'dog'])]

    # Draw bounding boxes and labels on the frame
    for _, row in filtered_objects.iterrows():
        label = f"{row['name']} {row['confidence']:.2f}"
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save frame to video output file
    out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Display the frame in a window
    cv2.imshow("Screen Capture with Detection", frame_rgb)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cv2.destroyAllWindows()
sct.close()
