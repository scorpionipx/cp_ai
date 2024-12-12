from ultralytics import YOLO

# Load the trained model
model = YOLO('yolov8s.pt_old')  # Replace with the path to your trained model

# Run validation
results = model.val()

# Print results
print(results)
