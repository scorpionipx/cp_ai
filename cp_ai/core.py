from ultralytics import YOLO

# Load a pre-trained YOLO model (YOLOv8n is lightweight; you can use 'yolov8s.pt_old' for larger models)
model = YOLO('yolov8s.pt_old')

# Train the model
model.train(data='data.yaml', epochs=500, imgsz=640)

# Validate the model
results = model.val()

# Save the best model
model.export(format='onnx')  # Export for deployment if needed
