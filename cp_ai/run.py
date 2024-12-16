from ultralytics import YOLO

# Load the trained model
model = YOLO(r'C:\Users\ScorpionIPX\PycharmProjects\cp_ai\cp_ai\runs\detect\train16\weights\best.pt')

# Test the model on a single image
results = model(r"C:\Users\ScorpionIPX\Downloads\frame1734350640.jpg", save=True)  # Replace with the path to a val image

# Save and visualize the results
# results.save(save_dir='save_results')  # Saves the annotated image
print(f"Results saved in: {results[0].path}")
# Extract the number of detections
detections = results[0].boxes  # List of detected bounding boxes
button_count = len(detections)
print(f"Number of buttons detected: {button_count}")