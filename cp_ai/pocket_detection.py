import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pathlib

model = YOLO(r"C:\Users\ScorpionIPX\Downloads\best (1)\pocket_detection.pt")

test_images = []
rig_dataset_path = pathlib.Path(r'D:\projects\clothes_processing\test_data\from_rig\f1733932475')
# print(rig_dataset_path)
rig_dataset = [
    pathlib.Path(img) for img in
    list(rig_dataset_path.glob('*.jpg'))
]
test_images.extend(rig_dataset)


for image_path in test_images:
    # image_path = r"C:\Users\ScorpionIPX\Downloads\frame1734350640.jpg"

    results = model(image_path)  # results list


    for r in results:
        print(r.obb)  # print the OBB object containing the oriented detection bounding boxes


    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Coordinates of the intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Area of intersection
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Union area
        union_area = box1_area + box2_area - inter_area

        # IoU calculation
        iou = inter_area / union_area if union_area > 0 else 0
        return iou


    # Load the image with OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Store bounding boxes and differences
    side_differences = []
    bounding_boxes = []  # To store bounding box details

    # Overlay segmentation masks and analyze bounding boxes
    for r in results:
        for mask in r.masks.xy:  # Access segmentation masks
            # Convert mask coordinates to a filled polygon
            pts = np.array(mask, dtype=np.int32)

            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(pts)
            bbox_area = w * h

            # Check for duplicates: Remove duplicates using IoU threshold
            to_remove = []
            for i, existing_box in enumerate(bounding_boxes):
                iou = calculate_iou((x, y, w, h), existing_box)
                if iou > 0.5:  # If IoU is greater than 0.5, we consider it a duplicate
                    to_remove.append(i)

            # Remove duplicates if any
            for i in sorted(to_remove, reverse=True):
                del bounding_boxes[i]

            # Add the current bounding box
            bounding_boxes.append((x, y, w, h))

            # Calculate left and right half areas
            left_area = (w // 2) * h
            right_area = (w - w // 2) * h
            difference = abs(left_area - right_area) / bbox_area * 100 if bbox_area > 0 else 0
            side_differences.append(difference)

            # Visualize the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, f"Diff: {difference:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # Print bounding box side differences
    print("Bounding box side differences (%):", side_differences)
