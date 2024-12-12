import os
import xml.etree.ElementTree as ET


def convert_xml_to_yolo(image_folder, xml_folder, yolo_folder, classes):
    """
    Converts XML annotations to YOLO format.

    Parameters:
    - image_folder (str): Path to the folder containing images.
    - xml_folder (str): Path to the folder containing XML annotations.
    - yolo_folder (str): Path to the folder to save YOLO annotations.
    - classes (list): List of class names.

    """
    # Ensure YOLO folder exists
    os.makedirs(yolo_folder, exist_ok=True)

    def convert_bbox(size, box):
        """Convert bounding box from XML format to YOLO format."""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    # Loop through all XML files in the folder
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        # Get image dimensions
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        # Create a corresponding YOLO .txt file
        txt_file = os.path.join(yolo_folder, os.path.splitext(xml_file)[0] + ".txt")
        with open(txt_file, "w") as f:
            for obj in root.iter("object"):
                cls = obj.find("name").text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find("bndbox")
                b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
                     float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
                bbox = convert_bbox((w, h), b)
                f.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bbox]) + "\n")

    print(f"Conversion complete! YOLO annotations saved in: {yolo_folder}")


if __name__ == '__main__':
    # Example Usage:
    # Replace these paths and class names with your actual dataset information
    image_folder = r"C:\Users\ScorpionIPX\PycharmProjects\cp_ai\dataset\images\train"
    xml_folder = r"C:\Users\ScorpionIPX\PycharmProjects\cp_ai\dataset\images\train"
    yolo_folder = r"C:\Users\ScorpionIPX\PycharmProjects\cp_ai\dataset\images\train"
    classes = ["button"]  # Replace with your actual class names

    convert_xml_to_yolo(image_folder, xml_folder, yolo_folder, classes)
