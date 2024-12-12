import torch

from pathlib import Path
from ultralytics import YOLO


MODEL_YOLOV8X = 'yolov8x'  # requires RTX3090 or better
TRAINING_RESOLUTION = 1024
TRAINING_EPOCHS = 500


def train_model():
    """
    train_model
    :return:
    """

    dataset_directory = Path(r'C:\Users\ScorpionIPX\PycharmProjects\cp_ai\datasets\yolo\buttons_detection\v1')
    data_yaml = dataset_directory.joinpath('data.yaml')

    # Load a pre-trained YOLO model (YOLOv8n is lightweight; you can use 'yolov8s.pt_old' for larger models)
    model = YOLO(MODEL_YOLOV8X)

    # Train the model
    model.train(data=data_yaml, epochs=TRAINING_EPOCHS, imgsz=TRAINING_RESOLUTION, device=0)
    print(model.device)

    # Validate the model
    results = model.val()

    # Save the best model
    model.export()  # Export for deployment if needed


if __name__ == '__main__':
    print(torch.version.cuda)  # Shows the CUDA version PyTorch is using
    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.get_device_name(0))  # Prints the name of your GPU
    train_model()