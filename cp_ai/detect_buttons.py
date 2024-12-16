from ultralytics import YOLO
from IPython.display import Image, display
import pathlib
import cv2



def scale_image(image, scale_percent):
    """

    """

    # Get the dimensions of the original image
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)
    new_dimensions = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image


def display_scaled(image, scale_percent, view='Image', blocking=False):
    """

    """
    cv2.imshow(view, scale_image(image, scale_percent))
    if blocking:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print('main')
    # model = YOLO(rf"C:\Users\ScorpionIPX\PycharmProjects\cp_ai\cp_ai\runs\detect\train16\weights\best.pt")
    model = YOLO(rf"C:\Users\ScorpionIPX\PycharmProjects\cp_ai\cp_ai\runs\detect\train16\weights\best.pt")
    # model = YOLO(rf"C:\Users\ScorpionIPX\Downloads\best (1)\pocket_detection.pt")
    # model.conf_thresh = .6

    test_images = [
        pathlib.Path(r'C:\Users\ScorpionIPX\Downloads\frame1734350640.jpg'),
    ]
    rig_dataset_path = pathlib.Path(r'D:\projects\clothes_processing\test_data\from_rig\f1733932475')
    # print(rig_dataset_path)
    rig_dataset = [
        pathlib.Path(img) for img in
        list(rig_dataset_path.glob('*.jpg'))
    ]
    test_images.extend(rig_dataset)

    shit_rig_dataset_path = pathlib.Path(r'D:\projects\clothes_processing\test_data\referinta-dut2-v01')
    # print(rig_dataset_path)
    shit_rig_dataset = [
        pathlib.Path(img) for img in
        list(shit_rig_dataset_path.glob('*.jpeg'))
    ]
    test_images.extend(shit_rig_dataset)


    kaggle_dataset = [
        pathlib.Path(img) for img in
        list(pathlib.Path(r'D:\projects\clothes_processing\test_data\t-shirt-dataset-kaggle\tshirt').glob('*.jpg'))
    ]
    # test_images.extend(kaggle_dataset)
    for img in test_images:
        print(img)
        results = model(img, conf=.4)

        res_plotted = results[0].plot()
        for result in results:
            boxes = result.boxes
            masks = result.masks
            probs = result.probs
            print(boxes, masks, probs)

        img_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        display_scaled(img_rgb, 50, 'cacat', blocking=True)
    cv2.destroyAllWindows()
