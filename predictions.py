import cv2
import os
import numpy as np
from ultralytics import YOLO

def draw_contours_on_predictions(model_path, test_images_path, output_folder, min_area=500):
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Running predictions on images in: {test_images_path}")
    results = model.predict(source=test_images_path, save=True, save_txt=True, stream=True)

    print(f"Ensuring output folder exists: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    for result in results:
        print(f"Result content: {result}")  # Inspect the result object
        img_path = result.path
        print(f"Image path: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        output_img = img.copy()
        if result.boxes and len(result.boxes.xyxy) > 0:
            for box in result.boxes.xyxy:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                roi = img[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(roi_gray, (7, 7), 0)
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > min_area:
                            contour += [x_min, y_min]
                            color = tuple(np.random.randint(0, 256, 3).tolist())
                            cv2.drawContours(output_img, [contour], -1, color, 2)

        output_image_path = os.path.join(output_folder, os.path.basename(img_path))
        success = cv2.imwrite(output_image_path, output_img)
        if success:
            print(f"Successfully saved: {output_image_path}")
        else:
            print(f"Failed to save: {output_image_path}")

# Example usage
model_path = "C:\\Users\\la7tim\\Desktop\\best.pt"
test_images_path = "C:\\Users\\la7tim\\Downloads\\1frame_02416.jpg"
output_folder = "C:\\Users\\la7tim\\Desktop"
draw_contours_on_predictions(model_path, test_images_path, output_folder)

