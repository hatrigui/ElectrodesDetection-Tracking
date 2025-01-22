# This code works perfectly in the non-under plasma situation.
# It processes images in a specified folder to:
# 1. Detect objects (3 electrodes) using contour detection after Gaussian blur and thresholding.
# 2. Filter objects based on a minimum area threshold.
# 3. Calculate bounding box coordinates and normalize them to YOLO format.
# 4. Save the processed images with bounding boxes and centroids drawn.
# 5. Generate YOLO-compatible label files for the detected objects.
# Input: Folder with images (.jpg or .png).
# Output: Processed images and label files saved in respective subdirectories. 


import cv2
import numpy as np
import os
from itertools import combinations

def calculate_positions_and_distances(input_folder, output_folder, min_area=500):
    # Ensure output folders exist
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    
    def normalize_coordinates(x, y, w, h, img_width, img_height):
        """Normalize bounding box coordinates for YOLO format."""
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return x_center, y_center, width, height
    
    # Iterate over each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(images_folder, filename)
            label_path = os.path.join(labels_folder, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

            # Load the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape

            # Apply Gaussian blur and thresholding
            blurred = cv2.GaussianBlur(img, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Open label file for writing
            with open(label_path, "w") as label_file:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area:
                        # Calculate bounding box and centroid
                        x, y, w, h = cv2.boundingRect(contour)
                        cX = int(x + w / 2)
                        cY = int(y + h / 2)

                        # Normalize bounding box coordinates
                        x_center, y_center, width, height = normalize_coordinates(x, y, w, h, img_width, img_height)

                        # Write to YOLO label file
                        label_file.write(f"0 {x_center} {y_center} {width} {height}\n")

                        # Draw bounding box and centroid on the image
                        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(output_img, (cX, cY), 5, (0, 0, 255), -1)
                        cv2.putText(output_img, f"({cX}, {cY})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Save the processed image
            cv2.imwrite(output_image_path, output_img)
            print(f"Processed frame saved: {output_image_path}, Labels saved: {label_path}")

input_folder = "D:\\Frames\\vid4"
output_folder = "D:\\Frames\\vid4\\yolo_dataset"
calculate_positions_and_distances(input_folder, output_folder)
