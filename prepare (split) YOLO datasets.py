# This script splits a dataset of images and labels into train, validation, and test sets.
# The dataset must follow the structure:
#   dataset_path/
#       images/
#           image1.jpg, image2.jpg, ...
#       labels/
#           image1.txt, image2.txt, ...
# The script:
# 1. Normalizes train, validation, and test ratios if they do not sum to 1.
# 2. Randomly shuffles the dataset and splits it into specified proportions.
# 3. Copies images and corresponding labels to respective folders for train, val, and test sets.
# Input: A dataset folder containing 'images' and 'labels' subfolders.
# Output: Train, validation, and test datasets stored in organized subdirectories under the output folder.
 
import os
import shutil
import random

def create_split_folders(dataset_path, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split a dataset of images and labels into train, validation, and test sets.

    Args:
        dataset_path (str): Path to the dataset containing 'images' and 'labels' folders.
        output_path (str): Path to the output folder where split datasets will be stored.
        train_ratio (float): Proportion of training data.
        val_ratio (float): Proportion of validation data.
        test_ratio (float): Proportion of testing data.

    Folders structure should be:
        dataset_path/images
        dataset_path/labels
    """
    # Normalize the ratios if they don't sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    # Input folders
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    # Output folders
    train_images = os.path.join(output_path, 'train/images')
    val_images = os.path.join(output_path, 'val/images')
    test_images = os.path.join(output_path, 'test/images')

    train_labels = os.path.join(output_path, 'train/labels')
    val_labels = os.path.join(output_path, 'val/labels')
    test_labels = os.path.join(output_path, 'test/labels')

    # Create folders if they do not exist
    for folder in [train_images, val_images, test_images, train_labels, val_labels, test_labels]:
        os.makedirs(folder, exist_ok=True)

    # List all image files
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]

    # Shuffle the files
    random.shuffle(image_files)

    # Split the dataset
    num_images = len(image_files)
    train_split = int(num_images * train_ratio)
    val_split = int(num_images * (train_ratio + val_ratio))

    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]

    def copy_files(file_list, src_images, src_labels, dest_images, dest_labels):
        for file_name in file_list:
            base_name = os.path.splitext(file_name)[0]

            # Copy image file
            src_image = os.path.join(src_images, file_name)
            dest_image = os.path.join(dest_images, file_name)
            if os.path.exists(src_image):
                shutil.copy(src_image, dest_image)

            # Copy corresponding label file
            src_label = os.path.join(src_labels, f"{base_name}.txt")
            dest_label = os.path.join(dest_labels, f"{base_name}.txt")
            if os.path.exists(src_label):
                shutil.copy(src_label, dest_label)

    # Copy files to respective folders
    copy_files(train_files, images_path, labels_path, train_images, train_labels)
    copy_files(val_files, images_path, labels_path, val_images, val_labels)
    copy_files(test_files, images_path, labels_path, test_images, test_labels)

    print("Dataset successfully split!")

dataset_path = "D:\\new training\\new dataset"  
output_path = "D:\\new training\\new dataset"  
create_split_folders(dataset_path, output_path)
