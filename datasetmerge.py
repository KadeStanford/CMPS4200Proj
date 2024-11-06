import os
import shutil
import random

# Define source and destination paths
new_dataset_images_path = r"E:\CMPS4200Proj\new_dataset\images"
new_dataset_labels_path = r"E:\CMPS4200Proj\new_dataset\labels"
mtg_dataset_images_train = r"E:\CMPS4200Proj\mtg_dataset\images\train"
mtg_dataset_images_val = r"E:\CMPS4200Proj\mtg_dataset\images\val"
mtg_dataset_labels_train = r"E:\CMPS4200Proj\mtg_dataset\labels\train"
mtg_dataset_labels_val = r"E:\CMPS4200Proj\mtg_dataset\labels\val"

# Create destination directories if they don't exist
os.makedirs(mtg_dataset_images_train, exist_ok=True)
os.makedirs(mtg_dataset_images_val, exist_ok=True)
os.makedirs(mtg_dataset_labels_train, exist_ok=True)
os.makedirs(mtg_dataset_labels_val, exist_ok=True)

# Get all images and labels
image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
image_files = sorted([f for f in os.listdir(new_dataset_images_path) if f.lower().endswith(image_extensions)])
label_files = sorted([f for f in os.listdir(new_dataset_labels_path) if f.endswith('.txt')])

# Ensure matching sets
image_files = [f for f in image_files if f.rsplit('.', 1)[0] + '.txt' in label_files]

# Shuffle the dataset for random splitting
random.seed(42)
random.shuffle(image_files)

# Define split ratio
train_ratio = 0.8

# Split data into train and validation sets
train_count = int(train_ratio * len(image_files))
train_images = image_files[:train_count]
val_images = image_files[train_count:]

# Function to copy matching image and label files
def copy_matching_files(images, src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir):
    for image_file in images:
        # Define corresponding label file
        label_file = image_file.rsplit('.', 1)[0] + '.txt'
        
        # Define source and destination paths for image and label
        src_image_path = os.path.join(src_images_dir, image_file)
        src_label_path = os.path.join(src_labels_dir, label_file)
        dest_image_path = os.path.join(dest_images_dir, image_file)
        dest_label_path = os.path.join(dest_labels_dir, label_file)
        
        # Copy image and label files if label exists
        if os.path.exists(src_label_path):
            shutil.copy(src_image_path, dest_image_path)
            shutil.copy(src_label_path, dest_label_path)
            print(f"Copied {src_image_path} and {src_label_path} to {dest_images_dir} and {dest_labels_dir}")
        else:
            print(f"Label file missing for {src_image_path}. Skipping...")

# Copy training data
copy_matching_files(train_images, new_dataset_images_path, new_dataset_labels_path, mtg_dataset_images_train, mtg_dataset_labels_train)

# Copy validation data
copy_matching_files(val_images, new_dataset_images_path, new_dataset_labels_path, mtg_dataset_images_val, mtg_dataset_labels_val)

print("Dataset split and merge complete!")
