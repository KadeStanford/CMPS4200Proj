import os
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
from ultralytics import YOLO

# Function to find the latest model
def find_latest_model(runs_dir):
    """Find the latest best.onnx model in the runs directory."""
    best_model_path = None
    latest_time = 0

    # Traverse the directory tree to find all best.onnx files
    for root, _, files in os.walk(runs_dir):
        for file in files:
            if file == "best.onnx":
                file_path = os.path.join(root, file)
                # Get the last modified time
                file_time = os.path.getmtime(file_path)
                # Update if this file is newer
                if file_time > latest_time:
                    latest_time = file_time
                    best_model_path = file_path

    return best_model_path

# Base directory where the project is located
project_dir = os.path.dirname(os.path.abspath(__file__))

# Find the latest best.onnx model
runs_dir = os.path.join(project_dir, "runs")  # Base directory where runs are stored
yolo_model_path = find_latest_model(runs_dir)

if yolo_model_path:
    print(f"Using latest model: {yolo_model_path}")
else:
    print("No best.onnx model found. Exiting.")
    exit()

# Load the YOLO model with the latest best.onnx
yolo_model = YOLO(yolo_model_path)  # Load the trained YOLO model

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

def isolate_and_extract_card_name(image_path):
    """
    Detects multiple card names using YOLO and extracts text using TrOCR.
    Returns a list of results for each detected card.
    """
    try:
        # Perform object detection with YOLO to find the card name regions
        results = yolo_model(image_path, imgsz=640, device="cpu")[0]  # Assuming single detection per image

        # Read the original image for visualization
        original_image = cv2.imread(image_path)
        if original_image is None:
            return None, "Error: Could not read image."

        extracted_cards = []  # List to store information of detected cards

        # Iterate over each detected box
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 1:  # Assuming 'card_name' is class 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                card_name_roi = extract_roi(image_path, x1, y1, x2, y2)
                
                # Perform OCR on the detected ROI
                extracted_text = perform_ocr(card_name_roi)
                cleaned_text = clean_extracted_text(extracted_text)
                
                # Store each card's information
                extracted_cards.append({
                    'original_image': original_image,
                    'card_name_roi': card_name_roi,
                    'extracted_text': extracted_text,
                    'cleaned_text': cleaned_text,
                    'bbox': (x1, y1, x2, y2)  # Bounding box coordinates for visualization
                })

        if len(extracted_cards) == 0:
            return None, "Error: Could not find any card name bounding box."

        return extracted_cards[0], None  # Return the first card found

    except Exception as e:
        return None, f"Error during image processing: {str(e)}"

def extract_roi(image_path, x1, y1, x2, y2):
    """
    Extracts the region of interest (ROI) from the image based on the bounding box coordinates.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        roi = image[y1:y2, x1:x2]
        return roi
    except Exception as e:
        print(f"Error extracting ROI: {str(e)}")
        return None

def perform_ocr(roi_image):
    """
    Perform Optical Character Recognition (OCR) on the region of interest (ROI) image.
    """
    try:
        # Convert the ROI image to RGB if needed
        if len(roi_image.shape) == 2 or roi_image.shape[2] == 1:
            roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)
        else:
            roi_image_rgb = roi_image

        pil_image = Image.fromarray(roi_image_rgb)

        # Prepare the image for the TrOCR model
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text
    except Exception as e:
        print(f"Error during OCR: {str(e)}")
        return ""

def clean_extracted_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text
