import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
from fuzzywuzzy import fuzz, process
import re

def isolate_and_straighten_card(image_path):
    # Ensure the image path is the same and accessible
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not read image."

    original_image = image.copy()
    height, width = image.shape[:2]
    max_height = 1000
    if height > max_height:
        scale = max_height / height
        image = cv2.resize(image, None, fx=scale, fy=scale)
        original_image = cv2.resize(original_image, None, fx=scale, fy=scale)
        height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    card_contour = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            card_contour = approx
            break

    if card_contour is None:
        return None, "Error: Could not find card contour."

    rect = order_points(card_contour.reshape(4, 2))
    warped = four_point_transform(image, rect)

    output_path = 'output.jpg'
    cv2.imwrite(output_path, warped)

    card_name_roi = extract_card_name_roi_fixed(warped)
    extracted_text = perform_ocr(card_name_roi)
    cleaned_text = clean_extracted_text(extracted_text)
    return {
        'original_image': original_image,
        'warped_image': warped,
        'card_name_roi': card_name_roi,
        'extracted_text': extracted_text,
        'cleaned_text': cleaned_text
    }, None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect
    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth -1, 0], [maxWidth -1, maxHeight -1], [0, maxHeight -1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def extract_card_name_roi_fixed(card_image):
    height, width = card_image.shape[:2]
    roi_x = int(width * 0.05)
    roi_y = int(height * 0.03)
    roi_w = int(width * 0.80)
    roi_h = int(height * 0.10)
    roi = card_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    return roi

def perform_ocr(roi_image):
    # Pre-process the ROI image for better OCR results
    roi_image_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Increase contrast
    roi_image_gray = cv2.equalizeHist(roi_image_gray)

    # Apply a slight blur to reduce noise
    roi_image_blur = cv2.GaussianBlur(roi_image_gray, (3, 3), 0)

    # Thresholding to convert to binary image
    _, roi_image_thresh = cv2.threshold(roi_image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert single-channel image back to 3-channel by repeating the grayscale values across three channels
    roi_image_rgb = cv2.cvtColor(roi_image_thresh, cv2.COLOR_GRAY2RGB)

    # Check the dimensions of the image to debug
    print(f"Image dimensions before processing: {roi_image_rgb.shape}")

    # Convert OpenCV image (numpy array) to PIL Image
    pil_image = Image.fromarray(roi_image_rgb)

    # Load TrOCR processor and model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

    # Prepare image for the model
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

    # Generate text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def clean_extracted_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def load_card_names(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        card_names = json.load(f)
    return card_names

def fuzzy_match_card_name(extracted_text, card_names):
    matches = process.extract(extracted_text, card_names, scorer=fuzz.token_sort_ratio, limit=5)
    threshold = 60
    best_match = None
    best_score = 0
    for match in matches:
        name, score = match
        if score > best_score:
            best_match = name
            best_score = score
    return best_match
