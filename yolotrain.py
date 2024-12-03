import os
from ultralytics import YOLO
import torch

def find_latest_model(runs_dir):
    """Find the latest best.pt model in the runs directory."""
    best_model_path = None
    latest_time = 0

    # Traverse the directory tree to find all best.pt files
    for root, _, files in os.walk(runs_dir):
        for file in files:
            if file == "best.pt":
                file_path = os.path.join(root, file)
                # Get the last modified time
                file_time = os.path.getmtime(file_path)
                # Update if this file is newer
                if file_time > latest_time:
                    latest_time = file_time
                    best_model_path = file_path

    return best_model_path

# Main training function
def main():
    # Check if CUDA is available and print GPU details
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 0  # Use GPU 0
    else:
        print("CUDA is not available. Using CPU.")
        device = "cpu"  # Fallback to CPU

    # Base directory where runs are stored
    runs_dir = "./runs"  # Relative path to the runs directory

    # Check if the runs directory exists, if not, create it
    if not os.path.exists(runs_dir):
        print(f"Runs directory '{runs_dir}' does not exist. Creating a new one.")
        os.makedirs(runs_dir)

    # Find the latest best.pt model
    best_model_path = find_latest_model(runs_dir)
    if best_model_path:
        print(f"Found latest model: {best_model_path}")
        # Load the previously trained YOLO model with best weights
        model = YOLO(best_model_path)
    else:
        print("No best.pt model found. Initializing a new YOLO model.")
        # Initialize a new YOLO model
        model = YOLO('yolo11n.pt')  # Use the base model

    # Start new training
    train_results = model.train(
        data="./mtg_dataset/data.yaml",  # Relative path to your dataset YAML
        epochs=300,  # Number of new training epochs
        imgsz=640,  # Training image size
        device=device,  # Use the detected device
        workers=6,  # Set the number of workers to 4, or adjust as needed
        project=runs_dir,  # Save new runs to the runs directory
        name='train',  # Name of the new run
        amp=False,  # Disable Automatic Mixed Precision
        exist_ok=True  # Continue training in the same directory if it exists
    )

    # Evaluate model performance on the validation set using the selected device
    metrics = model.val(device=device)

    # Perform object detection on a sample image using the selected device
    sample_image_path = "output.jpg"  # Relative path to the sample image
    results = model(sample_image_path, device=device)
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx", device=device)
    print(f"Model exported to: {path}")

if __name__ == '__main__':
    main()
