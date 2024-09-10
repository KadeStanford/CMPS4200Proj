from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import shutil
import atexit
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection

app = Flask(__name__)

#Defines the upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'svg'}

#Creates the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to get the most recent file in the uploads folder
def get_most_recent_file():
    files_in_folder = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    most_recent_file = max(files_in_folder, key=os.path.getmtime)
    return most_recent_file

#Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

#Route for uploading the image
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file part'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index', message='No selected file'))
    
    if file and allowed_file(file.filename):
        # Save the file directly in the upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        #Redirect to the detect route
        return redirect(url_for('detect_objects'))

    return redirect(url_for('index', message='Invalid file type'))

# Route to serve files from the uploads folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect', methods=['GET'])
def detect_objects():
    # Get the most recent file
    recent_file = get_most_recent_file()

    # Load the image using PIL
    image = Image.open(recent_file).convert('RGB')
    image = image.resize((640, 640)) 
    print(f"Image mode: {image.mode}")
    print(f"Image size: {image.size}")

    # Initialize YOLOs model and feature extractor
    feature_extractor = YolosImageProcessor.from_pretrained('hustvl/yolos-small')
    
    # Configure image processor options
    processor_options = {
        "do_resize": True,
        "size": {"shortest_edge": 800, "longest_edge": 1333},
        "resample": Image.BILINEAR,
        "do_rescale": True,
        "rescale_factor": 1/255,
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],  # Example mean values (ImageNet defaults)
        "image_std": [0.229, 0.224, 0.225],   # Example std values (ImageNet defaults)
        "do_pad": True,
        "pad_size": {"height": 800, "width": 800}  # Adjust as needed
    }

    # Prepare the image for the model
    try:
        inputs = YolosImageProcessor(images=image, return_tensors="pt", **processor_options)
    except Exception as e:
        return f"Error processing image: {e}"
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform object detection
    outputs = model(**inputs)

    # Get the logits and bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Convert the recent file path to a filename
    filename = os.path.basename(recent_file)

    # Process the results (e.g., number of detected objects)
    num_detected_objects = logits.shape[1]

    # Render the template with the image and detection information
    return render_template('index.html', image_path=filename, num_detected_objects=num_detected_objects)

#Function to clean up the uploads folder
def cleanup_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)  #Recreate the folder for future use

#Registers the cleanup function to be called on exit
atexit.register(cleanup_upload_folder)

if __name__ == '__main__':
    app.run(debug=True)
