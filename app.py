from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import shutil
import atexit
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

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
        return redirect(url_for('detect_text'))

    return redirect(url_for('index', message='Invalid file type'))

# Route to serve files from the uploads folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for performing OCR on the most recent image
@app.route('/detect', methods=['GET'])
def detect_text():
    # Get the most recent file
    recent_file = get_most_recent_file()

    # Load the image using PIL
    image = Image.open(recent_file).convert('RGB')

    # Initialize the OCR model and processor
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    # Process the image and generate text
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Convert the recent file path to a filename
    filename = os.path.basename(recent_file)

    # Show image and text
    return render_template('index.html', image_path=filename, extracted_text=generated_text)

#Function to clean up the uploads folder
def cleanup_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)  #Recreate the folder for future use

#Register the cleanup function to be called on exit
atexit.register(cleanup_upload_folder)

if __name__ == '__main__':
    app.run(debug=True)
