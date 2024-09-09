from flask import Flask, render_template, request, redirect, url_for
import os
import shutil
import atexit
from werkzeug.utils import secure_filename
from datetime import datetime

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
        #Creates a unique folder for each upload
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
        os.makedirs(folder_path, exist_ok=True)
        
        #Saves the file to the folder
        filename = secure_filename(file.filename)
        file.save(os.path.join(folder_path, filename))

        #Redirects to the index page with a success message and the uploaded file path
        return redirect(url_for('index', message=f'File successfully uploaded to {folder_path}/{filename}'))
    
    return redirect(url_for('index', message='Invalid file type'))

#Function to clean up the uploads folder
def cleanup_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)  #Recreate the folder for future use

#Registers the cleanup function to be called on exit
atexit.register(cleanup_upload_folder)

if __name__ == '__main__':
    app.run(debug=True)
