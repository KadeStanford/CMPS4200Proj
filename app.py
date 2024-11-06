from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import shutil
import atexit
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import card_detector
import gc

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
TEMP_FOLDER = './temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'svg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_most_recent_file():
    try:
        files_in_folder = [os.path.join(TEMP_FOLDER, f) for f in os.listdir(TEMP_FOLDER) if os.path.isfile(os.path.join(TEMP_FOLDER, f))]
        if not files_in_folder:
            return None
        most_recent_file = max(files_in_folder, key=os.path.getmtime)
        return most_recent_file
    except Exception as e:
        print(f"Error finding most recent file: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return redirect(url_for('index', message='No file part'))

        file = request.files['file']

        if file.filename == '':
            return redirect(url_for('index', message='No selected file'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_file_path = os.path.join(app.config['TEMP_FOLDER'], filename)
            file.save(temp_file_path)
            
            del file
            gc.collect()

            return redirect(url_for('detect_text', temp_image=filename))

        return redirect(url_for('index', message='Invalid file type'))
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return redirect(url_for('index', message='Error during file upload'))

@app.route('/temp_uploads/<filename>')
def temp_uploaded_file(filename):
    try:
        return send_from_directory(app.config['TEMP_FOLDER'], filename)
    except Exception as e:
        print(f"Error sending temp file {filename}: {str(e)}")
        return "File not found", 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        # Check if the file exists in the permanent uploads folder
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        # Fall back to temp_uploads if not in uploads
        return send_from_directory(app.config['TEMP_FOLDER'], filename)
    except Exception as e:
        print(f"Error sending file {filename}: {str(e)}")
        return "File not found", 404


@app.route('/detect', methods=['GET'])
def detect_text():
    try:
        recent_file = get_most_recent_file()
        if not recent_file:
            return render_template('index.html', message="No uploaded image found.")
        
        result, error = card_detector.isolate_and_extract_card_name(recent_file)

        if error:
            return render_template('index.html', message=error)

        filename = os.path.basename(recent_file)
        card_name = result['cleaned_text']

        scryfall_url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
        response = requests.get(scryfall_url)

        if response.status_code == 200:
            card_data = response.json()
            card_info = {
                'name': card_data.get('name'),
                'type_line': card_data.get('type_line'),
                'mana_cost': card_data.get('mana_cost', 'N/A'),
                'oracle_text': card_data.get('oracle_text', 'N/A'),
                'image_url': card_data.get('image_uris', {}).get('normal'),
                'usd_price': card_data.get('prices', {}).get('usd', 'N/A')
            }
            return render_template('index.html', image_path=filename, card_info=card_info)
        else:
            return render_template('index.html', image_path=filename, extracted_text=card_name, message="Card not found!")
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return render_template('index.html', message="Error during detection process.")
    finally:
        gc.collect()

@app.route('/store_card', methods=['POST'])
def store_card():
    try:
        temp_image = request.form.get('image_path')
        temp_image_path = os.path.join(app.config['TEMP_FOLDER'], temp_image)
        
        if os.path.exists(temp_image_path):
            permanent_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_image)
            shutil.move(temp_image_path, permanent_path)
            return jsonify({"message": "Card stored successfully!", "success": True})
        else:
            return jsonify({"message": "Temporary image not found. Please reupload the card.", "success": False})
    except Exception as e:
        print(f"Error storing card: {str(e)}")
        return jsonify({"message": "Error storing card.", "success": False})

@app.route('/remove_card', methods=['POST'])
def remove_card():
    try:
        image_path = request.form.get('image_path')
        permanent_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
        
        if os.path.exists(permanent_image_path):
            os.remove(permanent_image_path)
            return jsonify({"message": "Card removed successfully!", "success": True})
        else:
            return jsonify({"message": "Card not found in the uploads folder.", "success": False})
    except Exception as e:
        print(f"Error removing card: {str(e)}")
        return jsonify({"message": "Error removing card.", "success": False})

@app.route('/upload_history', methods=['GET'])
def upload_history():
    try:
        uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f)) and allowed_file(f)]
        return jsonify(uploaded_files)
    except Exception as e:
        print(f"Error fetching upload history: {str(e)}")
        return jsonify({"message": "Error fetching upload history", "success": False}), 500
    

@app.route('/clear_upload_history', methods=['POST'])
def clear_upload_history():
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return jsonify({"message": "Upload history cleared successfully!", "success": True})
    except Exception as e:
        print(f"Error clearing upload history: {str(e)}")
        return jsonify({"message": "Error clearing upload history.", "success": False})


@app.route('/search', methods=['POST'])
def search_card():
    try:
        card_name = request.form.get('card_name')

        scryfall_url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
        response = requests.get(scryfall_url)

        if response.status_code == 200:
            card_data = response.json()
            card_info_manual = {
                'name': card_data.get('name'),
                'type_line': card_data.get('type_line'),
                'mana_cost': card_data.get('mana_cost', 'N/A'),
                'oracle_text': card_data.get('oracle_text', 'N/A'),
                'image_url': card_data.get('image_uris', {}).get('normal'),
                'usd_price': card_data.get('prices', {}).get('usd', 'N/A')
            }
            return render_template('index.html', card_info_manual=card_info_manual)
        else:
            return render_template('index.html', message="Card not found!")
    except Exception as e:
        print(f"Error during manual search: {str(e)}")
        return render_template('index.html', message="Error during manual search.")
    finally:
        gc.collect()

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    try:
        term = request.args.get('term')
        scryfall_url = f"https://api.scryfall.com/cards/autocomplete?q={term}"

        response = requests.get(scryfall_url)

        if response.status_code == 200:
            data = response.json()
            suggestions = data.get('data', [])
            return jsonify(suggestions)
        else:
            return jsonify([])
    except Exception as e:
        print(f"Error during autocomplete: {str(e)}")
        return jsonify([])
    finally:
        gc.collect()

def cleanup_temp_folder():
    try:
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER)
            os.makedirs(TEMP_FOLDER)
    except Exception as e:
        print(f"Error cleaning up temp folder: {str(e)}")

atexit.register(cleanup_temp_folder)

if __name__ == '__main__':
    app.run(debug=False, port=5001)
