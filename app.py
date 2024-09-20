from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import shutil
import atexit
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import card_detector

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'svg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def get_most_recent_file():
    files_in_folder = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    most_recent_file = max(files_in_folder, key=os.path.getmtime)
    return most_recent_file


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file part'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index', message='No selected file'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        return redirect(url_for('detect_text'))

    return redirect(url_for('index', message='Invalid file type'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



@app.route('/detect', methods=['GET'])
def detect_text():
    recent_file = get_most_recent_file()
    result, error = card_detector.isolate_and_straighten_card(recent_file)

    if error:
        return render_template('index.html', message=error)

    filename = os.path.basename(recent_file)
    card_name = result['extracted_text']

    # Scryfall API request
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


# Manual card search
@app.route('/search', methods=['POST'])
def search_card():
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


# Autocomplete card search
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term')
    scryfall_url = f"https://api.scryfall.com/cards/autocomplete?q={term}"

    response = requests.get(scryfall_url)

    if response.status_code == 200:
        data = response.json()
        suggestions = data.get('data', [])
        return jsonify(suggestions)
    else:
        return jsonify([])



def cleanup_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)


atexit.register(cleanup_upload_folder)

if __name__ == '__main__':
    app.run(debug=True)