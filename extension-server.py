from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import os
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app, origins="chrome-extension://hipkbhlddcpllknigdjmklkkdhcpnljd")

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    title = data['title']
    labels = ['educational', 'non-educational', 'entertainment', 'sports', 'tutorial']
    result = classifier(title, labels)

 
    label = result['labels'][0]
    if label in ['educational', 'tutorial']:
        classification = 'productive'
    else:
        classification = 'unproductive'

    return jsonify({'classification': classification})


@app.route('/save_screenshot/<folder>', methods=['POST'])
def save_screenshot(folder):
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    folder_path = os.path.join('screenshots', folder)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    with open(file_path, 'wb') as file:
        file.write(image_data)
    return jsonify({'status': 'success', 'file_path': file_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
