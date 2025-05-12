from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)

    image = cv2.imread(filepath)
    results = model(image)[0]
    annotated = results.plot()

    output_path = os.path.join(UPLOAD_FOLDER, f"detected_{filename}")
    cv2.imwrite(output_path, annotated)

    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
