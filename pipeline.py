# pipeline.py
# This file integrates the entire process, from video upload to metadata generation, as a pipeline.
# It processes videos, extracts features, classifies them, and serves the model as a REST API using Flask.

import os
import torch
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model_development import MultiModalModel, transformer_tokenizer, resnet_model, train_model
from feature_engineering import extract_visual_features, generate_text_embeddings
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model (for inference)
model = MultiModalModel(resnet_model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True),
                        transformer_model='bert-base-uncased')
model.eval()

# Define file upload configurations
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """
    Check if the uploaded file is of an allowed format.

    Args:
        filename (str): The file name.

    Returns:
        bool: Whether the file is allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to upload a video file. The video is processed to generate metadata tags.

    Returns:
        str: JSON response with success or failure.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Extract features from video
        visual_features = extract_visual_features(video_path)
        transcribed_text = "This is a mock transcription of the video's audio."
        text_input = generate_text_embeddings(transcribed_text)

        # Make prediction using the multi-modal model
        with torch.no_grad():
            visual_tensor = torch.tensor(visual_features).unsqueeze(0)  # Add batch dimension
            text_tensor = text_input
            output = model(visual_tensor, text_tensor)
            predicted_label = torch.argmax(output, dim=1).item()

        return jsonify({"metadata": f"Predicted class: {predicted_label}"})


if __name__ == '__main__':
    app.run(debug=True)
