# deploy_api.py
# This file handles the deployment of the Flask-based REST API for the video metadata tagging system.
# It integrates all components into a deployable API, allowing easy interaction with the video metadata classifier.

from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from model_development import MultiModalModel, transformer_tokenizer, resnet_model
from feature_engineering import extract_visual_features, generate_text_embeddings
from database_integration import create_connection, insert_metadata

# Initialize Flask app
app = Flask(__name__)

# Database configuration
db_file = "video_metadata.db"  # SQLite database file
conn = create_connection(db_file)

# File upload settings
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model for inference
model = MultiModalModel(resnet_model=resnet_model, transformer_model="bert-base-uncased")
model.eval()


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
        JSON response: Contains metadata (predicted tags) for the video.
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

        # Step 1: Extract visual features from the video
        visual_features = extract_visual_features(video_path)

        # Step 2: Generate text embeddings from the transcribed audio (mocked in this example)
        transcribed_text = "This is a mock transcription of the video's audio."
        text_input = generate_text_embeddings(transcribed_text)

        # Step 3: Make prediction using the multi-modal model
        with torch.no_grad():
            visual_tensor = torch.tensor(visual_features).unsqueeze(0)  # Add batch dimension
            text_tensor = text_input
            output = model(visual_tensor, text_tensor)
            predicted_label = torch.argmax(output, dim=1).item()

        # Step 4: Store metadata in the database
        metadata = f"Predicted class: {predicted_label}"
        insert_metadata(conn, filename, metadata)

        # Return the metadata as a response
        return jsonify({"video_name": filename, "metadata": metadata})


@app.route('/search', methods=['GET'])
def search_video():
    """
    Endpoint to search for a video based on its metadata tags stored in the database.

    Returns:
        JSON: A list of video names matching the search query.
    """
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query parameter provided."}), 400

    # Search the database for videos with metadata matching the query
    matching_videos = []
    cursor = conn.cursor()
    cursor.execute("SELECT video_name, metadata FROM video_metadata")
    rows = cursor.fetchall()

    for row in rows:
        video_name, metadata = row
        if query.lower() in metadata.lower():
            matching_videos.append({"video_name": video_name, "metadata": metadata})

    if matching_videos:
        return jsonify(matching_videos)
    else:
        return jsonify({"message": "No matching videos found."}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Expose API on all network interfaces at port 5000
