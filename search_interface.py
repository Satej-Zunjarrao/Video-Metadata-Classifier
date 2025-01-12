# search_interface.py
# This file implements the search functionality, allowing users to search for videos based on metadata tags stored in the SQL database.
# It uses Flask to create a simple search interface that queries the database and returns relevant video metadata.

from flask import Flask, request, jsonify
from database_integration import create_connection, fetch_metadata

# Initialize Flask app
app = Flask(__name__)

# Initialize the database connection
db_file = "video_metadata.db"  # SQLite database file
conn = create_connection(db_file)

@app.route('/search', methods=['GET'])
def search_video():
    """
    Endpoint to search for a video based on its metadata tags.

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
    app.run(debug=True)
