# database_integration.py
# This file handles the integration with the SQL database to store metadata tags and query them for video search functionality.
# It uses SQLAlchemy for database interaction and stores metadata (e.g., predicted categories) for efficient retrieval.

import sqlite3
from sqlite3 import Error

# Initialize the database connection
def create_connection(db_file):
    """
    Establish a connection to the SQLite database.

    Args:
        db_file (str): Path to the SQLite database file.

    Returns:
        Connection object or None: Connection to the database or None if connection fails.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to database {db_file}")
    except Error as e:
        print(f"Error: {e}")
    
    return conn


def create_table(conn):
    """
    Creates a table in the database to store video metadata.

    Args:
        conn (Connection): The database connection object.

    Returns:
        None
    """
    try:
        sql_create_metadata_table = """
        CREATE TABLE IF NOT EXISTS video_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL,
            metadata TEXT NOT NULL
        );
        """
        cursor = conn.cursor()
        cursor.execute(sql_create_metadata_table)
        print("Table created successfully.")
    except Error as e:
        print(f"Error: {e}")


def insert_metadata(conn, video_name, metadata):
    """
    Inserts metadata for a video into the database.

    Args:
        conn (Connection): The database connection object.
        video_name (str): The name of the video.
        metadata (str): The metadata (predicted tags) for the video.

    Returns:
        None
    """
    try:
        sql_insert_metadata = """
        INSERT INTO video_metadata (video_name, metadata)
        VALUES (?, ?);
        """
        cursor = conn.cursor()
        cursor.execute(sql_insert_metadata, (video_name, metadata))
        conn.commit()
        print(f"Metadata for {video_name} inserted successfully.")
    except Error as e:
        print(f"Error: {e}")


def fetch_metadata(conn, video_name):
    """
    Fetches metadata for a specific video based on its name.

    Args:
        conn (Connection): The database connection object.
        video_name (str): The name of the video for which metadata is to be fetched.

    Returns:
        str: The metadata of the video.
    """
    try:
        sql_fetch_metadata = "SELECT metadata FROM video_metadata WHERE video_name = ?;"
        cursor = conn.cursor()
        cursor.execute(sql_fetch_metadata, (video_name,))
        result = cursor.fetchone()
        if result:
            print(f"Metadata for {video_name} found.")
            return result[0]
        else:
            print(f"No metadata found for {video_name}.")
            return None
    except Error as e:
        print(f"Error: {e}")
        return None


# Example usage
if __name__ == "__main__":
    db_file = "video_metadata.db"  # SQLite database file
    conn = create_connection(db_file)

    if conn:
        create_table(conn)

        # Insert metadata for a video (example)
        video_name = "example_video.mp4"
        metadata = "Predicted class: 5"
        insert_metadata(conn, video_name, metadata)

        # Fetch metadata for a video (example)
        fetched_metadata = fetch_metadata(conn, video_name)
        print(f"Fetched Metadata: {fetched_metadata}")
