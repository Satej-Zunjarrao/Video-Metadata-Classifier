# Video Metadata Classifier
Built a machine learning-based system to automatically categorize video content and generate metadata tags.

# Video Metadata Classification System

## Overview
The **Video Metadata Classification System** is a Python-based solution designed to analyze video content and automatically generate metadata tags to enhance video management and searchability. The system utilizes machine learning models that combine visual features from video frames and textual data from transcribed audio to classify videos into predefined categories.

This project includes a scalable pipeline for video data extraction, feature engineering, multi-modal model development, and deployment as a REST API for easy integration with video platforms.

---

## Key Features
- **Data Collection**: Extracts audio and visual content from video files.
- **Data Preprocessing**: Cleans and tokenizes transcribed audio text for classification.
- **Feature Engineering**: Extracts visual features using pre-trained CNNs (e.g., ResNet) and generates text embeddings using Transformers (e.g., BERT).
- **Model Development**: Combines visual and textual features for multi-modal classification.
- **Metadata Tagging**: Classifies videos based on content and stores metadata in a SQL database.
- **Search Functionality**: Allows users to search for videos based on generated metadata tags.
- **API Deployment**: Deploys the entire pipeline as a REST API using Flask for easy integration.

---

## Directory Structure
```
project/
│
├── data_preprocessing.py         # Handles extraction and preprocessing of video data
├── feature_engineering.py        # Extracts visual features and generates text embeddings
├── model_development.py         # Defines and trains the multi-modal classification model
├── pipeline.py                  # Orchestrates the entire video processing pipeline
├── database_integration.py      # Manages interaction with the SQL database for metadata storage
├── search_interface.py          # Provides search functionality to query metadata tags
├── deploy_api.py                # Deploys the entire system as a Flask-based API
├── requirements.txt             # Lists the necessary dependencies for the project
├── README.md                    # Project documentation
```

# Modules

## 1. data_preprocessing.py
- Extracts audio from video using OpenCV and FFmpeg.
- Transcribes audio to text and performs tokenization and stop-word removal for text classification.

## 2. feature_engineering.py
- Extracts visual features from video frames using pre-trained ResNet CNN.
- Generates text embeddings using a Transformer model (e.g., BERT).

## 3. model_development.py
- Defines the multi-modal model combining visual and textual features.
- Trains the model using labeled datasets of video content and metadata.

## 4. pipeline.py
- Orchestrates the entire video metadata tagging pipeline.
- Extracts features, classifies videos, and stores metadata in the database.

## 5. database_integration.py
- Manages the SQL database connection and stores metadata tags for each video.
- Provides functions to query metadata from the database for video search.

## 6. search_interface.py
- Implements a search interface to retrieve videos based on metadata tags stored in the database.

## 7. deploy_api.py
- Deploys the video metadata tagging system as a REST API using Flask.
- Handles video uploads, metadata classification, and metadata search functionality.

## 8. requirements.txt
- Specifies the Python dependencies and libraries used in the project.

---

# Contact

For queries or collaboration, feel free to reach out:

- **Name**: Satej Zunjarrao  
- **Email**: zsatej1028@gmail.com
