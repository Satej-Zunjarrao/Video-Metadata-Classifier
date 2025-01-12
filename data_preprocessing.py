# data_preprocessing.py
# This file handles the preprocessing of video data, including extracting audio and visual content,
# transcribing the audio to text, and cleaning the text for further classification tasks.
# Libraries: OpenCV, FFmpeg, NLTK, SpeechRecognition

import cv2
import os
import nltk
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data for tokenization and stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stop words
stop_words = set(stopwords.words('english'))

def extract_audio_from_video(video_path: str, audio_output_path: str):
    """
    Extracts the audio from the given video file and saves it as an audio file.

    Args:
        video_path (str): Path to the input video file.
        audio_output_path (str): Path to save the extracted audio file.

    Returns:
        None
    """
    # Use FFmpeg to extract audio from video
    os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_output_path}")
    print(f"Audio extracted and saved at {audio_output_path}")


def audio_to_text(audio_path: str) -> str:
    """
    Converts speech in the audio file to text using SpeechRecognition.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: The transcribed text from the audio.
    """
    recognizer = sr.Recognizer()

    # Load the audio file using SpeechRecognition
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    # Transcribe the audio to text
    try:
        text = recognizer.recognize_google(audio_data)
        print("Audio transcription successful.")
    except sr.UnknownValueError:
        text = "Audio could not be transcribed."
        print("Audio could not be transcribed.")
    except sr.RequestError as e:
        text = f"Error with speech recognition service: {e}"
        print(f"Error with speech recognition service: {e}")
    
    return text


def clean_and_tokenize_text(text: str) -> list:
    """
    Cleans and tokenizes the text data, removing stopwords and non-alphanumeric characters.

    Args:
        text (str): The raw transcribed text to be processed.

    Returns:
        list: A list of cleaned and tokenized words.
    """
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and non-alphanumeric tokens
    cleaned_tokens = [word.lower() for word in tokens if word.isalpha() and word not in stop_words]
    
    print(f"Cleaned and tokenized text: {cleaned_tokens[:10]}...")  # Display first 10 tokens for preview
    return cleaned_tokens


# Example usage
if __name__ == "__main__":
    video_file_path = "path/to/video.mp4"  # Replace with actual video file path
    audio_file_path = "path/to/audio.wav"  # Replace with desired audio output path

    # Step 1: Extract audio from video
    extract_audio_from_video(video_file_path, audio_file_path)

    # Step 2: Convert audio to text
    transcribed_text = audio_to_text(audio_file_path)

    # Step 3: Clean and tokenize the transcribed text
    cleaned_tokens = clean_and_tokenize_text(transcribed_text)
