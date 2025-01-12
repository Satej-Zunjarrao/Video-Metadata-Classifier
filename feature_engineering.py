# feature_engineering.py
# This file handles the extraction of visual features from video frames using pre-trained CNN models like ResNet,
# and the generation of text embeddings using Transformer models from Hugging Face.

import cv2
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from torchvision import models, transforms
from PIL import Image

# Load pre-trained models
transformer_model = AutoModel.from_pretrained("bert-base-uncased")
transformer_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load pre-trained ResNet model for visual feature extraction
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Define image transformation pipeline for ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_visual_features(video_path: str) -> np.ndarray:
    """
    Extracts visual features from a video by processing individual frames through a pre-trained CNN (ResNet).

    Args:
        video_path (str): Path to the input video file.

    Returns:
        np.ndarray: A 2D array with the extracted visual features from each frame.
    """
    cap = cv2.VideoCapture(video_path)
    visual_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image for transformation
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transformed_frame = resnet_transform(frame_image).unsqueeze(0)

        # Extract features from the frame using ResNet
        with torch.no_grad():
            features = resnet_model(transformed_frame)
            visual_features.append(features.squeeze().cpu().numpy())

    cap.release()
    return np.array(visual_features)


def generate_text_embeddings(text: str) -> torch.Tensor:
    """
    Converts input text to embeddings using a pre-trained Transformer model.

    Args:
        text (str): The input text for which embeddings are to be generated.

    Returns:
        torch.Tensor: The tensor representation of the input text.
    """
    inputs = transformer_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = transformer_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling to get the sentence embedding

    return embeddings


# Example usage
if __name__ == "__main__":
    video_path = "path/to/video.mp4"  # Replace with actual video file path
    sample_text = "This is an example text for generating embeddings."

    # Step 1: Extract visual features from the video
    visual_features = extract_visual_features(video_path)

    # Step 2: Generate text embeddings
    text_embeddings = generate_text_embeddings(sample_text)

    print(f"Extracted visual features shape: {visual_features.shape}")
    print(f"Generated text embeddings shape: {text_embeddings.shape}")
