import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import requests
import numpy as np
import pandas as pd
import pickle
from urllib.parse import urlparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load the OpenAI CLIP processor and model
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Load dataset CSV and extract Image URLs and Names
#dataset_path = './lib/dataset_test.csv'
dataset_path = './lib/dataset.csv'
df = pd.read_csv(dataset_path)
images = df['Image URL'].tolist()
img_names = df['Name'].tolist()


def load_image(image_path):
    allowed_extensions = ('.jpg', '.jpeg', '.png')

    # Parse the URL to remove query parameters and fragments
    parsed_url = urlparse(image_path)
    path_without_query = parsed_url.path

    # Check if the path (without query parameters) has an allowed extension
    if not path_without_query.lower().endswith(allowed_extensions):
        print(f"Skipping {image_path}: Unsupported file type.")
        return None

    try:
        # Use the URL for HTTP requests or local file paths
        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path, stream=True)

            if response.status_code == 403:
                print(f"Skipping {image_path}: Access denied (403 Forbidden).")
                return None
        
            content_type = response.headers.get('Content-Type')
            if content_type not in ['image/jpeg','image/png']:
                print(f"Skipping {image_path}: Fetched image file is in an incorrect format")
                return None
            
            return Image.open(response.raw)
        else:
            return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def cosine_similarity(vec1, vec2):
    # Convert tensors to numpy arrays if they aren't already
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.cpu().numpy()
    
    # Compute the dot product of vec1 and vec2
    dot_product = np.dot(vec1, vec2)
    
    # Compute the L2 norm of vec1 and vec2
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity


def extract_features_clip(image):
    with torch.no_grad():
        inputs1 = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs1)
    return image_features

filename='images.pkl'

# Check if 'images.pkl' already exists
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        features = pickle.load(f)
else:
    # Process and save features if 'images.pkl' doesn't exist
    total_image_features = []
    for image_url in images:
        img = load_image(image_url)
        if img is None:
            continue  # Skip unsupported or failed images
        clip_feature = extract_features_clip(img)
        total_image_features.append(clip_feature)
    
    with open(filename, 'wb') as f:
        pickle.dump(total_image_features, f)
    features = total_image_features


def image_similarity(url):
    image = load_image(url)

    if image is None:
        return {"error": f"Failed to load image from {url}. Make sure the URL or file path is correct and the file type is supported (JPG, JPEG, PNG)."}
    
    image_features=extract_features_clip(image)
    similarity_scores = [cosine_similarity(image_features, i[0]) for i in features]
    merged_dict=dict(zip(img_names,similarity_scores))
    sorted_dict = dict(sorted(merged_dict.items(), key=lambda item: item[1],reverse=True))
    return sorted_dict

# !~~TEST~~!

test_url = images[0]  # Use the first image URL from the dataset
results = image_similarity(test_url)

# Example of sorting and displaying the top N closest matches
top_n = 10 
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

# Print the top matches with their confidence scores
for item, score in sorted_results[:top_n]:
    print(f"Recall Item: {item}, Confidence Score: {score[0]:.2f}")