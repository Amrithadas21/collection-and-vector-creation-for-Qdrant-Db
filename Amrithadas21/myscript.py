from datasets import load_dataset
import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms

# Load the dataset
ds = load_dataset("rokmr/pets")
train_ds = ds['train']
test_ds = ds['test']

# Define output directory and image size
output_dir = 'C:/Users/amrit/OneDrive/Documents/output dir'
img_size = 640

# Create output directories
os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train/embeddings'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test/embeddings'), exist_ok=True)

# Load pre-trained model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to generate embeddings
def generate_embedding(image):
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        embedding = model(image_tensor)
        return embedding.squeeze().numpy()  # Remove batch dimension

# Preprocess training images and generate embeddings
for i, sample in enumerate(train_ds):
    img = sample['image']
    label = sample['label']

    # Convert PIL image to numpy array
    img = np.array(img)

    # Check if the image is grayscale (2D)
    if img.ndim == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    elif img.ndim == 3 and img.shape[2] == 4:  # If the image has an alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

    # Resize and pad image
    scale = img_size / max(img.shape[:2])
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Create a padded image
    padded_img = np.full((img_size, img_size, 3), 128)  # Fill with gray
    padded_img[:new_h, :new_w] = img_resized

    # Save preprocessed image
    cv2.imwrite(os.path.join(output_dir, 'train/images', f'train_{i}.jpg'), padded_img)

    # Generate and save embedding
    embedding = generate_embedding(padded_img)
    np.save(os.path.join(output_dir, 'train/embeddings', f'train_{i}.npy'), embedding)

    # Save corresponding label (if needed, adjust as per your label format)
    with open(os.path.join(output_dir, 'train/labels', f'train_{i}.txt'), 'w') as f:
        f.write(str(label))  # Adjust this line based on your label format

# Preprocess test images and generate embeddings
for i, sample in enumerate(test_ds):
    img = sample['image']
    label = sample['label']

    # Convert PIL image to numpy array
    img = np.array(img)

    # Check if the image is grayscale (2D)
    if img.ndim == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    elif img.ndim == 3 and img.shape[2] == 4:  # If the image has an alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

    # Resize and pad image
    scale = img_size / max(img.shape[:2])
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Create a padded image
    padded_img = np.full((img_size, img_size, 3), 128)  # Fill with gray
    padded_img[:new_h, :new_w] = img_resized

    # Save preprocessed image
    cv2.imwrite(os.path.join(output_dir, 'test/images', f'test_{i}.jpg'), padded_img)

    # Generate and save embedding
    embedding = generate_embedding(padded_img)
    np.save(os.path.join(output_dir, 'test/embeddings', f'test_{i}.npy'), embedding)

    # Save corresponding label (if needed, adjust as per your label format)
    with open(os.path.join(output_dir, 'test/labels', f'test_{i}.txt'), 'w') as f:
        f.write(str(label))  # Adjust this line based on your label format

print("Preprocessing and embedding generation complete!")