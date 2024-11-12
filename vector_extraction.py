import os
import pandas as pd
from PIL import Image
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array

# Specify the folder path containing images
folder_path = r"C:\Users\amrit\Downloads\archive (12)\Animals\snakes"  # Change this to your folder path

# Initialize a list to hold the images
images = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
        img_path = os.path.join(folder_path, filename)  # Get the full path of the image
        img = Image.open(img_path)  # Load the image
        images.append(img)  # Append the image to the list

# Check the number of images loaded
print(f"Loaded {len(images)} images.")

# Function to preprocess images for VGG16
def preprocess_images(images):
    processed_images = []
    for img in images:
        # Resize the image to 224x224 pixels
        img_resized = img.resize((224, 224))
        # Convert the image to a numpy array
        img_array = img_to_array(img_resized)
        # Expand dimensions to match the input shape of VGG16
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess the image
        img_preprocessed = preprocess_input(img_array)
        processed_images.append(img_preprocessed)
    return np.vstack(processed_images)

# Preprocess the loaded images
images_preprocessed = preprocess_images(images)

# Check the shape of the preprocessed images
print(f"Shape of preprocessed images: {images_preprocessed.shape}")

# Load the VGG16 model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')  # Use 'avg' pooling to get a 4096-dimensional vector

# Extract features (vectors) from the preprocessed images
image_vectors = model.predict(images_preprocessed)

# Check the shape of the extracted vectors
print(f"Shape of extracted image vectors: {image_vectors.shape}")  # Should be (number_of_images, 4096)

# Optionally, convert the vectors to a DataFrame for easier handling
df_vectors = pd.DataFrame(image_vectors)

# Save the DataFrame to a CSV file (optional)
output_csv_path = r"C:\Users\amrit\Downloads\image_vectors.csv"  # Change this to your desired output path
df_vectors.to_csv(output_csv_path, index=False)

print(f"Image vectors saved to {output_csv_path}")