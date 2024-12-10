from qdrant_client import QdrantClient, models
import numpy as np
import os

# Initialize the Qdrant client
qdrant_client = QdrantClient(
    url="https://9a319ec6-faea-4a0f-9a20-b3e6e2b020c0.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eqz6zqIoLjes2MNuK94T3mGTAmWLlSGfbA83ohcBZ2QoBF1Cws4hcA"
)

# Create a collection
collection_name = "mycollection2"

# Attempt to create the collection with vectors_config
try:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1000, distance=models.Distance.COSINE)  # Adjust size based on your model output
    )
    print(f"Collection '{collection_name}' created successfully.")
except Exception as e:
    print(f"Error creating collection: {e}")

# Test the connection by listing collections
try:
    collections = qdrant_client.get_collections()
    print("Collections:", collections)
except Exception as e:
    print("Error connecting to Qdrant:", e)

# Load the embeddings from .npy files
embeddings_dir = r'C:/Users/amrit/OneDrive/Documents/output dir/train/embeddings'  # Adjust this path as needed
points = []

try:
    for idx, filename in enumerate(os.listdir(embeddings_dir)):
        if filename.endswith('.npy'):
            # Load the embedding
            embedding = np.load(os.path.join(embeddings_dir, filename))
            if embedding.shape[0] != 1000:  # Ensure the embedding size matches
                print(f"Warning: Embedding at index {idx} has size {embedding.shape[0]}, expected 1000.")
                continue  # Skip this embedding if the size is incorrect
            
            points.append(models.PointStruct(id=idx, vector=embedding.tolist()))  # Convert to list for Qdrant

    # Upload points to the collection
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully added {len(points)} vectors to the collection '{collection_name}'.")
except Exception as e:
    print(f"Error adding vectors to collection: {e}")