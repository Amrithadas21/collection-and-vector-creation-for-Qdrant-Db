from qdrant_client import QdrantClient, models
import pandas as pd

# Initialize the Qdrant client
qdrant_client = QdrantClient(
    url="https://9a319ec6-faea-4a0f-9a20-b3e6e2b020c0.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="CyOKqbWb5d4iS9uduG1f6Uo5otZ9lOMfPdGTKBh5Od6OKhZJ95S_Cg"
)

# Create a collection
collection_name = "my_collection_images"

# Attempt to create the collection with vectors_config
try:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)  # Specify vector parameters
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

# Load the image vectors from a CSV file
try:
    image_vectors = pd.read_csv(r'C:\Users\amrit\Documents\image_vectors.csv').values  # Load as a NumPy array
    print("Image vectors loaded successfully.")
except Exception as e:
    print(f"Error loading image vectors: {e}")

# Add the vectors to the Qdrant collection
try:
    # Prepare the payload for Qdrant
    points = []
    for idx, vector in enumerate(image_vectors):
        points.append(models.PointStruct(id=idx, vector=vector.tolist()))  # Convert to list for Qdrant

    # Upload points to the collection
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully added {len(points)} vectors to the collection '{collection_name}'.")
except Exception as e:
    print(f"Error adding vectors to collection: {e}")