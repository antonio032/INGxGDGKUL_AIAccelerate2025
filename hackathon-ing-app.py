import os
import numpy as np
import faiss  # import Faiss
import pandas as pd
import vertexai
from vertexai.language_models import TextEmbeddingModel

# --- 0. Initialize GCP ---
PROJECT_ID = "wis-exercise-4-api"  # replace with your project ID
LOCATION = "europe-west1"            # replace with your location
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- 1. Load the embedding model ---
# We'll use this model to convert text into vectors
print("Loading embedding model...")
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

# --- 2. Prepare texts to index ---
CHUNKS_DIR_NL = "./chunks/500_750_processed_be_nl_2025_09_23/"  # assume this is your Dutch chunks path
all_chunk_texts = []
chunk_file_names = []  # we need this to know which file corresponds to each vector

print(f"Reading files from {CHUNKS_DIR_NL}...")
for filename in os.listdir(CHUNKS_DIR_NL):
    if filename.endswith(".txt"):
        file_path = os.path.join(CHUNKS_DIR_NL, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            all_chunk_texts.append(f.read())
            chunk_file_names.append(filename)  # store file name

print(f"Found {len(all_chunk_texts)} text chunks.")

# --- 3. Generate embeddings ---
# Note: get_embeddings accepts a list
print("Generating vectors for all chunks...")

# Vertex AI limits number of instances per request (usually 250). We need to batch.
batch_size = 250
all_embeddings = []
for i in range(0, len(all_chunk_texts), batch_size):
    batch_texts = all_chunk_texts[i:i + batch_size]
    print(f"  - Processing batch {i // batch_size + 1}...")
    # get embeddings for the current batch
    batch_embeddings = embedding_model.get_embeddings(batch_texts)
    all_embeddings.extend(batch_embeddings)


# convert Vertex AI embedding objects to numpy arrays
# Faiss requires numpy arrays
embedding_vectors = [e.values for e in all_embeddings]
embedding_vectors_np = np.array(embedding_vectors, dtype='float32')

print(f"Vectors generated, dimensions: {embedding_vectors_np.shape}")

# --- 4. Create and populate Faiss index ---
# Faiss needs to know the vector dimension (e.g., 768)
dimension = embedding_vectors_np.shape[1]
index = faiss.IndexFlatL2(dimension)  # use L2 distance (Euclidean)

# add vectors to the index
index.add(embedding_vectors_np)

print(f"Faiss index created and populated with {index.ntotal} vectors.")

# --- 5. Save your work (very important!) ---
# You don't want to rerun this process every time
faiss.write_index(index, "ing_chunks_nl.index")

# You also need to save the filename mapping
# IDs in the index (0, 1, 2, ...) correspond to indices in chunk_file_names list
# For simplicity, we save the mapping to a simple .txt file
with open("ing_chunks_nl.map", "w", encoding='utf-8') as f:
    for name in chunk_file_names:
        f.write(f"{name}\n")

print("Index and file mapping saved to local disk!")
print("Stage 1 complete.")