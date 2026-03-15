

import faiss        # Facebook AI Similarity Search library
import numpy as np  # For working with vectors (arrays of numbers)
import pickle       # For saving/loading Python objects to disk
import os           # For checking if files exist

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DIMENSION  = 512                 # CLIP produces 512-dimensional vectors
INDEX_FILE = "faiss_index.bin"   # Where FAISS index is saved on disk
META_FILE  = "faiss_meta.pkl"    # Where image paths are saved on disk


# ══════════════════════════════════════════════════════════════
# FUNCTION 1 — Create a new FAISS index
# ══════════════════════════════════════════════════════════════

def create_index():
    """
    Create a brand new FAISS index.

    IndexFlatL2 means:
        Flat = stores all vectors (no compression)
        L2   = uses Euclidean distance to measure similarity
               smaller distance = more similar images

    Returns:
        faiss index object (empty, ready to store vectors)
    """
    index = faiss.IndexFlatL2(DIMENSION)
    print(f"[FAISS] New index created | Dimension: {DIMENSION}")
    return index


# ══════════════════════════════════════════════════════════════
# FUNCTION 2 — Add one image embedding to FAISS
# ══════════════════════════════════════════════════════════════

def add_vector(index, vector, metadata_list, image_path):
    """
    Store one image embedding inside FAISS.

    How it works:
        1. Reshape vector to correct format (1 row, 512 columns)
        2. Add vector to FAISS index
        3. Save image path in metadata_list

    Args:
        index         : FAISS index object
        vector        : numpy array shape (1, 512) from CLIP
        metadata_list : Python list storing image file paths
        image_path    : string path to the image file
    """
    vector = np.array(vector).astype("float32").reshape(1, DIMENSION)
    index.add(vector)
    metadata_list.append(image_path)
    print(f"[FAISS] Vector added → {image_path} | Total stored: {index.ntotal}")


# ══════════════════════════════════════════════════════════════
# FUNCTION 3 — Search for similar images
# ══════════════════════════════════════════════════════════════

def search_similar(index, query_vector, metadata_list, top_k=3):
    """
    Find the most similar images to a query image.

    How it works:
        1. Take query image embedding (512-dim vector)
        2. Compare with ALL stored vectors using L2 distance
        3. Return top_k closest matches

    Args:
        index         : FAISS index object
        query_vector  : numpy array shape (1, 512)
        metadata_list : list of image paths
        top_k         : how many results to return (default 3)

    Returns:
        List of tuples → [(distance, image_path), ...]
        Smaller distance = more similar
    """
    if index.ntotal == 0:
        print("[FAISS] Index is empty — no vectors stored yet.")
        return []

    query_vector = np.array(query_vector).astype("float32").reshape(1, DIMENSION)
    k = min(top_k, index.ntotal)

    # Core FAISS search
    # distances : how far each result is  (smaller = more similar)
    # indices   : position in metadata_list
    distances, indices = index.search(query_vector, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        image_path = metadata_list[idx]
        results.append((float(dist), image_path))
        print(f"[FAISS] Match → {image_path} | Distance: {dist:.4f}")

    return results


# ══════════════════════════════════════════════════════════════
# FUNCTION 4 — Save index to disk
# ══════════════════════════════════════════════════════════════

def save_index(index, metadata_list):
    """
    Save FAISS index + metadata to disk (persists between runs).

    Saves:
        faiss_index.bin → the actual FAISS vectors
        faiss_meta.pkl  → the image path list
    """
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata_list, f)
    print(f"[FAISS] Saved → {INDEX_FILE} + {META_FILE} | Vectors: {index.ntotal}")


# ══════════════════════════════════════════════════════════════
# FUNCTION 5 — Load index from disk
# ══════════════════════════════════════════════════════════════

def load_index():
    """
    Load saved FAISS index from disk.
    If no saved index → create fresh empty one.

    Returns:
        (index, metadata_list) tuple
    """
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata_list = pickle.load(f)
        print(f"[FAISS] Loaded from disk | Total vectors: {index.ntotal}")
        return index, metadata_list
    else:
        print("[FAISS] No saved index found → creating new empty index.")
        return create_index(), []


# ══════════════════════════════════════════════════════════════
# FUNCTION 6 — Reset / clear the index
# ══════════════════════════════════════════════════════════════

def reset_index():
    """
    Delete saved files and return fresh empty index.
    Use when you want to re-index all images from scratch.
    """
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        print(f"[FAISS] Deleted {INDEX_FILE}")
    if os.path.exists(META_FILE):
        os.remove(META_FILE)
        print(f"[FAISS] Deleted {META_FILE}")
    print("[FAISS] Index reset complete.")
    return create_index(), []


# ══════════════════════════════════════════════════════════════
# Quick Test — python vector_db.py
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  FAISS Vector DB — Quick Test")
    print("="*50)

    # Step 1: Create fresh index
    index, meta = create_index(), []

    # Step 2: Add 5 fake vectors (simulating CLIP embeddings)
    print("\n[Test] Adding 5 random vectors...")
    fake_images = [
        "uploads/anime.png",
        "uploads/ninja.png",
        "uploads/city.png",
        "uploads/forest.png",
        "uploads/space.png",
    ]
    for img in fake_images:
        fake_vector = np.random.random((1, 512)).astype("float32")
        add_vector(index, fake_vector, meta, img)

    # Step 3: Search with a random query vector
    print("\n[Test] Searching for 3 similar vectors...")
    query   = np.random.random((1, 512)).astype("float32")
    results = search_similar(index, query, meta, top_k=3)

    print("\n[Test] Results:")
    for rank, (dist, path) in enumerate(results, 1):
        print(f"  #{rank} → {path} (distance: {dist:.4f})")

    # Step 4: Save and reload
    print("\n[Test] Saving index...")
    save_index(index, meta)

    print("\n[Test] Loading index back from disk...")
    loaded_index, loaded_meta = load_index()
    print(f"[Test] Loaded vectors: {loaded_index.ntotal}")

    print("\n[Test] FAISS is working correctly!")
    print("="*50 + "\n")