

import os, sys
from embedding_generator import process_image, get_image_embedding
from vector_db           import create_index, add_vector, search_similar, save_index, load_index
from video_generator     import generate_frames_from_image, create_video, create_slideshow_video, choose_effect
from rife_interpolator   import generate_ai_frames

UPLOAD_FOLDER    = "uploads"
OUTPUT_FOLDER    = "outputs"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def index_all_images():
    """Scan uploads/ → CLIP embedding → FAISS. Also runs BLIP captions."""
    print("\n" + "═"*55)
    print("  STEP 1 — Indexing images into FAISS")
    print("═"*55)

    index, metadata_list = load_index()
    caption_map = {}

    image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)]

    if not image_files:
        print("[App] No images in uploads/ folder.")
        return index, metadata_list, caption_map

    print(f"[App] Found {len(image_files)} image(s).")

    for filename in image_files:
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        if image_path in metadata_list:
            print(f"[App] Already indexed: {filename}")
            continue

        result = process_image(image_path)
        add_vector(index, result["embedding"], metadata_list, image_path)
        caption_map[image_path] = {
            "clip_label"   : result["clip_label"],
            "confidence"   : result["confidence"],
            "blip_caption" : result["blip_caption"],
        }

    save_index(index, metadata_list)
    print(f"\n[App] Indexing complete. Total vectors: {index.ntotal}")
    return index, metadata_list, caption_map


def find_similar_images(query_image_path, index, metadata_list, top_k=3):
    """CLIP embed query image → FAISS search → return similar paths."""
    print("\n" + "═"*55)
    print("  STEP 2 — FAISS Similarity Search")
    print("═"*55)

    query_embedding = get_image_embedding(query_image_path)
    results         = search_similar(index, query_embedding, metadata_list, top_k=top_k)

    if not results:
        print("[App] No similar images found.")
        return []

    similar_paths = []
    print(f"\n[App] Top {len(results)} similar images:")
    for rank, (distance, path) in enumerate(results, 1):
        # Skip query image itself (distance 0.0 = same image)
        if path == query_image_path or distance < 0.0001:
            print(f"  #{rank}  {path}  (distance: {distance:.4f}) ← skipped (self)")
            continue
        print(f"  #{rank}  {path}  (distance: {distance:.4f})")
        similar_paths.append(path)

    return similar_paths


def generate_animation(query_image_path, similar_image_paths, clip_label="unknown", caption=""):
    """
    AI-Enhanced Animation Pipeline:
        Step 1 → OpenCV generates 15 base frames (content-aware effect)
        Step 2 → RIFE doubles frames to 29 via AI interpolation
        Result → smooth 29-frame video at 10fps ≈ 3 seconds
        Falls back to OpenCV only if RIFE fails.
    """
    print("\n" + "═"*55)
    print("  STEP 3 — AI-Enhanced Animation (OpenCV + RIFE)")
    print("═"*55)

    all_images = [query_image_path] + similar_image_paths
    seen, unique_images = set(), []
    for img in all_images:
        if img not in seen:
            seen.add(img); unique_images.append(img)

    all_frames, mode_used = generate_ai_frames(
        image_path  = unique_images[0],
        clip_label  = clip_label,
        caption     = caption,
        base_frames = 15,
        multiplier  = 2,
    )
    print(f"[App] Mode: {mode_used} | Total frames: {len(all_frames)}")
    return create_video(all_frames, output_filename="animation.mp4")


def main():
    print("\n" + "★"*55)
    print("  GenAI Animation Video Generator")
    print("  CLIP + BLIP + FAISS + RIFE + MoviePy")
    print("★"*55)

    upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if not upload_files:
        print("\n[App] ERROR: No images in uploads/. Add images and retry.")
        sys.exit(1)

    query_image = os.path.join(UPLOAD_FOLDER, upload_files[0])
    print(f"\n[App] Query image: {query_image}")

    index, metadata_list, caption_map = index_all_images()
    similar_images = find_similar_images(query_image, index, metadata_list, top_k=3)
    query_result   = process_image(query_image)

    video_path = generate_animation(
        query_image,
        similar_images,
        clip_label = query_result["clip_label"],
        caption    = query_result["blip_caption"]
    )

    print("\n" + "★"*55)
    print("  PIPELINE COMPLETE")
    print("★"*55)
    print(f"  Query Image   : {query_image}")
    print(f"  CLIP Label    : {query_result['clip_label']} ({query_result['confidence']:.2%})")
    print(f"  BLIP Caption  : {query_result['blip_caption']}")
    print(f"  Similar Found : {len(similar_images)}")
    print(f"  Video Saved   : {video_path}")
    print("★"*55 + "\n")


if __name__ == "__main__":
    main()