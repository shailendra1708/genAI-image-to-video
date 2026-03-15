
#   Open: http://localhost:7860

import gradio as gr
import os
import time
from PIL import Image as PILImage

from embedding_generator import process_image, get_image_embedding
from vector_db           import load_index, add_vector, save_index, search_similar
from video_generator     import create_video, choose_effect
from rife_interpolator   import generate_ai_frames

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline(image, progress=gr.Progress()):

    if image is None:
        return None, "❌ Upload an image first.", "", "", "", "", None, None

    start_time = time.time()

    # Step 1 — Save image
    progress(0.10, desc="💾 Saving image...")
    timestamp  = int(time.time())
    image_path = os.path.join(UPLOAD_FOLDER, f"query_{timestamp}.png")
    PILImage.fromarray(image).save(image_path)

    # Step 2 — CLIP + BLIP
    progress(0.25, desc="🤖 Running CLIP + BLIP...")
    result       = process_image(image_path)
    clip_label   = result["clip_label"]
    confidence   = f"{result['confidence']:.2%}"
    blip_caption = result["blip_caption"]

    # Step 3 — FAISS
    progress(0.45, desc="🔍 Searching similar images...")
    try:
        index, metadata_list = load_index()
        if index is None:
            similar_paths = []
        else:
            if image_path not in metadata_list:
                add_vector(index, result["embedding"], metadata_list, image_path)
                save_index(index, metadata_list)
            similar_results = search_similar(index, result["embedding"], metadata_list, top_k=3)
            similar_paths   = [p for _, p in similar_results if p != image_path]
    except Exception as e:
        print(f"[FAISS] Error: {e}")
        similar_paths = []

    similar_text = "\n".join([f"#{i+1} {os.path.basename(p)}" for i, p in enumerate(similar_paths)]) \
                   or "No similar images found yet."

    # Step 4 — Effect selection
    progress(0.60, desc="🎨 Selecting effect...")
    effect = choose_effect(clip_label, blip_caption)

    # Step 5 — RIFE AI frame generation
    progress(0.75, desc="🎬 Generating AI frames (RIFE)...")
    all_frames, mode_used = generate_ai_frames(
        image_path  = image_path,
        clip_label  = clip_label,
        caption     = blip_caption,
        base_frames = 15,
        multiplier  = 2,
    )
    video_path = create_video(all_frames, output_filename=f"animation_{timestamp}.mp4")

    progress(1.0, desc="✅ Done!")
    elapsed = round(time.time() - start_time, 2)
    status  = f"✅ Done in {elapsed}s | Effect: {effect.upper()} | Frames: {len(all_frames)} | Mode: {mode_used}"

    return image, status, clip_label, confidence, blip_caption, similar_text, video_path, video_path


# ══════════════════════════════════════════════════════════════
# SIMPLE UI
# ══════════════════════════════════════════════════════════════

with gr.Blocks(title="GenAI Video Generator", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🎬 GenAI Animation Video Generator")
    gr.Markdown("**CLIP · BLIP · FAISS · RIFE · OpenCV · MoviePy**")
    gr.Markdown("---")

    # Status
    out_status = gr.Markdown("**Status:** Ready")

    # Row 1 — Upload + Preview
    with gr.Row():
        input_image   = gr.Image(label="📤 Upload Image", type="numpy", height=280)
        preview_image = gr.Image(label="🖼️ Preview",      height=280,  interactive=False)

    # Generate button
    run_button = gr.Button("🚀 Generate Animation", variant="primary", size="lg")

    gr.Markdown("---")

    # Row 2 — AI Results
    with gr.Row():
        out_clip   = gr.Textbox(label="🏷️ CLIP Label",   interactive=False)
        out_conf   = gr.Textbox(label="📈 Confidence",    interactive=False)
    out_caption    = gr.Textbox(label="📝 BLIP Caption",  interactive=False, lines=2)
    out_similar    = gr.Textbox(label="🔍 Similar Images (FAISS)", interactive=False, lines=3)

    gr.Markdown("---")

    # Row 3 — Video + Download
    out_video    = gr.Video(label="🎥 Generated Animation", height=400)
    download_btn = gr.File(label="⬇️ Download Video",       interactive=False)

    # Connect
    run_button.click(
        fn      = run_pipeline,
        inputs  = [input_image],
        outputs = [preview_image, out_status, out_clip, out_conf,
                   out_caption, out_similar, out_video, download_btn]
    )


# ══════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n★ Open browser → http://localhost:7860\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)