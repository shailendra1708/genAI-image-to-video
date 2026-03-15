
import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
)

# ─────────────────────────────────────────────────────────────
# Device setup  (GPU if available, else CPU)
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using: {DEVICE.upper()}")


#  Load CLIP Model (embedding generator)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

print("[CLIP] Loading model...")
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()
print("[CLIP] Model loaded successfully.")


# Load BLIP Model (caption generator)

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

print("[BLIP] Loading model...")
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model     = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)
blip_model.eval()
print("[BLIP] Model loaded successfully.")


# CLIP

def get_image_embedding(image_path):
    """
    Convert image → 512-dimensional vector using CLIP.
    This vector is stored in FAISS for similarity search.

    Returns:
        numpy array shape (1, 512) float32
    """
    image  = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = clip_model.get_image_features(**inputs)

    # Fix: some CLIP versions return BaseModelOutputWithPooling object
    # instead of a plain tensor — extract the tensor safely
    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output        # extract from object
    elif hasattr(output, "last_hidden_state"):
        embedding = output.last_hidden_state[:, 0, :]  # CLS token
    else:
        embedding = output                      # already a plain tensor

    embedding = embedding.cpu().numpy().astype("float32")
    print(f"[CLIP] Image embedding created | Shape: {embedding.shape} | File: {image_path}")
    return embedding  # shape: (1, 512)


def get_text_embedding(text):
    """
    Convert text → 512-dimensional vector using CLIP.
    Useful for text-based similarity search in FAISS.

    Returns:
        numpy array shape (1, 512) float32
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)

    if hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        embedding = output.last_hidden_state[:, 0, :]
    else:
        embedding = output

    embedding = embedding.cpu().numpy().astype("float32")
    print(f"[CLIP] Text embedding created | Shape: {embedding.shape} | Text: '{text}'")
    return embedding


def get_clip_label(image_path):
    """
    Zero-shot image classification using CLIP.
    Compares image to fixed candidate labels → returns best match.

    Returns:
        (label string, confidence float)
    """
    candidate_labels = [
        "anime character", "ninja fighting", "city skyline",
        "nature landscape", "fantasy scene", "action scene",
        "portrait", "space scene", "underwater scene", "cartoon character",
    ]

    image  = Image.open(image_path).convert("RGB")
    inputs = clip_processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs    = clip_model(**inputs)
        probs      = outputs.logits_per_image.softmax(dim=1)

    best_idx    = probs.argmax().item()
    best_label  = candidate_labels[best_idx]
    confidence  = probs[0][best_idx].item()

    print(f"[CLIP] Label: '{best_label}' | Confidence: {confidence:.2%}")
    return best_label, confidence


#  BLIP 
def generate_blip_caption(image_path):
    """
    Generate a natural language caption using BLIP.
    Example output: "a ninja jumping over a rooftop at night"

    Returns:
        caption string
    """
    try:
        image  = Image.open(image_path).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=50)

        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        print(f"[BLIP] Caption: '{caption}'")
        return caption

    except Exception as e:
        print(f"[BLIP] Error: {e}")
        return "caption unavailable"


# ══════════════════════════════════════════════════════════════
# SECTION 5 — Combined Function (use this in app.py)
# ══════════════════════════════════════════════════════════════

def process_image(image_path):
    """
    Run CLIP + BLIP together on one image.

    Returns a dictionary with:
        embedding   → 512-dim vector (for FAISS)
        clip_label  → category label from CLIP
        confidence  → CLIP confidence score
        blip_caption→ natural language description from BLIP

    Example return:
        {
            "embedding"    : array([[0.12, -0.45, ...]]),
            "clip_label"   : "ninja fighting",
            "confidence"   : 0.87,
            "blip_caption" : "a ninja jumping over a wall"
        }
    """
    print(f"\n[EmbeddingGenerator] Processing: {image_path}")
    print("─" * 45)

    # CLIP — embedding (for FAISS storage)
    embedding = get_image_embedding(image_path)

    # CLIP — zero-shot label
    clip_label, confidence = get_clip_label(image_path)

    # BLIP — natural language caption
    blip_caption = generate_blip_caption(image_path)

    result = {
        "embedding"    : embedding,
        "clip_label"   : clip_label,
        "confidence"   : confidence,
        "blip_caption" : blip_caption,
    }

    print(f"[EmbeddingGenerator] Done ✓")
    print(f"  CLIP label  : {clip_label} ({confidence:.2%})")
    print(f"  BLIP caption: {blip_caption}")
    print("─" * 45)
    return result


if __name__ == "__main__":
    import os

    test_image = "uploads/user_image.png"

    if os.path.exists(test_image):
        result = process_image(test_image)

        print("\n════ FINAL RESULT ════")
        print(f"  Embedding shape : {result['embedding'].shape}")
        print(f"  CLIP label      : {result['clip_label']}")
        print(f"  CLIP confidence : {result['confidence']:.2%}")
        print(f"  BLIP caption    : {result['blip_caption']}")
    else:
        print(f"\n[!] Place an image at '{test_image}' to test.")
        print("[!] Supported formats: jpg, jpeg, png, bmp, webp")