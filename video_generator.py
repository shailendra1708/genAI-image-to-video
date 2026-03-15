

import cv2
import numpy as np
import os
from moviepy.editor import ImageSequenceClip


OUTPUT_FOLDER = "outputs"
FPS           = 10
FRAME_WIDTH   = 512
FRAME_HEIGHT  = 512

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — Content-Aware Effect Selector (AI decides)
# ══════════════════════════════════════════════════════════════

def choose_effect(clip_label, caption=""):
    """
    Automatically select animation effect using BOTH:
        - CLIP label   → category-level understanding  ("ninja fighting")
        - BLIP caption → sentence-level understanding  ("a ninja running across rooftops")

    Args:
        clip_label : string from CLIP  e.g. "ninja fighting"
        caption    : string from BLIP  e.g. "a ninja running across rooftops"

    Returns:
        effect name string: "zoom" | "pan" | "fade" | "blur"
    """
    text = (clip_label + " " + caption).lower()

    print(f"[AI Effect] Combined text: '{text}'")

    action_keywords = [
        "running", "run", "fight", "fighting", "jumping", "jump",
        "flying", "fly", "attack", "attacking", "chasing", "chase",
        "exploding", "explosion", "racing", "speed", "fast", "battle",
        "warrior", "ninja", "sword", "kick", "punch", "action"
    ]
    if any(word in text for word in action_keywords):
        effect = "blur"
        matched = [w for w in action_keywords if w in text]
        reason  = f"action keywords detected {matched} → motion blur"

    elif any(word in text for word in [
        "space", "underwater", "galaxy", "stars", "night", "dark",
        "ocean", "deep", "fog", "mist", "glowing", "neon", "dream",
        "shadow", "moonlight", "sunset", "sunrise", "silhouette"
    ]):
        effect = "fade"
        reason = "cinematic/dark keywords → fade effect"

    elif any(word in text for word in [
        "city", "skyline", "landscape", "street", "road", "field",
        "forest", "mountain", "valley", "beach", "ocean", "river",
        "village", "town", "horizon", "wide", "aerial", "rooftop",
        "nature", "scenery", "background", "environment"
    ]):
        effect = "pan"
        reason = "wide scene/landscape keywords → pan effect"

    elif any(word in text for word in [
        "portrait", "face", "person", "man", "woman", "girl", "boy",
        "character", "anime", "cartoon", "looking", "staring", "smile",
        "standing", "sitting", "posing", "close", "hero", "fantasy"
    ]):
        effect = "zoom"
        reason = "character/face keywords → zoom effect"

    else:
        effect = "zoom"
        reason = "no keywords matched → zoom (safe default)"

    print(f"[AI Effect] CLIP: '{clip_label}' | BLIP: '{caption}'")
    print(f"[AI Effect] Decision: '{effect}' | Reason: {reason}")
    return effect


# ══════════════════════════════════════════════════════════════
# SECTION 2 — OpenCV: Load and Resize Image
# ══════════════════════════════════════════════════════════════

def load_and_resize(image_path, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """
    Load image using OpenCV and resize to fixed dimensions.

    Why convert BGR to RGB?
        OpenCV loads images in BGR format by default.
        MoviePy expects RGB — wrong colors if not converted.

    Returns:
        numpy array shape (512, 512, 3) in RGB format
    """
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"[OpenCV] Cannot load image: {image_path}")

    img_resized = cv2.resize(img, (width, height))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    print(f"[OpenCV] Loaded: {image_path} | Resized to: {width}x{height}")
    return img_rgb


# ══════════════════════════════════════════════════════════════
# SECTION 3 — OpenCV: Apply Animation Effects to Frames
# ══════════════════════════════════════════════════════════════

def apply_effect(frame, effect="zoom", step=0, total_steps=20):
    """
    Apply animation effect to a single frame.

    Effects:
        zoom  : camera slowly zooms into image (100% → 130%)
        pan   : image slides from left to right
        fade  : image fades in from black
        blur  : starts blurry → becomes sharp

    Args:
        frame       : numpy array (H, W, 3)
        effect      : effect name
        step        : current frame number
        total_steps : total frames in animation

    Returns:
        modified frame as numpy array
    """
    h, w = frame.shape[:2]
    t    = step / total_steps    # progress 0.0 → 1.0

    if effect == "zoom":
        scale  = 1.0 + (0.3 * t)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        zoomed = cv2.resize(frame, (new_w, new_h))
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        return zoomed[y1:y1+h, x1:x1+w]

    elif effect == "pan":
        shift = int(50 * t)
        M     = np.float32([[1, 0, shift], [0, 1, 0]])
        return cv2.warpAffine(frame, M, (w, h))

    elif effect == "fade":
        return (frame * t).astype(np.uint8)

    elif effect == "blur":
        blur_level = max(1, int(15 * (1 - t)))
        if blur_level % 2 == 0:
            blur_level += 1
        return cv2.GaussianBlur(frame, (blur_level, blur_level), 0)

    else:
        return frame


# ══════════════════════════════════════════════════════════════
# SECTION 4 — Generate Multiple Frames from One Image
# ══════════════════════════════════════════════════════════════

def generate_frames_from_image(image_path, effect="zoom", num_frames=20):
    """
    Generate multiple animation frames from a single image.

    Args:
        image_path : path to input image
        effect     : animation effect name
        num_frames : how many frames to generate

    Returns:
        list of numpy arrays (each = one frame)
    """
    base_frame = load_and_resize(image_path)
    frames     = []

    for i in range(num_frames):
        frame = apply_effect(base_frame.copy(), effect=effect, step=i, total_steps=num_frames)
        frames.append(frame)

    print(f"[Frames] Generated {len(frames)} frames | Effect: '{effect}'")
    return frames


# ══════════════════════════════════════════════════════════════
# SECTION 5 — Generate Transition Frames Between Two Images
# ══════════════════════════════════════════════════════════════

def generate_transition_frames(image_path_1, image_path_2, num_frames=15):
    """
    Create smooth blend transition between two images.

    Frame 1  : 100% image1 +   0% image2
    Frame 8  :  50% image1 +  50% image2
    Frame 15 :   0% image1 + 100% image2

    Returns:
        list of blended numpy arrays
    """
    frame1 = load_and_resize(image_path_1)
    frame2 = load_and_resize(image_path_2)
    frames = []

    for i in range(num_frames):
        alpha   = i / num_frames
        blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        frames.append(blended)

    print(f"[Frames] Generated {len(frames)} transition frames.")
    return frames


# ══════════════════════════════════════════════════════════════
# SECTION 6 — MoviePy: Create Video from Frames
# ══════════════════════════════════════════════════════════════

def create_video(frames, output_filename="animation.mp4", fps=FPS):
    """
    Combine list of frames into an MP4 video using MoviePy.

    Args:
        frames          : list of numpy arrays (H, W, 3) in RGB
        output_filename : output video filename
        fps             : frames per second

    Returns:
        full path to saved video file
    """
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    print(f"[MoviePy] Creating video | Frames: {len(frames)} | FPS: {fps} | Output: {output_path}")

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)

    print(f"[MoviePy] Video saved → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
# SECTION 7 — Create Slideshow Video (Multiple Images)
# ══════════════════════════════════════════════════════════════

def create_slideshow_video(image_paths, seconds_per_image=2, transition_frames=15, output_filename="slideshow.mp4"):
    """
    Create slideshow video from multiple images with transitions.

    Structure:
        [Image 1 held] → [transition] → [Image 2 held] → [transition] → ...

    Returns:
        full path to saved video
    """
    all_frames  = []
    hold_frames = seconds_per_image * FPS

    print(f"[Slideshow] Building from {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        base_frame = load_and_resize(img_path)

        for _ in range(hold_frames):
            all_frames.append(base_frame.copy())

        if i < len(image_paths) - 1:
            transition = generate_transition_frames(img_path, image_paths[i + 1], transition_frames)
            all_frames.extend(transition)

    print(f"[Slideshow] Total frames: {len(all_frames)}")
    return create_video(all_frames, output_filename=output_filename)


# ══════════════════════════════════════════════════════════════
# Quick Test — python video_generator.py
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    upload_dir   = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_files  = [
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    if not image_files:
        print("\n[Test] No images in uploads/ — add images and retry.")
    elif len(image_files) == 1:
        frames     = generate_frames_from_image(image_files[0], effect="zoom", num_frames=30)
        video_path = create_video(frames, output_filename="test_animation.mp4")
        print(f"\n[Test] Video saved → {video_path}")
    else:
        video_path = create_slideshow_video(image_files[:3], output_filename="test_slideshow.mp4")
        print(f"\n[Test] Slideshow saved → {video_path}")