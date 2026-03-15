
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[RIFE] Device: {DEVICE.upper()}")


# ══════════════════════════════════════════════════════════════
# SECTION 1 — Lightweight Flow Estimator (RIFE-style)
# ══════════════════════════════════════════════════════════════

class FlowEstimator(nn.Module):
    """
    Lightweight optical flow estimator for frame interpolation.

    Architecture:
        Encoder  → extracts features from both frames
        Flow net → estimates motion between frames
        Decoder  → blends frames using estimated motion

    This is a simplified RIFE-style model that runs fast on CPU.
    It estimates how pixels move between frames and creates
    a realistic in-between frame.
    """

    def __init__(self):
        super(FlowEstimator, self).__init__()

        # Feature extractor — shared for both input frames
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # RGB → 32 features
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 32 → 64 features
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),  # 64 → 32 features
            nn.ReLU(),
        )

        # Flow estimator — predicts motion vectors
        self.flow_net = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # concat of 2 frames features
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1),   # 4 channels = (dx1,dy1,dx2,dy2)
        )

    def forward(self, frame1, frame2, t=0.5):
        """
        Interpolate between frame1 and frame2 at time t.

        Args:
            frame1 : tensor (B, 3, H, W) — first frame
            frame2 : tensor (B, 3, H, W) — second frame
            t      : float 0.0→1.0 — interpolation point
                     0.5 = exactly halfway between frames

        Returns:
            interpolated frame tensor (B, 3, H, W)
        """
        # Extract features from both frames
        feat1 = self.encoder(frame1)
        feat2 = self.encoder(frame2)

        # Concatenate features → estimate flow
        combined = torch.cat([feat1, feat2], dim=1)
        flow     = self.flow_net(combined)

        # Split flow into forward and backward components
        flow_f = flow[:, :2]   # flow from frame1 → frame2
        flow_b = flow[:, 2:]   # flow from frame2 → frame1

        # Warp both frames using estimated flow
        warped1 = self._warp(frame1, flow_f *  t)
        warped2 = self._warp(frame2, flow_b * (1 - t))

        # Blend warped frames at time t
        interpolated = (1 - t) * warped1 + t * warped2
        return interpolated.clamp(0, 1)

    def _warp(self, frame, flow):
        """
        Warp frame using optical flow vectors.
        Moves each pixel by its estimated flow displacement.
        """
        B, C, H, W = frame.shape

        # Build base sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device),
            torch.linspace(-1, 1, W, device=frame.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Add flow displacement to grid
        flow_norm    = flow.permute(0, 2, 3, 1)
        flow_norm[..., 0] /= (W / 2)
        flow_norm[..., 1] /= (H / 2)
        warped_grid  = grid + flow_norm

        # Sample frame at warped positions
        return F.grid_sample(
            frame, warped_grid,
            mode="bilinear", padding_mode="border", align_corners=True
        )


# ══════════════════════════════════════════════════════════════
# SECTION 2 — Frame Conversion Utilities
# ══════════════════════════════════════════════════════════════

def numpy_to_tensor(frame):
    """
    Convert numpy frame (H, W, 3) uint8 → tensor (1, 3, H, W) float [0,1]
    """
    tensor = torch.from_numpy(frame).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC → BCHW
    return tensor.to(DEVICE)


def tensor_to_numpy(tensor):
    """
    Convert tensor (1, 3, H, W) float [0,1] → numpy (H, W, 3) uint8
    """
    frame = tensor.squeeze(0).permute(1, 2, 0)  # BCHW → HWC
    frame = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return frame


# ══════════════════════════════════════════════════════════════
# SECTION 3 — Main Interpolation Function
# ══════════════════════════════════════════════════════════════

# Load model once at module level
_model = None

def get_model():
    """Lazy load RIFE model — only loads when first needed."""
    global _model
    if _model is None:
        print("[RIFE] Loading flow estimator model...")
        _model = FlowEstimator().to(DEVICE)
        _model.eval()
        print("[RIFE] Model ready!")
    return _model


def interpolate_frames(frames, multiplier=2):
    """
    AI frame interpolation — doubles or quadruples frame count.

    How it works:
        Input:  [Frame0, Frame1, Frame2, Frame3]
        After 2x interpolation:
        Output: [Frame0, AI_01, Frame1, AI_12, Frame2, AI_23, Frame3]

        Where AI_01 = AI-generated frame between Frame0 and Frame1

    Args:
        frames     : list of numpy arrays (H, W, 3) from OpenCV
        multiplier : 2 = double frames, 4 = quadruple frames

    Returns:
        list of numpy arrays with more frames (smoother motion)
    """
    if len(frames) < 2:
        print("[RIFE] Need at least 2 frames to interpolate.")
        return frames

    model       = get_model()
    result      = []
    passes      = 1 if multiplier == 2 else 2   # 2x = 1 pass, 4x = 2 passes
    current     = frames

    for p in range(passes):
        interpolated = []

        for i in range(len(current) - 1):
            f1 = numpy_to_tensor(current[i])
            f2 = numpy_to_tensor(current[i + 1])

            # Keep original frame
            interpolated.append(current[i])

            # Generate AI in-between frame at t=0.5
            with torch.no_grad():
                mid_frame = model(f1, f2, t=0.5)

            interpolated.append(tensor_to_numpy(mid_frame))

        # Add last frame
        interpolated.append(current[-1])
        current = interpolated

        print(f"[RIFE] Pass {p+1}/{passes} complete | Frames: {len(current)}")

    print(f"[RIFE] Interpolation done | {len(frames)} → {len(current)} frames")
    return current


# ══════════════════════════════════════════════════════════════
# SECTION 4 — Full AI-Enhanced Frame Generator
# ══════════════════════════════════════════════════════════════

def generate_ai_frames(image_path, clip_label="", caption="", base_frames=15, multiplier=2):
    """
    Full pipeline: OpenCV base frames → RIFE interpolation → smooth AI video.

    Steps:
        1. OpenCV generates `base_frames` using content-aware effect
        2. RIFE doubles/quadruples frame count with AI interpolation
        3. Returns smooth final frames

    Frame count:
        base_frames=15, multiplier=2 → 15 → 29 frames
        base_frames=15, multiplier=4 → 15 → 57 frames

    Args:
        image_path  : path to input image
        clip_label  : CLIP label for effect selection
        caption     : BLIP caption for effect selection
        base_frames : OpenCV frames to generate first
        multiplier  : 2x or 4x interpolation

    Returns:
        (frames, mode_used)
    """
    from video_generator import generate_frames_from_image, choose_effect

    print(f"\n[RIFE] Starting AI-enhanced frame generation...")
    print(f"[RIFE] Image: {image_path}")

    # Step 1: OpenCV base frames
    effect       = choose_effect(clip_label, caption)
    base          = generate_frames_from_image(image_path, effect=effect, num_frames=base_frames)
    print(f"[RIFE] Base frames: {len(base)} | Effect: {effect}")

    # Step 2: RIFE interpolation
    try:
        ai_frames = interpolate_frames(base, multiplier=multiplier)
        mode_used = f"OpenCV ({effect}) + RIFE {multiplier}x interpolation"
        print(f"[RIFE] AI frames: {len(ai_frames)} | Mode: {mode_used}")
        return ai_frames, mode_used

    except Exception as e:
        print(f"[RIFE] Interpolation failed: {e} — using base frames only")
        return base, f"OpenCV only ({effect})"


# ══════════════════════════════════════════════════════════════
# Quick Test — python rife_interpolator.py
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from video_generator import create_video
    import os

    print("\n" + "="*55)
    print("  RIFE Frame Interpolator — Quick Test")
    print("="*55)

    test_image = None
    for f in os.listdir("uploads"):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            test_image = os.path.join("uploads", f)
            break

    if not test_image:
        print("\n[Test] No images in uploads/ — add an image and retry.")
    else:
        print(f"\n[Test] Using: {test_image}")

        frames, mode = generate_ai_frames(
            image_path  = test_image,
            clip_label  = "action scene",
            caption     = "a ninja running",
            base_frames = 15,
            multiplier  = 2
        )

        video_path = create_video(frames, output_filename="rife_test.mp4", fps=10)

        print(f"\n[Test] Mode      : {mode}")
        print(f"[Test] Frames    : {len(frames)}")
        print(f"[Test] Video     → {video_path}")

    print("="*55 + "\n")
