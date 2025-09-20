#!/usr/bin/env python3
"""
Extract one embedding per video with DINOv3 (configurable backbone),
recursively, and save embeddings mirroring the input folder structure.

Example:
  INPUT_DIR/areaA/clip1.mp4 -> OUTPUT_DIR/areaA/clip1.pt
"""

import os
import torch
from torch import nn
import torchvision

# ========= Configuration =========
INPUT_DIR   = "./all_videos_cropped_v2"
OUTPUT_DIR  = "./embedding_v2"

# Choose: "dinov3_convnext_small", "dinov3_convnext_base", "dinov3_convnext_large"
MODEL_NAME  = "dinov3_convnext_large"
WEIGHTS     = "/home/spactureai/xalil/dinov3/models/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"

FRAME_STRIDE = 2       # take 1 of every N frames
RESIZE       = 256     # input size for backbone
BATCH_SIZE   = 64      # frames per forward pass

SUPPORTED_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
# =================================


def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("[WARN] CUDA not available, running on CPU.")
    else:
        torch.backends.cudnn.benchmark = True
    return dev


def list_videos_recursive(input_dir: str) -> list[str]:
    vids = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                vids.append(os.path.join(root, f))
    vids.sort()
    return vids


@torch.inference_mode()
def load_model(model_name: str, weights: str, device: torch.device) -> nn.Module:
    model = torch.hub.load(
        ".",
        model_name,
        source="local",
        weights=weights,
        pretrained=True,
    )
    model.eval().to(device)
    return model


def read_video_frames(path: str) -> torch.Tensor:
    """Return frames as [T, H, W, C] uint8 (CPU)."""
    try:
        video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
        return video
    except Exception:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        ok = cap.isOpened()
        while ok:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
        cap.release()
        if not frames:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8)
        return torch.stack(frames, dim=0)
    except Exception as e:
        raise RuntimeError(f"Could not read video {path}: {e}")


def preprocess_frames(frames: torch.Tensor, size: int = RESIZE) -> torch.Tensor:
    """[T,H,W,C] -> [T,3,size,size], float32 normalized (imagenet)."""
    if frames.numel() == 0:
        return frames
    frames = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0
    frames = nn.functional.interpolate(
        frames, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=frames.dtype, device=frames.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=frames.dtype, device=frames.device)[:, None, None]
    return (frames - mean) / std


def batched_iter(frames: torch.Tensor, n: int):
    """Yield [B,3,H,W] batches without extra deps."""
    t = frames.shape[0]
    for i in range(0, t, n):
        yield frames[i:i+n]


@torch.inference_mode()
def embed_video(model: nn.Module, frames: torch.Tensor, device: torch.device, batch_size: int = BATCH_SIZE):
    if frames.numel() == 0 or frames.shape[0] == 0:
        return torch.zeros(getattr(model, "embed_dim", 0), device="cpu"), 0

    #TODO
    
    n_frames = frames.shape[0]
    sum_feat = None
    use_autocast = (device.type == "cuda")

    for batch in batched_iter(frames, batch_size):
        batch = batch.to(device, non_blocking=True)
        if use_autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model(batch)   # [B, D]
        else:
            feats = model(batch)
        feats = feats.float()
        s = feats.sum(dim=0)
        sum_feat = s if sum_feat is None else (sum_feat + s)

    emb = (sum_feat / float(n_frames)).detach().cpu().contiguous()
    return emb, n_frames


def ensure_out_dir_for(video_path: str) -> str:
    """Mirror the input folder structure in OUTPUT_DIR."""
    rel_path = os.path.relpath(video_path, INPUT_DIR)               # e.g. areaA/clip1.mp4
    rel_dir = os.path.dirname(rel_path)                             # e.g. areaA
    out_dir = os.path.join(OUTPUT_DIR, rel_dir)                     # e.g. OUTPUT_DIR/areaA
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    videos = list_videos_recursive(INPUT_DIR)
    if not videos:
        print(f"No videos in {INPUT_DIR} (extensions: {sorted(SUPPORTED_EXTS)})")
        return

    print(f"Loading model {MODEL_NAME} on {device} ...")
    model = load_model(MODEL_NAME, WEIGHTS, device)
    print("Model loaded.")

    for vid_path in videos:
        out_dir = ensure_out_dir_for(vid_path)
        stem = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(out_dir, stem + ".pt")
        if os.path.exists(out_path):
            print(f"[skip] {vid_path} -> {out_path} (exists)")
            continue

        try:
            video = read_video_frames(vid_path)  # [T,H,W,C], uint8
        except Exception as e:
            print(f"[error] read {vid_path}: {e}")
            continue

        if video.numel() == 0 or video.shape[0] == 0:
            print(f"[warn] Empty video: {vid_path}")
            continue

        frames = video[:: max(1, FRAME_STRIDE), ...]
        frames = preprocess_frames(frames, size=RESIZE)

        emb, n = embed_video(model, frames, device=device, batch_size=BATCH_SIZE)

        torch.save({
            "video": vid_path,
            "n_frames_used": int(n),
            "frame_stride": int(FRAME_STRIDE),
            "model_name": MODEL_NAME,
            "resize": int(RESIZE),
            "embedding": emb,   # [D]
        }, out_path)

        print(f"[ok] {vid_path} -> {out_path} (frames={n}, dim={emb.shape[-1]})")


if __name__ == "__main__":
    main()