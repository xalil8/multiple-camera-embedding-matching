#!/usr/bin/env python3
"""
Extract one embedding per IMAGE (not video) with DINOv3 (configurable backbone),
recursively, and save embeddings mirroring the input folder structure.

Example:
  INPUT_DIR/camA/vid1/id123/frame_000123.jpg -> OUTPUT_DIR/camA/vid1/id123/frame_000123.pt
"""

import os
import torch
from torch import nn
import torchvision

# ========= Configuration =========
INPUT_DIR   = "./all_videos_cropped_v3_images"   # root of your frames
OUTPUT_DIR  = "./embedding_v3_images"            # mirror structure here
#!/usr/bin/env python3

# Choose: "dinov3_convnext_small", "dinov3_convnext_base", "dinov3_convnext_large"
MODEL_NAME  = "dinov3_convnext_large"
WEIGHTS     = "/home/spactureai/xalil/dinov3/models/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"

RESIZE      = 256    # model input size (H=W=RESIZE)
BATCH_SIZE  = 128    # images per forward pass (bump if VRAM allows)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# =================================


def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARN] CUDA not available, running on CPU.")
    return dev


def list_images_recursive(input_dir: str) -> list[str]:
    ims = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                ims.append(os.path.join(root, f))
    ims.sort()
    return ims


@torch.inference_mode()
def load_model(model_name: str, weights: str, device: torch.device) -> nn.Module:
    model = torch.hub.load(
        ".", model_name, source="local",
        weights=weights, pretrained=True
    )
    model.eval().to(device)
    return model


def read_image_tensor(path: str) -> torch.Tensor:
    """
    Read image -> [3,H,W] uint8 RGB tensor on CPU.
    Falls back to OpenCV if torchvision fails.
    """
    try:
        img = torchvision.io.read_image(path)  # [C,H,W], uint8, RGB
        if img.ndim == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3, ...]
        return img
    except Exception:
        pass

    try:
        import cv2
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2.imread returned None")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # [3,H,W], uint8
        return t
    except Exception as e:
        raise RuntimeError(f"Could not read image {path}: {e}")


def ensure_out_dir_for(image_path: str) -> str:
    """Mirror the input folder structure in OUTPUT_DIR for this image."""
    rel_path = os.path.relpath(image_path, INPUT_DIR)  # e.g. camA/vid1/id123/frame_000123.jpg
    rel_dir  = os.path.dirname(rel_path)               # e.g. camA/vid1/id123
    out_dir  = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def batched_paths(paths: list[str], n: int):
    for i in range(0, len(paths), n):
        yield paths[i:i+n]


@torch.inference_mode()
def forward_batch(model: nn.Module, batch_cpu_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    batch_cpu_u8: [B,3,RESIZE,RESIZE] uint8 on CPU
    Returns features [B,D] on CPU.
    """
    x = batch_cpu_u8.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0

    # Imagenet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    x = (x - mean) / std

    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            feats = model(x)  # [B, D]
    else:
        feats = model(x)

    return feats.float().detach().cpu().contiguous()


def main():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    images = list_images_recursive(INPUT_DIR)
    if not images:
        print(f"No images in {INPUT_DIR} (extensions: {sorted(SUPPORTED_EXTS)})")
        return

    print(f"Loading model {MODEL_NAME} on {device} ...")
    model = load_model(MODEL_NAME, WEIGHTS, device)
    print("Model loaded. Images:", len(images))
    printed_dim = False

    for paths in batched_paths(images, BATCH_SIZE):
        batch_list = []
        keep_paths = []

        for p in paths:
            out_dir = ensure_out_dir_for(p)
            stem = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(out_dir, stem + ".pt")
            if os.path.exists(out_path):
                continue
            try:
                t = read_image_tensor(p)  # [3,H,W], uint8
                if t.numel() == 0:
                    continue
                # --- resize here so all tensors have identical shapes before stack ---
                t = nn.functional.interpolate(
                    t.unsqueeze(0).float(), size=(RESIZE, RESIZE),
                    mode="bilinear", align_corners=False, antialias=True
                ).squeeze(0).to(dtype=torch.uint8)  # -> [3,RESIZE,RESIZE] uint8
                batch_list.append(t)
                keep_paths.append((p, out_path))
            except Exception as e:
                print(f"[error] read {p}: {e}")

        if not batch_list:
            continue

        batch = torch.stack(batch_list, dim=0)  # [B,3,RESIZE,RESIZE] uint8
        feats = forward_batch(model, batch, device)  # [B,D] on CPU

        # Print embed dim once
        if not printed_dim:
            print("embed_dim detected:", feats.shape[-1])
            printed_dim = True

        for i, (src_path, out_path) in enumerate(keep_paths):
            emb = feats[i]  # [D]
            meta = {
                "image": src_path,
                "model_name": MODEL_NAME,
                "resize": int(RESIZE),
                "embedding": emb,  # [D], float32
            }
            try:
                torch.save(meta, out_path)
            except Exception as e:
                print(f"[error] save {out_path}: {e}")

    print("[OK] Done.")


if __name__ == "__main__":
    main()