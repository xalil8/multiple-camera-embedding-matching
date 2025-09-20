#!/usr/bin/env python3
import os
import cv2
import math
import subprocess
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, deque

# ====== CONFIG ======
MODEL_PATH = "/home/spactureai/xalil/dinov3/notebooks/spacture_yolo_model.pt"
INPUT_DIR  = "/home/spactureai/xalil/camera_matching_videos"
OUTPUT_DIR = "/home/spactureai/xalil/dinov3/all_videos_cropped_v2"

PADDING = 0
MIN_DETECTIONS_PER_ID = 15
MIN_BOX_SIZE = 224
CONF_TH = 0.5
IOU_TH  = 0.45
MAX_SECONDS = 5           # max video length in seconds
CONVERT_TO_H264 = True    # toggle ffmpeg conversion

# Smoothing config
SMOOTHING_WINDOW = 5      # frames for moving average
EMA_ALPHA = 0.5           # exponential smoothing factor
USE_EMA = False           # switch between mean vs exponential
# =====================

# storage for smoothing
box_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
smoothed_boxes = {}  # tid -> last smoothed box


def smooth_box(tid, box):
    """
    Smooth bounding box coordinates for stability.
    box = (x1, y1, x2, y2)
    """
    if USE_EMA:
        if tid not in smoothed_boxes:
            smoothed_boxes[tid] = np.array(box, dtype=np.float32)
        else:
            smoothed_boxes[tid] = (
                EMA_ALPHA * np.array(box, dtype=np.float32)
                + (1 - EMA_ALPHA) * smoothed_boxes[tid]
            )
        return smoothed_boxes[tid].astype(int)
    else:
        box_history[tid].append(np.array(box, dtype=np.float32))
        avg_box = np.mean(box_history[tid], axis=0)
        return avg_box.astype(int)


def ensure_fps_size(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if math.isclose(fps, 0.0):
        fps = 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, (w, h)


def convert_to_h264(input_path: str, out_path: str):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out_path]
    subprocess.run(cmd, check=True)


def track_video(video_path: str, model: YOLO, out_dir: Path, conf: float = CONF_TH, iou: float = IOU_TH):
    """
    Tracks humans with YOLOv8 + ByteTrack.
    For each ID:
      - Collect crops with smoothed bounding boxes
      - Resize crops to average (w,h)
      - Save each ID's video inside its subfolder
    """
    vpath = Path(video_path)
    fps, (W, H) = ensure_fps_size(str(vpath))

    # storage
    crops_per_id = defaultdict(list)
    sizes_per_id = defaultdict(list)

    results_gen = model.track(
        source=str(vpath),
        stream=True,
        conf=conf,
        iou=iou,
        classes=[0],
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        half=True,
        device="cuda",
        imgsz=640,
    )

    for r in results_gen:
        frame = r.orig_img
        if frame is None or r.boxes is None or r.boxes.id is None:
            continue

        ids  = r.boxes.id.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()

        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            x1 = max(0, int(x1) - PADDING)
            y1 = max(0, int(y1) - PADDING)
            x2 = min(W - 1, int(x2) + PADDING)
            y2 = min(H - 1, int(y2) + PADDING)

            # apply smoothing
            x1, y1, x2, y2 = smooth_box(tid, (x1, y1, x2, y2))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            h, w = crop.shape[:2]
            crops_per_id[tid].append(crop)
            sizes_per_id[tid].append((w, h))

    # output directory preserving structure
    rel_path = vpath.relative_to(INPUT_DIR)
    out_subdir = out_dir / rel_path.parent
    out_subdir.mkdir(parents=True, exist_ok=True)

    for tid, crops in crops_per_id.items():
        if len(crops) < MIN_DETECTIONS_PER_ID:
            print(f"[SKIP] {vpath.name} ID {tid}: only {len(crops)} detections")
            continue

        arr = np.array(sizes_per_id[tid])
        avg_w, avg_h = np.mean(arr, axis=0).astype(int)

        if avg_w < MIN_BOX_SIZE or avg_h < MIN_BOX_SIZE:
            print(f"[SKIP] {vpath.name} ID {tid}: avg size {avg_w}x{avg_h} < {MIN_BOX_SIZE}")
            continue

        stem = vpath.stem
        temp_out = out_subdir / f"{stem}_id{tid}_temp.mp4"
        final_out = out_subdir / f"{stem}_id{tid}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_out), fourcc, fps, (avg_w, avg_h))

        max_frames = int(fps * MAX_SECONDS) if MAX_SECONDS > 0 else None

        for i, crop in enumerate(crops):
            if max_frames is not None and i >= max_frames:
                break
            resized = cv2.resize(crop, (avg_w, avg_h))
            writer.write(resized)

        writer.release()

        if CONVERT_TO_H264:
            try:
                convert_to_h264(str(temp_out), str(final_out))
            finally:
                if temp_out.exists():
                    temp_out.unlink()
        else:
            try:
                temp_out.rename(final_out)
            except Exception as e:
                print(f"[ERROR] Renaming temp file failed: {e}")
            finally:
                if temp_out.exists():
                    temp_out.unlink()

        print(f"[OK] {vpath.name} -> ID {tid}: {final_out}")


def process_folder(input_dir: str, output_dir: str):
    in_dir  = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    model = YOLO(MODEL_PATH)

    video_files = []
    for ext in ["*.mp4", "*.MP4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
        video_files.extend(in_dir.rglob(ext))

    if not video_files:
        print(f"No videos found in {in_dir}")
        return

    for vp in video_files:
        print(f"\n=== Processing {vp} ===")
        try:
            track_video(str(vp), model, out_dir)
        except Exception as e:
            print(f"[ERROR] {vp.name}: {e}")


if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR)
    