#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ========= CONFIG =========
DATASET_DIR   = "/home/spactureai/xalil/embedding-matching/datasets/combined_dataset"  # *.json & *.mp4 live side-by-side
OUTPUT_DIR    = "/home/spactureai/xalil/embedding-matching/datasets/cropped_tracks"    # output root

# Filters
MIN_BOX_SIZE  = 224    # average box height threshold (pixels)
MIN_FRAMES    = 40     # minimum frames per track
MIN_DURATION  = 2.0    # minimum seconds per track

# Cropping
PADDING       = 0      # optional extra pixels around boxes; >=0
CLAMP_BORDER  = False   # clamp padded boxes to frame boundaries

# Parallelism
N_WORKERS      = 2     # processes (videos in parallel)
WRITER_THREADS = 4    # threads pe r video (tracks in parallel)
# ==========================

def write_clip(crops, fps, out_path, size_wh):
    """Write list of cropped frames to a video file."""
    if not crops:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size_wh)
    for img in crops:
        if (img.shape[1], img.shape[0]) != size_wh:
            img = cv2.resize(img, size_wh)
        writer.write(img)
    writer.release()

def process_one_json(json_path: str):
    """Make per-track cropped clips from one detection JSON + its paired MP4."""
    video_path = json_path.replace(".json", ".mp4")
    if not os.path.exists(video_path):
        print(f"[WARN] Missing video for {json_path}")
        return

    # Load detections
    with open(json_path, "r") as f:
        data = json.load(f)
    detections = data.get("detections", [])
    json_fps = float(data.get("fps", 25.0))

    # Read actual video & fps
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or json_fps or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Preload frames (simple + fast for short/medium clips)
    frames = {}
    fidx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames[fidx] = fr
        fidx += 1
    cap.release()

    # Group boxes by track id
    tracks = defaultdict(list)
    for det in detections:
        tid = int(det["id"])
        x1, y1, x2, y2 = det["bbox"]
        # ints for indexing
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        h = y2 - y1
        tracks[tid].append((int(det["frame_idx"]), x1, y1, x2, y2, w, h))

    # Output foldering:
    # /OUTPUT_DIR/{camera}/{stem}/id{tid}.mp4
    camera = os.path.basename(os.path.dirname(json_path))             # parent folder name (camera)
    stem   = os.path.splitext(os.path.basename(json_path))[0]         # e.g. "253111_2025-08-15_09-00-12-215337"
    out_dir = os.path.join(OUTPUT_DIR, camera, stem)
    os.makedirs(out_dir, exist_ok=True)

    # Threaded writing per track
    tasks = []
    with ThreadPoolExecutor(max_workers=WRITER_THREADS) as pool:
        for tid, boxes in tracks.items():
            # sort by frame_idx to preserve time order
            boxes.sort(key=lambda t: t[0])

            # avg height filter
            avg_h = float(np.mean([b[6] for b in boxes])) if boxes else 0.0
            if avg_h < MIN_BOX_SIZE:
                continue

            # collect crops
            crops = []
            for fidx, x1, y1, x2, y2, bw, bh in boxes:
                if fidx not in frames:
                    continue
                # padding
                if PADDING > 0:
                    x1p = x1 - PADDING; y1p = y1 - PADDING
                    x2p = x2 + PADDING; y2p = y2 + PADDING
                else:
                    x1p, y1p, x2p, y2p = x1, y1, x2, y2

                # clamp
                if CLAMP_BORDER:
                    x1p = max(0, min(x1p, W - 1))
                    y1p = max(0, min(y1p, H - 1))
                    x2p = max(0, min(x2p, W - 1))
                    y2p = max(0, min(y2p, H - 1))
                    if x2p <= x1p or y2p <= y1p:
                        continue

                crop = frames[fidx][y1p:y2p, x1p:x2p]
                if crop is None or crop.size == 0:
                    continue
                crops.append(crop)

            # noise filters
            if len(crops) < MIN_FRAMES:
                continue
            if (len(crops) / vid_fps) < MIN_DURATION:
                continue

            # choose a stable output size from boxes (faster than computing from images)
            mean_w = int(np.mean([max(1, b[5] + 2*PADDING) for b in boxes]))
            mean_h = int(np.mean([max(1, b[6] + 2*PADDING) for b in boxes]))
            size_wh = (max(1, mean_w), max(1, mean_h))

            #out_path = os.path.join(out_dir, f"id{tid}.mp4")  # no camera/stem duplication
            out_path = os.path.join(out_dir, f"{stem}_id{tid}.mp4")
            tasks.append(pool.submit(write_clip, crops, vid_fps, out_path, size_wh))

        # finalize
        for fut in as_completed(tasks):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] clip writing failed: {e}")

    print(f"[OK] {json_path} → {out_dir}")

def process_all():
    json_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    if not json_files:
        print(f"No JSONs found in {DATASET_DIR}")
        return

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(process_one_json, jp) for jp in json_files]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] worker failed: {e}")

if __name__ == "__main__":
    process_all()