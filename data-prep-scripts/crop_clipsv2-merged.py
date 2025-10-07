#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========= CONFIG =========
DATASET_DIR   = "/home/spactureai/xalil/face-bodymatch/output/face"   # *.json
VIDEO_DIR     = "/home/spactureai/xalil/face-bodymatch/input/videos-face"                # *.mp4
OUTPUT_DIR    = "/home/spactureai/xalil/face-bodymatch/output/face"     # cropped clips

# Filters
MIN_BOX_SIZE  = 124    # average box height threshold (px)
MIN_FRAMES    = 20     # minimum frames per track
MIN_DURATION  = 1.0    # minimum seconds per track

# Cropping
PADDING       = 0
CLAMP_BORDER  = True

# Chunked frame loading
CHUNK_SIZE    = 1000   # frames per batch

# Parallelism
N_WORKERS     = 1      # processes (videos in parallel)
# ==========================


def process_one_json(json_path: str):
    """Process one detection JSON and its paired MP4 into per-track cropped clips."""
    stem = os.path.splitext(os.path.basename(json_path))[0]      # e.g. "253111_merged_detections"
    video_stem = stem.replace("_detections", "")                 # -> "253111_merged"
    video_path = os.path.join(VIDEO_DIR, f"{video_stem}.mp4")

    if not os.path.exists(video_path):
        print(f"[WARN] Missing video for {json_path}")
        return

    # Load detections
    with open(json_path, "r") as f:
        data = json.load(f)
    detections = data.get("detections", [])
    fps = float(data.get("fps", 25.0))

    # Group detections by frame
    frame_dets = defaultdict(list)
    track_boxes = defaultdict(list)  # for avg size & filters
    for det in detections:
        tid = int(det["id"])
        x1, y1, x2, y2 = map(int, det["bbox"])
        frame_idx = int(det["frame_idx"])
        w, h = x2 - x1, y2 - y1
        frame_dets[frame_idx].append((tid, x1, y1, x2, y2))
        track_boxes[tid].append((w, h))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 25.0

    # Output folder: /OUTPUT_DIR/{video_stem}/
    out_dir = os.path.join(OUTPUT_DIR, video_stem)
    os.makedirs(out_dir, exist_ok=True)

    # Writers per track
    writers = {}
    track_counts = defaultdict(int)

    frame_idx = 0
    while True:
        frames = {}
        for _ in range(CHUNK_SIZE):
            ok, fr = cap.read()
            if not ok:
                break
            frames[frame_idx] = fr
            frame_idx += 1

        if not frames:
            break

        for fidx, frame in frames.items():
            if fidx not in frame_dets:
                continue
            for tid, x1, y1, x2, y2 in frame_dets[fidx]:
                # padding
                if PADDING > 0:
                    x1p, y1p = x1 - PADDING, y1 - PADDING
                    x2p, y2p = x2 + PADDING, y2 + PADDING
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

                crop = frame[y1p:y2p, x1p:x2p]
                if crop is None or crop.size == 0:
                    continue

                # init writer lazily
                if tid not in writers:
                    mean_w = int(np.mean([max(1, w + 2*PADDING) for w, h in track_boxes[tid]]))
                    mean_h = int(np.mean([max(1, h + 2*PADDING) for w, h in track_boxes[tid]]))
                    size_wh = (max(1, mean_w), max(1, mean_h))

                    out_path = os.path.join(out_dir, f"{video_stem}_id{tid}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writers[tid] = (cv2.VideoWriter(out_path, fourcc, vid_fps, size_wh), size_wh)

                writer, size_wh = writers[tid]
                crop_resized = cv2.resize(crop, size_wh)
                writer.write(crop_resized)
                track_counts[tid] += 1

        frames.clear()

    cap.release()

    # close writers and filter short/noisy tracks
    for tid, (writer, _) in writers.items():
        writer.release()
        n_frames = track_counts[tid]
        duration = n_frames / vid_fps
        avg_h = np.mean([h for _, h in track_boxes[tid]]) if track_boxes[tid] else 0
        if n_frames < MIN_FRAMES or duration < MIN_DURATION or avg_h < MIN_BOX_SIZE:
            out_path = os.path.join(out_dir, f"{video_stem}_id{tid}.mp4")
            if os.path.exists(out_path):
                os.remove(out_path)
                print(f"[FILTER] Removed short/noisy track {tid} ({n_frames} frames, {duration:.1f}s, avg_h={avg_h:.1f})")

    print(f"[OK] {json_path} → {out_dir}")


def process_all():
    json_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".json")]
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