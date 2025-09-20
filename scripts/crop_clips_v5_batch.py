#!/usr/bin/env python3
import os
import cv2
import math
import subprocess
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ========= CONFIG =========
MODEL_PATH = "/home/spactureai/xalil/dinov3/notebooks/custom_yolo_model.pt"
INPUT_DIR  = "/home/spactureai/xalil/camera_matching_videos"   # scans recursively
OUTPUT_DIR = "/home/spactureai/xalil/dinov3/all_videos_cropped_v2"

# Detection / tracking
CONF_TH = 0.5
IOU_TH  = 0.45
BATCH_SIZE = 680                # frames per YOLO track() call in a batch

# Per-ID filters
PADDING = 0
MIN_DETECTIONS_PER_ID = 50      # need at least this many detections
MIN_CLIP_FRAMES = 75            # need at least this many frames
MIN_BOX_SIZE = 224              # pass if (w >= 224) OR (h >= 224)
MAX_SECONDS = 5                 # cap each ID clip length (0 = unlimited)

# Output encoding
CONVERT_TO_H264 = False          # if False, keep mp4v temp
FFMPEG_PIX_FMT = "yuv420p"

# Smoothing
SMOOTHING_WINDOW = 5            # moving-average window
USE_EMA = False                 # False=MA, True=EMA
EMA_ALPHA = 0.5                 # only if USE_EMA

# Parallelism
N_YOLO_WORKERS = 2              # process-level parallel YOLO for different videos
WRITER_THREADS = 16              # per-process threads to write multiple clips concurrently
DEVICE = "0"                    # YOLO device per process (e.g., "0", "cuda:0", or just "cuda")
# ==========================


def ensure_fps_size(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if math.isclose(fps, 0.0):
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, (w, h)


def convert_to_h264(input_path, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264", "-pix_fmt", FFMPEG_PIX_FMT, "-an", out_path],
        check=True
    )


def make_out_subdir(video_path):
    rel_path = os.path.relpath(video_path, INPUT_DIR)
    rel_dir = os.path.dirname(rel_path)
    out_subdir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(out_subdir, exist_ok=True)
    return out_subdir


def new_smoothers():
    box_hist = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
    ema_state = {}
    return box_hist, ema_state


def smooth_box(tid, box, box_hist, ema_state):
    if USE_EMA:
        if tid not in ema_state:
            ema_state[tid] = np.array(box, dtype=np.float32)
        else:
            ema_state[tid] = EMA_ALPHA * np.array(box, dtype=np.float32) + (1.0 - EMA_ALPHA) * ema_state[tid]
        return ema_state[tid].astype(int)
    else:
        box_hist[tid].append(np.array(box, dtype=np.float32))
        return np.mean(box_hist[tid], axis=0).astype(int)


def process_result_frame(frame, res, W, H, crops_per_id, sizes_per_id, box_hist, ema_state):
    if res is None or res.boxes is None or res.boxes.id is None:
        return
    ids = res.boxes.id
    xyxy = res.boxes.xyxy
    if ids is None or xyxy is None:
        return

    ids = ids.int().cpu().numpy().tolist()
    boxes = xyxy.cpu().numpy()

    for i, tid in enumerate(ids):
        x1, y1, x2, y2 = boxes[i].tolist()
        # pad + clamp
        x1 = max(0, int(x1) - PADDING)
        y1 = max(0, int(y1) - PADDING)
        x2 = min(W - 1, int(x2) + PADDING)
        y2 = min(H - 1, int(y2) + PADDING)

        # smooth
        x1, y1, x2, y2 = smooth_box(tid, (x1, y1, x2, y2), box_hist, ema_state)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        crops_per_id[tid].append(crop)
        sizes_per_id[tid].append((w, h))


def run_batch(batch_frames, model, W, H, crops_per_id, sizes_per_id, box_hist, ema_state):
    if not batch_frames:
        return
    results = model.track(
        source=batch_frames,
        persist=True,
        conf=CONF_TH, iou=IOU_TH, classes=[0],
        half=True, device=DEVICE, imgsz=640, verbose=False
    )
    for f_idx, res in enumerate(results):
        process_result_frame(
            frame=batch_frames[f_idx], res=res, W=W, H=H,
            crops_per_id=crops_per_id, sizes_per_id=sizes_per_id,
            box_hist=box_hist, ema_state=ema_state
        )


def track_video_batched(video_path, model):
    fps, (W, H) = ensure_fps_size(video_path)
    crops_per_id = defaultdict(list)
    sizes_per_id = defaultdict(list)
    box_hist, ema_state = new_smoothers()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return None, None, None

    batch_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch_frames.append(frame)
        if len(batch_frames) >= BATCH_SIZE:
            run_batch(batch_frames, model, W, H, crops_per_id, sizes_per_id, box_hist, ema_state)
            batch_frames.clear()

    # leftover
    run_batch(batch_frames, model, W, H, crops_per_id, sizes_per_id, box_hist, ema_state)
    batch_frames.clear()

    cap.release()
    return crops_per_id, sizes_per_id, fps


def _write_single_clip(crops, avg_w, avg_h, fps, temp_out, final_out):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_out, fourcc, fps, (avg_w, avg_h))
    for crop in crops:
        writer.write(cv2.resize(crop, (avg_w, avg_h)))
    writer.release()

    if CONVERT_TO_H264:
        try:
            convert_to_h264(temp_out, final_out)
        finally:
            if os.path.exists(temp_out):
                os.remove(temp_out)
    else:
        try:
            os.rename(temp_out, final_out)
        except Exception as e:
            print(f"[ERROR] rename temp -> final failed: {e}")
        finally:
            if os.path.exists(temp_out):
                os.remove(temp_out)


def write_id_clips_parallel(video_path, crops_per_id, sizes_per_id, fps):
    if not crops_per_id:
        print(f"[SKIP] No persons in {os.path.basename(video_path)}")
        return

    out_subdir = make_out_subdir(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    max_frames = int(fps * MAX_SECONDS) if MAX_SECONDS > 0 else None

    tasks = []
    with ThreadPoolExecutor(max_workers=WRITER_THREADS) as pool:
        for tid, crops in crops_per_id.items():
            if len(crops) < MIN_DETECTIONS_PER_ID or len(crops) < MIN_CLIP_FRAMES:
                print(f"[SKIP] {os.path.basename(video_path)} ID {tid}: frames={len(crops)} (<min)")
                continue

            arr = np.array(sizes_per_id[tid], dtype=np.int32)
            avg_w, avg_h = np.mean(arr, axis=0).astype(int)

            # size rule: keep if at least one dim >= MIN_BOX_SIZE
            if avg_w < MIN_BOX_SIZE and avg_h < MIN_BOX_SIZE:
                print(f"[SKIP] {os.path.basename(video_path)} ID {tid}: avg {avg_w}x{avg_h} too small")
                continue

            temp_out = os.path.join(out_subdir, f"{stem}_id{tid}_temp.mp4")
            final_out = os.path.join(out_subdir, f"{stem}_id{tid}.mp4")
            frames_to_write = crops if max_frames is None else crops[:max_frames]

            # submit writing task
            tasks.append(pool.submit(_write_single_clip, frames_to_write, int(avg_w), int(avg_h), fps, temp_out, final_out))

        # wait + log
        for fut in as_completed(tasks):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] writing clip failed: {e}")

    # print success per file (already printed by convert; optional here)


# ========== PROCESS WORKER ==========

def process_one_video(vp):
    # Each process loads its own model once
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] load model failed in worker: {e}")
        return

    print(f"\n=== Processing {vp} ===")
    try:
        crops_per_id, sizes_per_id, fps = track_video_batched(vp, model)
        if crops_per_id is None:
            return
        write_id_clips_parallel(vp, crops_per_id, sizes_per_id, fps)
    except Exception as e:
        print(f"[ERROR] {os.path.basename(vp)}: {e}")


def process_folder_parallel(input_dir, output_dir):
    # collect videos
    video_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    # multiprocess over videos
    with ProcessPoolExecutor(max_workers=N_YOLO_WORKERS) as ex:
        futs = [ex.submit(process_one_video, vp) for vp in video_files]
        for fut in as_completed(futs):
            # pull exceptions early
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] worker exception: {e}")


if __name__ == "__main__":
    process_folder_parallel(INPUT_DIR, OUTPUT_DIR)