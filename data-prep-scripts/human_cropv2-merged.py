#!/usr/bin/env python3
import os
import cv2
import math
import json
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ========= CONFIG =========
MODEL_PATH = "/home/spactureai/xalil/embedding-matching/models/custom_yolo_model.pt"
INPUT_DIR  = "/home/spactureai/xalil/camera_matching_merged"   # merged videos per camera
OUTPUT_DIR = "/home/spactureai/xalil/embedding-matching/human_detections"   # save JSONs
#!/usr/bin/env python3


# Detection / tracking
CONF_TH = 0.5
IOU_TH  = 0.45
BATCH_SIZE = 640

# Parallelism
N_YOLO_WORKERS = 2
WRITER_THREADS = 8
DEVICE = "0"
# ==========================


def ensure_fps_size(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if math.isclose(fps, 0.0):
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, (w, h)


def make_out_json_path(video_path):
    rel_path = os.path.relpath(video_path, INPUT_DIR)
    rel_dir = os.path.dirname(rel_path)
    out_subdir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(out_subdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(out_subdir, f"{stem}_detections.json")


def run_batch(batch_frames, model, frame_start_idx, detections, fps):
    results = model.track(
        source=batch_frames,
        persist=True,
        conf=CONF_TH, iou=IOU_TH, classes=[0],
        half=True, device=DEVICE, imgsz=640, verbose=False
    )

    for i, res in enumerate(results):
        f_idx = frame_start_idx + i
        if res is None or res.boxes is None or res.boxes.id is None:
            continue
        ids = res.boxes.id
        xyxy = res.boxes.xyxy
        if ids is None or xyxy is None:
            continue

        ids = ids.int().cpu().numpy().tolist()
        boxes = xyxy.cpu().numpy()

        for j, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[j].tolist()
            detections.append({
                "frame_idx": f_idx,
                "time_sec": round(f_idx / fps, 3),
                "id": int(tid),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })


def save_detections_json(video_path, detections, fps, W, H):
    out_path = make_out_json_path(video_path)
    data = {
        "video": os.path.basename(video_path),
        "fps": fps,
        "width": W,
        "height": H,
        "detections": detections
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] JSON saved → {out_path}")


def track_and_save(video_path, model, pool):
    out_path = make_out_json_path(video_path)
    if os.path.exists(out_path):
        print(f"[SKIP] JSON already exists: {out_path}")
        return

    fps, (W, H) = ensure_fps_size(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return

    detections = []
    batch_frames = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch_frames.append(frame)

        if len(batch_frames) >= BATCH_SIZE:
            run_batch(batch_frames, model, frame_idx - len(batch_frames) + 1, detections, fps)
            batch_frames.clear()
        frame_idx += 1

    if batch_frames:
        run_batch(batch_frames, model, frame_idx - len(batch_frames), detections, fps)

    cap.release()
    pool.submit(save_detections_json, video_path, detections, fps, W, H)


def process_one_video(vp):
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] load model failed: {e}")
        return

    print(f"\n=== Processing {vp} ===")
    try:
        with ThreadPoolExecutor(max_workers=WRITER_THREADS) as pool:
            track_and_save(vp, model, pool)
            pool.shutdown(wait=True)
    except Exception as e:
        print(f"[ERROR] {os.path.basename(vp)}: {e}")


def process_folder_parallel(input_dir, output_dir):
    video_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    with ProcessPoolExecutor(max_workers=N_YOLO_WORKERS) as ex:
        futs = [ex.submit(process_one_video, vp) for vp in video_files]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] worker exception: {e}")


if __name__ == "__main__":
    process_folder_parallel(INPUT_DIR, OUTPUT_DIR)