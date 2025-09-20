

#!/usr/bin/env python3
import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ========= CONFIG =========
MODEL_PATH = "/home/spactureai/xalil/dinov3/notebooks/spacture_yolo_model.pt"
INPUT_DIR  = "/home/spactureai/xalil/camera_matching_videos"   # scans recursively
OUTPUT_DIR = "/home/spactureai/xalil/dinov3/all_videos_cropped_v3_images"  # will mirror structure as images

# Detection / tracking
CONF_TH    = 0.5
IOU_TH     = 0.45
BATCH_SIZE = 680                 # frames per YOLO track() call in a batch
DETECT_STRIDE = 2                # run YOLO on every 2nd frame (~half the frames)

# Per-ID filters
PADDING = 0
MIN_DETECTIONS_PER_ID = 50       # need at least this many detections
MIN_CLIP_FRAMES       = 75       # need at least this many frames
MIN_BOX_SIZE          = 224      # keep if (avg_w >= 224) OR (avg_h >= 224)

# Saving (images)
SAVE_ONE_PER_SECOND = True       # write ~1 frame/sec per ID

# Parallelism
N_YOLO_WORKERS = 2               # process-level parallel YOLO for different videos
WRITER_THREADS = 24              # per-process threads to write multiple crops concurrently
DEVICE = "0"                     # "0" / "cuda:0" / "cuda" etc.
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


def make_video_out_base(video_path):
    # OUTPUT_DIR/<camera_subfolder>/<video_stem>
    rel_path = os.path.relpath(video_path, INPUT_DIR)     # e.g. 253105/2025-xx.mp4
    rel_dir  = os.path.dirname(rel_path)                  # e.g. 253105
    stem     = os.path.splitext(os.path.basename(video_path))[0]
    out_dir  = os.path.join(OUTPUT_DIR, rel_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_batch(batch_frames, batch_indices, model, W, H, boxes_per_id, sizes_per_id):
    if not batch_frames:
        return
    results = model.track(
        source=batch_frames,
        persist=True,
        conf=CONF_TH, iou=IOU_TH, classes=[0],
        half=True, device=DEVICE, imgsz=640, verbose=False
    )
    # record boxes per ID (no smoothing)
    for f_idx, res in enumerate(results):
        frame_idx = batch_indices[f_idx]
        if res is None or res.boxes is None or res.boxes.id is None:
            continue
        ids  = res.boxes.id
        xyxy = res.boxes.xyxy
        if ids is None or xyxy is None:
            continue

        ids  = ids.int().cpu().numpy().tolist()
        boxes = xyxy.cpu().numpy()
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i].tolist()
            # pad + clamp
            x1 = max(0, int(x1) - PADDING)
            y1 = max(0, int(y1) - PADDING)
            x2 = min(W - 1, int(x2) + PADDING)
            y2 = min(H - 1, int(y2) + PADDING)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_per_id[tid].append((frame_idx, x1, y1, x2, y2))
            sizes_per_id[tid].append((x2 - x1, y2 - y1))


def track_boxes_batched(video_path, model):
    fps, (W, H) = ensure_fps_size(video_path)

    boxes_per_id = defaultdict(list)  # tid -> [(frame_idx, x1,y1,x2,y2), ...]
    sizes_per_id = defaultdict(list)  # tid -> [(w,h), ...]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return None, None, None

    batch_frames = []
    batch_indices = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % DETECT_STRIDE == 0:
            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            if len(batch_frames) >= BATCH_SIZE:
                run_batch(batch_frames, batch_indices, model, W, H, boxes_per_id, sizes_per_id)
                batch_frames.clear(); batch_indices.clear()

        frame_idx += 1

    # leftover
    run_batch(batch_frames, batch_indices, model, W, H, boxes_per_id, sizes_per_id)
    batch_frames.clear(); batch_indices.clear()
    cap.release()
    return boxes_per_id, sizes_per_id, fps


def filter_ids_by_rules(boxes_per_id, sizes_per_id):
    keep = set()
    for tid, boxes in boxes_per_id.items():
        n = len(boxes)
        if n < MIN_DETECTIONS_PER_ID or n < MIN_CLIP_FRAMES:
            continue
        arr = np.array(sizes_per_id[tid], dtype=np.int32)
        if arr.size == 0:
            continue
        avg_w, avg_h = np.mean(arr, axis=0).astype(int)
        if avg_w < MIN_BOX_SIZE and avg_h < MIN_BOX_SIZE:
            continue
        keep.add(tid)
    return keep


def build_one_fps_crop_index(boxes_per_id, selected_ids, fps):
    """
    From detected frames (possibly strided), pick ~1 frame/sec per ID:
      keep the first detection seen for each integer second bucket.
    Return: frame_to_crops: dict[frame_idx] -> list[(tid, x1,y1,x2,y2)]
    """
    frame_to_crops = defaultdict(list)
    for tid in selected_ids:
        seen_secs = set()
        # sorted by frame index
        for (fi, x1, y1, x2, y2) in sorted(boxes_per_id[tid], key=lambda x: x[0]):
            sec = int(fi / fps + 1e-6)
            if sec in seen_secs:
                continue
            seen_secs.add(sec)
            frame_to_crops[fi].append((tid, x1, y1, x2, y2))
    return frame_to_crops


def save_frames_for_ids(video_path, boxes_per_id, selected_ids, out_base, fps):
    """
    Re-read the video once, crop and save frames for selected IDs.
    Output: OUTPUT_DIR/<camera>/<video_stem>/id<TID>/frame_<frameIdx>.jpg
    """
    # 1 fps per ID index
    if SAVE_ONE_PER_SECOND:
        frame_to_crops = build_one_fps_crop_index(boxes_per_id, selected_ids, fps)
    else:
        frame_to_crops = defaultdict(list)
        for tid in selected_ids:
            for (fi, x1, y1, x2, y2) in boxes_per_id[tid]:
                frame_to_crops[fi].append((tid, x1, y1, x2, y2))

    # ensure id dirs exist
    for tid in selected_ids:
        os.makedirs(os.path.join(out_base, f"id{tid}"), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot re-open: {video_path}")
        return

    with ThreadPoolExecutor(max_workers=WRITER_THREADS) as pool:
        futures = []
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            crops = frame_to_crops.get(fi, [])
            for (tid, x1, y1, x2, y2) in crops:
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                out_path = os.path.join(out_base, f"id{tid}", f"frame_{fi:06d}.jpg")
                futures.append(pool.submit(cv2.imwrite, out_path, crop.copy()))

            fi += 1

        for f in as_completed(futures):
            _ = f.result()

    cap.release()


# ========== PROCESS WORKER ==========

def process_one_video(vp):
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] load model failed in worker: {e}")
        return

    print(f"\n=== Processing {vp} ===")
    try:
        boxes_per_id, sizes_per_id, fps = track_boxes_batched(vp, model)
        if boxes_per_id is None:
            return
        selected = filter_ids_by_rules(boxes_per_id, sizes_per_id)
        if not selected:
            print(f"[SKIP] No IDs passed filters in {os.path.basename(vp)}")
            return
        out_base = make_video_out_base(vp)
        save_frames_for_ids(vp, boxes_per_id, selected, out_base, fps)
        print(f"[OK] Saved frames -> {out_base}")
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

    with ProcessPoolExecutor(max_workers=N_YOLO_WORKERS) as ex:
        futs = [ex.submit(process_one_video, vp) for vp in video_files]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] worker exception: {e}")


if __name__ == "__main__":
    process_folder_parallel(INPUT_DIR, OUTPUT_DIR)