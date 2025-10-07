#!/usr/bin/env python3
import os
import shutil

# Paths
VIDEO_DIR = "/home/spactureai/xalil/camera_matching_videos"
DETECTIONS_DIR = "/home/spactureai/xalil/embedding-matching/datasets/human_detections"
OUTPUT_DIR = "/home/spactureai/xalil/embedding-matching/datasets/combined_dataset"

# Extensions we care about
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def main():
    for root, _, files in os.walk(VIDEO_DIR):
        for f in files:
            if not f.lower().endswith(VIDEO_EXTS):
                continue

            video_path = os.path.join(root, f)
            rel_path = os.path.relpath(video_path, VIDEO_DIR)   # relative path inside VIDEO_DIR
            rel_dir = os.path.dirname(rel_path)                 # e.g. "253104"
            camera_name = os.path.basename(rel_dir)             # just the folder name

            # Detection JSON path
            stem = os.path.splitext(f)[0]                       # "2025-08-15_09-06-41-722306"
            json_name = f"{stem}_detections.json"
            json_path = os.path.join(DETECTIONS_DIR, rel_dir, json_name)

            if not os.path.exists(json_path):
                print(f"[WARN] Missing JSON for {video_path}")
                continue

            # Output subdir
            out_subdir = os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(out_subdir, exist_ok=True)

            # Prefix with camera_name
            new_stem = f"{camera_name}_{stem}"
            out_video = os.path.join(out_subdir, f"{new_stem}{os.path.splitext(f)[1]}")
            out_json  = os.path.join(out_subdir, f"{new_stem}.json")

            # Copy files
            shutil.copy2(video_path, out_video)
            shutil.copy2(json_path, out_json)

            print(f"[OK] {f} + detections → {out_subdir} (renamed with prefix {camera_name}_)")

if __name__ == "__main__":
    main()