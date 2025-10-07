#!/usr/bin/env python3
import os
import subprocess

INPUT_DIR  = "/home/spactureai/xalil/camera_matching_videos"
OUTPUT_DIR = "/home/spactureai/xalil/camera_matching_merged"

def merge_videos(camera_dir, out_dir):
    files = [f for f in os.listdir(camera_dir) if f.endswith(".mp4")]
    if not files:
        return None

    # sort by filename (timestamps in names → correct order)
    files.sort()

    # create ffmpeg concat list
    list_path = os.path.join(camera_dir, "concat_list.txt")
    with open(list_path, "w") as f:
        for fmp4 in files:
            full_path = os.path.join(camera_dir, fmp4)
            f.write(f"file '{full_path}'\n")

    cam_name = os.path.basename(camera_dir.rstrip("/"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cam_name}_merged.mp4")

    # ffmpeg concat (stream copy, no re-encode)
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy", out_path
    ]
    subprocess.run(cmd, check=True)

    os.remove(list_path)
    print(f"[OK] Merged {len(files)} clips into {out_path}")
    return out_path

def main():
    for cam in os.listdir(INPUT_DIR):
        cam_path = os.path.join(INPUT_DIR, cam)
        if not os.path.isdir(cam_path):
            continue
        try:
            merge_videos(cam_path, OUTPUT_DIR)
        except Exception as e:
            print(f"[ERROR] Failed merging {cam_path}: {e}")

if __name__ == "__main__":
    main()