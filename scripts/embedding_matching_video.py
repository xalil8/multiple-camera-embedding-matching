#!/usr/bin/env python3
"""
Given a QUERY_VIDEO and SEARCH_DIRS, load precomputed embeddings from EMBED_DIR,
find top-N most similar clips by cosine similarity, and re-encode those videos
to H.264 into OUTPUT_ROOT/<query_stem>/.

Assumes embeddings are stored mirroring the video folder structure:
  VIDEO_ROOT/.../<name>.<ext>  <->  EMBED_DIR/.../<name>.pt
"""

import os
import subprocess
import torch

# ========= Configuration =========
QUERY_VIDEO  = "/home/spactureai/xalil/dinov3/all_videos_cropped_v2/206-new-hyde-park-rd_face1/2025-08-15_09-00-04-819683_id2.mp4"   # <-- set this
SEARCH_DIRS  = [
    "./all_videos_cropped_v2",
    # "/another/search/folder",
]

EMBED_DIR    = "./embedding_v2"                # embeddings root (mirrors search dirs)
OUTPUT_ROOT  = "./search_results_h264"         # results saved under OUTPUT_ROOT/<query_stem>/

TOP_K        = 20                               # number of best matches to export
MIN_SIM      = 0.0                              # optional: ignore results below this cosine (0..1)

VIDEO_EXTS   = (".mp4", ".mov", ".mkv", ".avi", ".webm")

# ffmpeg encode params
CRF          = "18"                             # 18(high quality)..28(small)
PRESET       = "veryfast"                       # speed/quality tradeoff
PIX_FMT      = "yuv420p"
FASTSTART    = True                             # moov atom at start
THREADS      = "0"                              # "0" = auto
# =================================


def list_videos_recursive(dirs):
    vids = []
    for root_dir in dirs:
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(VIDEO_EXTS):
                    vids.append(os.path.join(root, f))
    vids.sort()
    return vids


def relpath_to_any_root(path, roots):
    """Return (root, rel) where rel = relative path of 'path' under the first matching root; else (None, None)."""
    ap = os.path.abspath(path)
    for r in roots:
        ar = os.path.abspath(r)
        try:
            rel = os.path.relpath(ap, ar)
            # If path is not under root, relpath will go up with ".." parts; reject those
            if not rel.startswith(".."):
                return ar, rel
        except Exception:
            pass
    return None, None


def embed_path_for_video(video_path, embed_root, search_roots):
    """
    Map a video path to its embedding:
      <some_search_root>/<rel_dir>/<stem>.<ext> -> EMBED_DIR/<rel_dir>/<stem>.pt
    """
    base, rel = relpath_to_any_root(video_path, search_roots)
    if base is None:
        return None
    rel_dir = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    emb_path = os.path.join(embed_root, rel_dir, stem + ".pt") if rel_dir else os.path.join(embed_root, stem + ".pt")
    return emb_path


def find_embed_by_stem(stem, embed_root):
    """Fallback: search EMBED_DIR recursively for the first file whose stem matches."""
    for r, _, files in os.walk(embed_root):
        for f in files:
            if f.endswith(".pt") and os.path.splitext(f)[0] == stem:
                return os.path.join(r, f)
    return None


def load_embedding(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    emb = obj.get("embedding", None)
    if emb is None:
        raise RuntimeError(f"Missing 'embedding' key in {pt_path}")
    return emb.float().view(-1)


def reencode_h264(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-crf", CRF,
        "-preset", PRESET,
        "-pix_fmt", PIX_FMT,
        "-an",
        "-threads", THREADS,
    ]
    if FASTSTART:
        # MUST be an *output* option, i.e., before dst_path, after -i and codec opts
        cmd += ["-movflags", "+faststart"]
    cmd += [dst_path]

    # optional: cleaner errors
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("ffmpeg command:", " ".join(cmd))
        raise RuntimeError(f"ffmpeg failed for {src_path} -> {dst_path}\n{proc.stderr}")

def main():
    if not os.path.exists(QUERY_VIDEO):
        raise FileNotFoundError(f"Query video not found: {QUERY_VIDEO}")
    if not os.path.isdir(EMBED_DIR):
        raise FileNotFoundError(f"Embedding root not found: {EMBED_DIR}")

    # Resolve query embedding
    q_stem = os.path.splitext(os.path.basename(QUERY_VIDEO))[0]
    q_emb_path = embed_path_for_video(QUERY_VIDEO, EMBED_DIR, SEARCH_DIRS)
    if not q_emb_path or not os.path.exists(q_emb_path):
        # fallback: find by stem anywhere in EMBED_DIR
        q_emb_path = find_embed_by_stem(q_stem, EMBED_DIR)
    if not q_emb_path or not os.path.exists(q_emb_path):
        raise FileNotFoundError(f"No embedding found for query video (stem={q_stem}) in {EMBED_DIR}")

    q = load_embedding(q_emb_path)
    q = q / (q.norm().clamp_min(1e-8))

    # Collect candidate videos + embeddings from SEARCH_DIRS
    vids = list_videos_recursive(SEARCH_DIRS)
    if not vids:
        print("No candidate videos found in SEARCH_DIRS.")
        return

    q_abs = os.path.abspath(QUERY_VIDEO)
    scored = []
    for vp in vids:
        if os.path.abspath(vp) == q_abs:
            continue
        emb_path = embed_path_for_video(vp, EMBED_DIR, SEARCH_DIRS)
        if not emb_path or not os.path.exists(emb_path):
            # If your embedding tree mirrors SEARCH_DIRS, this should exist; otherwise skip
            continue
        try:
            e = load_embedding(emb_path)
            e = e / (e.norm().clamp_min(1e-8))
            score = float(q.dot(e))
            if score >= MIN_SIM:
                scored.append((score, vp))
        except Exception as ex:
            print(f"[warn] Failed to load/score {vp}: {ex}")

    if not scored:
        print("No candidates produced a valid similarity above MIN_SIM.")
        return

    # Sort by similarity (desc), take top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    topn = scored[:TOP_K]

    # Output folder for this query
    out_dir = os.path.join(OUTPUT_ROOT, q_stem)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nTop {len(topn)} matches for {os.path.basename(QUERY_VIDEO)}:")
    for rank, (s, vp) in enumerate(topn, 1):
        print(f"{rank:02d}. {vp}   {s:.4f}")

    # Re-encode to H.264 into the query's result folder
    for rank, (s, vp) in enumerate(topn, 1):
        dst = os.path.join(out_dir, f"{rank:02d}_{os.path.splitext(os.path.basename(vp))[0]}_h264.mp4")
        try:
            reencode_h264(vp, dst)
        except Exception as e:
            print(f"[ERROR] ffmpeg encode failed for {vp}: {e}")

    print(f"\n[OK] Wrote {len(topn)} H.264 matches to: {out_dir}")


if __name__ == "__main__":
    main()