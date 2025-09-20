#!/usr/bin/env python3
"""
Find top matching pairs of video embeddings (cosine) BELOW a threshold,
across different subfolders only, and RE-ENCODE both source videos to H.264
into OUTPUT_DIR/pair_XX_score/ for visual review.
"""

import os
import subprocess
import torch

# ========= Configuration =========
EMBED_DIR   = "./embedding_v2"               # embeddings saved recursively here
VIDEO_DIR   = "./all_videos_cropped_v2"   # videos mirrored here
OUTPUT_DIR  = "./embedding_match_below095_h264"

TOP_PAIRS     = 20        # number of pairs to export
SIM_THRESHOLD = 0.95      # include only pairs with score < this

VIDEO_EXTS    = (".mp4", ".mov", ".mkv", ".avi", ".webm")

# ffmpeg encode params
CRF        = "23"         # lower=better quality, bigger size (e.g. 18-28)
PRESET     = "veryfast"   # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
PIX_FMT    = "yuv420p"
FASTSTART  = True         # move moov atom for faster web playback
THREADS    = "0"          # "0" = auto
# ================================


def list_pt_recursive(root_dir: str):
    pts = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".pt"):
                full = os.path.join(r, f)
                rel_path = os.path.relpath(full, root_dir)             # e.g. areaA/clip1.pt
                rel_dir  = os.path.dirname(rel_path)                   # e.g. areaA ('' if root)
                stem     = os.path.splitext(os.path.basename(f))[0]    # e.g. clip1
                pts.append((full, rel_dir, stem))
    pts.sort()
    return pts


def find_video(video_root: str, rel_dir: str, stem: str) -> str | None:
    for ext in VIDEO_EXTS:
        candidate = os.path.join(video_root, rel_dir, stem + ext) if rel_dir else os.path.join(video_root, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def load_embeddings(embed_dir: str):
    entries = list_pt_recursive(embed_dir)
    if not entries:
        raise RuntimeError(f"No .pt embeddings found in {embed_dir}")

    tensors, names, groups = [], [], []
    for full, rel_dir, stem in entries:
        obj = torch.load(full, map_location="cpu")
        emb = obj.get("embedding", None)
        if emb is None:
            continue
        emb = emb.float().view(-1)
        tensors.append(emb)
        names.append((rel_dir, stem))  # keep folder + stem
        groups.append(rel_dir)         # group by subfolder ('' means root)

    if not tensors:
        raise RuntimeError("No embeddings loaded (missing 'embedding' key?)")

    stacked = torch.stack(tensors, dim=0)  # [N, D]
    return stacked, names, groups


def cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    norms = vectors.norm(dim=1, keepdim=True).clamp_min(1e-8)
    normalized = vectors / norms
    return normalized @ normalized.T


def get_top_pairs_below(names, groups, sims: torch.Tensor, top_pairs: int, threshold: float):
    n = sims.shape[0]
    sims = sims.clone()
    sims.fill_diagonal_(-1.0)

    tri_i, tri_j = torch.triu_indices(n, n, offset=1)
    tri_vals = sims[tri_i, tri_j]

    # Only cross-subfolder comparisons
    mask_groups = torch.tensor([groups[tri_i[k]] != groups[tri_j[k]] for k in range(tri_i.numel())], dtype=torch.bool)
    tri_vals = tri_vals[mask_groups]; tri_i = tri_i[mask_groups]; tri_j = tri_j[mask_groups]

    # Only below threshold
    mask_thr = tri_vals < threshold
    tri_vals = tri_vals[mask_thr]; tri_i = tri_i[mask_thr]; tri_j = tri_j[mask_thr]

    if tri_vals.numel() == 0:
        print("No cross-folder pairs found below threshold.")
        return []

    # Highest scores but still < threshold
    top_pairs = min(top_pairs, tri_vals.numel())
    vals, order = torch.topk(tri_vals, k=top_pairs, largest=True, sorted=True)

    pairs = []
    for k in range(top_pairs):
        i = tri_i[order[k]].item()
        j = tri_j[order[k]].item()
        score = vals[k].item()
        pairs.append((names[i], names[j], score))  # ((rel_dir_i, stem_i), (rel_dir_j, stem_j), score)
    return pairs


def reencode_h264(src_path: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-crf", CRF,
        "-preset", PRESET,
        "-pix_fmt", PIX_FMT,
        "-movflags", "+faststart" if FASTSTART else "faststart",
        "-an",
        "-threads", THREADS,
        dst_path
    ]
    # Remove -movflags if FASTSTART is False to avoid setting a weird value
    if not FASTSTART:
        cmd.remove("-movflags"); cmd.remove("faststart")
    subprocess.run(cmd, check=True)


def export_pairs_as_h264(pairs, video_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for idx, (name1, name2, score) in enumerate(pairs, 1):
        (dir1, stem1) = name1
        (dir2, stem2) = name2

        pair_dir = os.path.join(output_dir, f"pair_{idx:02d}_{score:.4f}")
        os.makedirs(pair_dir, exist_ok=True)

        src1 = find_video(video_dir, dir1, stem1)
        src2 = find_video(video_dir, dir2, stem2)

        if not src1:
            print(f"[warn] Missing video for {os.path.join(dir1, stem1)} in {video_dir}")
        else:
            out1 = os.path.join(pair_dir, f"{stem1}_h264.mp4")
            try:
                reencode_h264(src1, out1)
            except Exception as e:
                print(f"[ERROR] reencode failed for {src1}: {e}")

        if not src2:
            print(f"[warn] Missing video for {os.path.join(dir2, stem2)} in {video_dir}")
        else:
            out2 = os.path.join(pair_dir, f"{stem2}_h264.mp4")
            try:
                reencode_h264(src2, out2)
            except Exception as e:
                print(f"[ERROR] reencode failed for {src2}: {e}")

        print(f"[OK] Wrote H.264 pair {idx}: {os.path.join(dir1, stem1)} <-> {os.path.join(dir2, stem2)} (score={score:.4f})")


def main():
    vectors, names, groups = load_embeddings(EMBED_DIR)
    sims = cosine_similarity_matrix(vectors)

    print(f"Loaded {len(names)} embeddings from {EMBED_DIR} (dim={vectors.shape[1]}).")

    pairs = get_top_pairs_below(names, groups, sims, TOP_PAIRS, SIM_THRESHOLD)
    if not pairs:
        return

    print("\nTop matches below threshold (cross-folder only):")
    for i, (n1, n2, score) in enumerate(pairs, 1):
        print(f"{i:02d}. {os.path.join(n1[0], n1[1])}  <->  {os.path.join(n2[0], n2[1])}   {score:.4f}")

    export_pairs_as_h264(pairs, VIDEO_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()