#!/usr/bin/env python3
"""
fetch_tiktok_assets.py

Download TikTok cover images (DIRECT or via yt-dlp) and, optionally,
download full videos and extract frames every N seconds.

Usage examples:
  # Direct covers (uses 'video/cover' columns if present); also writes video_url in manifest
  python fetch_tiktok_assets.py --inputs dataset.csv --out covers

  # Force yt-dlp fallback to build canonical video URLs and fetch thumbnails
  python fetch_tiktok_assets.py --inputs dataset.csv --out covers --use-ytdlp

  # Also download videos and extract frames every 2 seconds (to frames/)
  python fetch_tiktok_assets.py --inputs dataset.csv --out assets --extract-frames --interval 2 --frames-out frames

Dependencies:
  pip install yt-dlp requests pandas opencv-python
  Optional: install ffmpeg (recommended) for fast frame extraction.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# ----------------- helpers -----------------
def sanitize_video_id(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    m = re.search(r'(\d+)', v)
    return m.group(1) if m else None

def extract_handle_from_input(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        segments = [seg for seg in p.path.split('/') if seg]
        for seg in segments:
            if seg.startswith('@'):
                return seg.lstrip('@')
        if segments:
            return segments[-1].lstrip('@')
    except Exception:
        pass
    return None

def canonical_video_url(handle: Optional[str], vid: Optional[str]) -> Optional[str]:
    if not handle or not vid:
        return None
    return f"https://www.tiktok.com/@{handle}/video/{vid}"

def pick_cover_url(row: pd.Series) -> Optional[str]:
    for key in ["video/originCover", "video/cover", "video/dynamicCover"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    for key in ["originCover", "cover", "dynamicCover"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    return None

def guess_ext_from_headers(content_type: str) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct: return "jpg"
    if "png" in ct: return "png"
    if "webp" in ct: return "webp"
    return "jpg"

def guess_ext_from_url(url: str) -> str:
    lower = url.lower()
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if ext in lower:
            return ext.strip(".")
    return "jpg"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_manifest(path: Path, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") or shutil.which("ffmpeg.exe"))

def have_ytdlp() -> bool:
    return bool(shutil.which("yt-dlp") or shutil.which("yt-dlp.exe"))

# ----------------- main -----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Download TikTok covers and optionally extract frames.")
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV(s) exported from scraper")
    ap.add_argument("--out", default="covers", help="Output directory for images and downloads")
    ap.add_argument("--manifest", default="covers_manifest.csv", help="Covers manifest CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Limit for testing")
    ap.add_argument("--use-ytdlp", action="store_true", help="Force yt-dlp for covers (build URLs and grab thumbnails)")
    # Frame extraction
    ap.add_argument("--extract-frames", action="store_true", help="Download full videos and extract frames")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between frames (when --extract-frames)")
    ap.add_argument("--frames-out", default="frames", help="Folder for extracted frames")
    ap.add_argument("--frames-manifest", default="frames_manifest.csv", help="Manifest CSV for frames")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    frames_root = Path(args.frames_out)
    if args.extract_frames:
        ensure_dir(frames_root)

    # Load CSVs
    dfs = []
    for p in args.inputs:
        try:
            dfs.append(pd.read_csv(p, dtype=str, low_memory=False).assign(_src=p))
        except Exception as e:
            print(f"ERROR reading {p}: {e}", file=sys.stderr)
    if not dfs:
        print("No readable CSVs.", file=sys.stderr)
        return 2
    df = pd.concat(dfs, ignore_index=True, sort=False)

    # Identify id column
    id_col = "id" if "id" in df.columns else ("video/id" if "video/id" in df.columns else None)
    if not id_col:
        print("No 'id' or 'video/id' column found.", file=sys.stderr)
        return 3

    # Prepare base rows
    cover_rows: List[Dict] = []

    # Try DIRECT covers first (unless forcing yt-dlp)
    direct_candidates = []
    if not args.use_ytdlp:
        for _, row in df.iterrows():
            vid = sanitize_video_id(row.get(id_col))
            if not vid:
                continue
            handle = extract_handle_from_input(row.get("input")) or ""
            vurl = canonical_video_url(handle, vid) or ""
            url = pick_cover_url(row)
            if url:
                direct_candidates.append((row["_src"], vid, handle, vurl, url))
        if args.limit:
            direct_candidates = direct_candidates[:args.limit]

    if direct_candidates and not args.use_ytdlp:
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except Exception:
            print("ERROR: 'requests' not installed; run: pip install requests  OR use --use-ytdlp", file=sys.stderr)
            return 4

        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CoverFetcher/1.2)"}

        for src, vid, handle, vurl, url in direct_candidates:
            try:
                resp = session.get(url, headers=headers, timeout=20)
                if resp.status_code != 200 or not resp.content:
                    cover_rows.append({
                        "source_file": src, "video_id": vid, "handle": handle,
                        "video_url": vurl, "cover_url": url,
                        "status": "http_error", "saved_path": "", "error": f"status={resp.status_code}"
                    })
                    continue
                ext = guess_ext_from_headers(resp.headers.get("Content-Type","")) or guess_ext_from_url(url)
                out_path = out_dir / f"{vid}.{ext}"
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                cover_rows.append({
                    "source_file": src, "video_id": vid, "handle": handle,
                    "video_url": vurl, "cover_url": url,
                    "status": "ok", "saved_path": str(out_path), "error": ""
                })
            except Exception as e:
                cover_rows.append({
                    "source_file": src, "video_id": vid, "handle": handle,
                    "video_url": vurl, "cover_url": url,
                    "status": "error", "saved_path": "", "error": str(e)
                })

        write_manifest(Path(args.manifest), cover_rows,
                       ["source_file","video_id","handle","video_url","cover_url","status","saved_path","error"])
        print(f"Direct covers: {len([r for r in cover_rows if r['status']=='ok'])} OK / {len(direct_candidates)} tried.")
    else:
        # FALLBACK via yt-dlp: build canonical video URLs and let yt-dlp grab thumbnails
        if "input" not in df.columns:
            print("No 'input' column for handle extraction; cannot build canonical video URLs.", file=sys.stderr)
            return 5

        def build_video_url(row) -> Optional[str]:
            vid = sanitize_video_id(row.get(id_col))
            handle = extract_handle_from_input(row.get("input"))
            if not vid or not handle:
                return None
            return f"https://www.tiktok.com/@{handle}/video/{vid}"

        df["video_url"] = df.apply(build_video_url, axis=1)
        urls = df["video_url"].dropna().drop_duplicates().tolist()
        if args.limit:
            urls = urls[:args.limit]
        if not urls:
            print("No canonical video URLs could be constructed from CSV.", file=sys.stderr)
            return 6

        url_list_path = Path("video_urls.txt")
        with open(url_list_path, "w", encoding="utf-8") as f:
            for u in urls:
                f.write(u + "\n")

        ytdlp_exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
        cmd = [ytdlp_exe or sys.executable, "-m", "yt_dlp"] if not ytdlp_exe else [ytdlp_exe]
        cmd += ["--write-thumbnail", "--skip-download", "-o", str(out_dir / "%(id)s.%(ext)s"), "-a", str(url_list_path)]

        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"ERROR running yt-dlp: {e}", file=sys.stderr)
            return 8

        # Build covers manifest by scanning out_dir
        for u in urls:
            vid = u.rstrip("/").split("/")[-1]
            handle = u.split("/@")[1].split("/video/")[0] if "/@" in u and "/video/" in u else ""
            matched = list(out_dir.glob(f"{vid}.*"))
            if matched:
                cover_rows.append({
                    "source_file": "yt-dlp", "video_id": vid, "handle": handle,
                    "video_url": u, "cover_url": u, "status": "ok",
                    "saved_path": str(matched[0]), "error": ""
                })
            else:
                cover_rows.append({
                    "source_file": "yt-dlp", "video_id": vid, "handle": handle,
                    "video_url": u, "cover_url": u, "status": "missing",
                    "saved_path": "", "error": ""
                })
        write_manifest(Path(args.manifest), cover_rows,
                       ["source_file","video_id","handle","video_url","cover_url","status","saved_path","error"])
        print(f"yt-dlp thumbnails: {len([r for r in cover_rows if r['status']=='ok'])} OK / {len(urls)} tried.")

    # ----------------- Optional: download full videos & extract frames -----------------
    if not args.extract_frames:
        return 0

    # Build list of unique video URLs to download (prefer those we already constructed)
    if "video_url" in df.columns:
        urls_for_video = df["video_url"].dropna().drop_duplicates().tolist()
    else:
        # rebuild minimal set from covers manifest
        urls_for_video = sorted(set([r["video_url"] for r in cover_rows if r.get("video_url")]))

    if not urls_for_video:
        print("No video URLs available for --extract-frames.", file=sys.stderr)
        return 0

    # Download videos with yt-dlp
    downloads_dir = out_dir / "videos"
    ensure_dir(downloads_dir)
    url_list_path = Path("video_urls_for_download.txt")
    with open(url_list_path, "w", encoding="utf-8") as f:
        for u in urls_for_video:
            f.write(u + "\n")

    dl_cmd = [shutil.which("yt-dlp") or sys.executable]
    if dl_cmd[0].endswith("yt-dlp") or dl_cmd[0].endswith("yt-dlp.exe"):
        pass
    else:
        dl_cmd += ["-m", "yt_dlp"]
    dl_cmd += ["-o", str(downloads_dir / "%(id)s.%(ext)s"), "-a", str(url_list_path)]
    print("Downloading videos:", " ".join(dl_cmd))
    try:
        subprocess.run(dl_cmd, check=False)
    except Exception as e:
        print(f"ERROR downloading videos: {e}", file=sys.stderr)
        return 9

    # Extract frames
    frames_rows: List[Dict] = []
    ffmpeg_ok = have_ffmpeg()
    if not ffmpeg_ok:
        print("[!] ffmpeg not found; falling back to OpenCV for frame extraction (slower).", file=sys.stderr)

    # Build a quick map id->url for association
    id_to_url = {}
    for u in urls_for_video:
        vid = u.rstrip("/").split("/")[-1]
        id_to_url[vid] = u

    for vid_file in downloads_dir.glob("*"):
        if not vid_file.is_file():
            continue
        vid_stem = vid_file.stem
        video_url = id_to_url.get(vid_stem, "")
        frame_dir = frames_root / vid_stem
        ensure_dir(frame_dir)

        if ffmpeg_ok:
            # ffmpeg: one frame every N seconds; name with timestamp via frame number approximation
            # Using select=not(mod(t\,INTERVAL)) is simpler via fps=1/INTERVAL
            out_pattern = str(frame_dir / (vid_stem + "_%05d.jpg"))
            cmd = ["ffmpeg", "-y", "-i", str(vid_file), "-vf", f"fps=1/{max(args.interval, 0.01)}", "-q:v", "2", out_pattern]
            print("FFMPEG:", " ".join(cmd))
            subprocess.run(cmd, check=False)

            # Build manifest rows by enumerating files (timestamp is approximated by index*interval)
            for i, f in enumerate(sorted(frame_dir.glob(f"{vid_stem}_*.jpg"))):
                frames_rows.append({
                    "video_id": vid_stem,
                    "video_url": video_url,
                    "frame_file": str(f),
                    "timestamp_s": round(i * args.interval, 3),
                    "index": i
                })
        else:
            # Fallback: OpenCV extraction
            import cv2
            cap = cv2.VideoCapture(str(vid_file))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = max(1, int(round(fps * args.interval)))
            idx = 0; saved = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    ts = idx / fps
                    outp = frame_dir / f"{vid_stem}_{saved:05d}_{ts:.2f}s.jpg"
                    cv2.imwrite(str(outp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    frames_rows.append({
                        "video_id": vid_stem,
                        "video_url": video_url,
                        "frame_file": str(outp),
                        "timestamp_s": round(float(ts), 3),
                        "index": saved
                    })
                    saved += 1
                idx += 1
            cap.release()

    # Write frames manifest
    write_manifest(Path(args.frames_manifest), frames_rows,
                   ["video_id","video_url","frame_file","timestamp_s","index"])
    print(f"Frames extracted → {args.frames_out}. Manifest: {args.frames_manifest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
