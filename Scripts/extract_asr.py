#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ASR (speech-to-text) from videos using faster-whisper.
Outputs:
  - asr_segments.csv (timestamped lines)
  - asr_summary.csv  (per-video totals: coverage %, WPM, CTA flags)
"""

import argparse, os, re, csv, subprocess, tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# Import check with a friendly message
try:
    from faster_whisper import WhisperModel
except Exception as e:
    raise SystemExit(
        "Missing dependency: faster-whisper\n"
        "Install with:\n"
        "  python -m pip install faster-whisper opencv-python ffmpeg-python --user\n"
        f"Details: {e}"
    )

CTA_RE = re.compile(
    r"\b(book|shop|buy|save|offer|deal|promo\s*code|use\s*code|subscribe|follow|visit|link\s+in\s+bio|today|now|limited|last\s+chance)\b",
    re.I
)

def video_duration_seconds(p: Path) -> float:
    try:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened(): 
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps <= 0: 
            return 0.0
        return float(frames / fps)
    except Exception:
        return 0.0

def ensure_wav(audio_out: Path, video_path: Path, ffmpeg_bin: str=None, sr=16000):
    ffmpeg = ffmpeg_bin or "ffmpeg"
    cmd = [ffmpeg, "-y", "-i", str(video_path), "-ac", "1", "-ar", str(sr), str(audio_out)]
    # Let stderr print if ffmpeg not found — easier to debug
    subprocess.run(cmd, check=True)

def guess_video_id(path: Path) -> str:
    m = re.findall(r"\d{8,}", path.stem)
    if not m:
        m = re.findall(r"\d{8,}", str(path))
    if m:
        m.sort(key=len, reverse=True)
        return m[0]
    return path.stem

def words_count(text: str) -> int:
    return len([t for t in re.findall(r"\w+", text or "")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", required=True, help="Folder containing videos (subfolders OK)")
    ap.add_argument("--out_dir", required=True, help="Output folder for ASR manifests")
    ap.add_argument("--model", default="small", help="faster-whisper model: tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--compute_type", default="float32", help="float32|float16|int8|int8_float16")
    ap.add_argument("--language", default=None, help="force language code, e.g., en")
    ap.add_argument("--ffmpeg_bin", default=None, help="Path to ffmpeg.exe; if omitted, assumes in PATH")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seg_csv = out_dir / "asr_segments.csv"
    sum_csv = out_dir / "asr_summary.csv"

    print("[ASR] Loading model:", args.model)
    model = WhisperModel(args.model,
                         device=(None if args.device=="auto" else args.device),
                         compute_type=args.compute_type)

    seg_rows = []
    sum_rows = []

    vids = [p for p in Path(args.videos_root).rglob("*")
            if p.suffix.lower() in (".mp4",".mov",".mkv",".m4v",".avi")]
    print(f"[ASR] Found {len(vids)} videos under {args.videos_root}")

    for i, vp in enumerate(vids, 1):
        print(f"[ASR] ({i}/{len(vids)}) {vp}")
        try:
            vid = guess_video_id(vp)
            dur = video_duration_seconds(vp)
            if dur <= 0:
                print("   ! Skipping (no duration).")
                continue

            with tempfile.TemporaryDirectory() as td:
                wav = Path(td) / "audio.wav"
                ensure_wav(wav, vp, args.ffmpeg_bin, sr=16000)

                segments, info = model.transcribe(str(wav), language=args.language, vad_filter=True)
                speech_total = 0.0
                full_text = []
                for seg in segments:
                    start = float(seg.start); end = float(seg.end)
                    txt = (seg.text or "").strip()
                    if end > start:
                        speech_total += (end - start)
                    full_text.append(txt)
                    seg_rows.append({
                        "video_id": vid,
                        "video_path": str(vp),
                        "start": round(start,2),
                        "end": round(end,2),
                        "text": txt
                    })

                full = " ".join(full_text)
                wc = words_count(full)
                wpm = (wc / (speech_total/60.0)) if speech_total > 0 else 0.0
                coverage = (speech_total / dur * 100.0) if dur > 0 else 0.0
                cta_hits = CTA_RE.findall(full)
                sum_rows.append({
                    "video_id": vid,
                    "video_path": str(vp),
                    "video_seconds": round(dur,2),
                    "speech_seconds": round(speech_total,2),
                    "asr_words": wc,
                    "asr_wpm": round(wpm,1),
                    "asr_coverage_percent": round(coverage,1),
                    "cta_in_asr": 1 if cta_hits else 0,
                    "cta_terms": ", ".join(sorted(set(m.lower() for m in cta_hits)))
                })

        except subprocess.CalledProcessError:
            print("   ! ffmpeg failed; pass --ffmpeg_bin or add ffmpeg to PATH")
        except Exception as e:
            print("   ! Error:", e)

    if seg_rows:
        pd.DataFrame(seg_rows).to_csv(seg_csv, index=False)
    if sum_rows:
        pd.DataFrame(sum_rows).to_csv(sum_csv, index=False)

    print("[ASR] Wrote:", seg_csv if seg_rows else "(no segments)")
    print("[ASR] Wrote:", sum_csv if sum_rows else "(no summary)")

if __name__ == "__main__":
    main()
