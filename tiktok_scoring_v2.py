# tiktok_scoring_v2.py
# Unified TikTok scoring + brand tone + overlay/cut analysis
# ----------------------------------------------------------
# Usage examples at bottom.

import argparse, csv, json, re, sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2

# ---------- Optional deps ----------
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ---------- Brand color helpers ----------
def rgb_to_lab(rgb):
    rgb = np.array(rgb, dtype=np.float32) / 255.0
    def inv_gamma(u): return np.where(u <= 0.04045, u/12.92, ((u+0.055)/1.055)**2.4)
    r,g,b = inv_gamma(rgb[...,0]), inv_gamma(rgb[...,1]), inv_gamma(rgb[...,2])
    X = (0.4124564*r + 0.3575761*g + 0.1804375*b) / 0.95047
    Y = (0.2126729*r + 0.7151522*g + 0.0721750*b) / 1.00000
    Z = (0.0193339*r + 0.1191920*g + 0.9503041*b) / 1.08883
    def f(t):
        eps = 216/24389; kappa = 24389/27
        return np.where(t>eps, np.cbrt(t), (kappa*t+16)/116)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116*fy - 16; a = 500*(fx - fy); b = 200*(fy - fz)
    return np.stack([L,a,b], axis=-1)

def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def delta_e(lab1, lab2):
    return float(np.linalg.norm(lab1 - lab2))

def find_hex_codes_in_text(txt: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"#[0-9A-Fa-f]{6}", txt or "")))

def load_brand_palette(pdf_path: str=None, brand_json: str=None) -> List[str]:
    colors = []
    if brand_json and Path(brand_json).is_file():
        try:
            with open(brand_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            colors = data.get("colors", [])
        except Exception:
            pass
    if not colors and pdf_path and Path(pdf_path).is_file() and PDF_AVAILABLE:
        try:
            text = extract_text(pdf_path)
            hexes = find_hex_codes_in_text(text)
            colors = hexes[:12]
        except Exception:
            pass
    colors = [c.upper() if c.startswith("#") else ("#"+c.upper()) for c in colors]
    colors = [c for c in colors if re.match(r"^#[0-9A-F]{6}$", c)]
    return colors

# ---------- Video analysis ----------
def ocr_text(frame_bgr):
    if not OCR_AVAILABLE: return ""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return pytesseract.image_to_string(gray, config="--psm 6")

def kmeans_dominant_colors(bgr_img, k=5):
    img = cv2.resize(bgr_img, (256, max(1, int(bgr_img.shape[0]*256/bgr_img.shape[1]))))
    data = img.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    ret, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    counts = np.bincount(labels.flatten(), minlength=k)
    idx = np.argsort(-counts)
    centers = centers[idx]; counts = counts[idx]
    return centers[:, ::-1], counts  # RGB

def frame_histogram(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[16,16,8],[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_frames(video_path: Path, out_dir: Path, interval_s=2.0, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps*interval_s)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_info = []
    idx = 0; saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % frame_interval == 0:
            t = idx / fps
            fname = f"frame_{saved:05d}_{t:.2f}s.jpg"
            fp = out_dir / fname
            cv2.imwrite(str(fp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            frames_info.append((fp, t, frame))
            saved += 1
            if max_frames and saved >= max_frames: break
        idx += 1
    cap.release()
    return frames_info

def analyze_video(video_path: Path, out_root: Path, interval_s: float,
                  brand_colors: List[str], delta_e_thresh=12.0, k_colors=5):
    stem = video_path.stem
    out_dir = out_root / stem / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(video_path, out_dir, interval_s=interval_s)

    brand_lab = []
    for hx in brand_colors:
        lab = rgb_to_lab(np.array([hex_to_rgb(hx)]))[0]
        brand_lab.append((hx, lab))

    per_frame_rows = []
    prev_hist = None
    cuts = 0; overlay_frames = 0; brand_hit_frames = 0

    for i, (fp, t, frame) in enumerate(frames):
        hist = frame_histogram(frame)
        is_cut = False
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            is_cut = diff > 0.42
            if is_cut: cuts += 1
        prev_hist = hist

        dom_rgb, counts = kmeans_dominant_colors(frame, k=k_colors)
        dom_lab = rgb_to_lab(dom_rgb)
        hit = False
        if brand_lab:
            for c_lab in dom_lab:
                if min(delta_e(c_lab, b_lab) for _, b_lab in brand_lab) < delta_e_thresh:
                    hit = True; break
        if hit: brand_hit_frames += 1

        text_found = ocr_text(frame) if OCR_AVAILABLE else ""
        has_overlay = len(text_found.strip())>0
        if has_overlay: overlay_frames += 1

        per_frame_rows.append({
            "video": video_path.name,
            "timestamp_s": round(float(t),2),
            "frame_file": str(fp),
            "is_cut": int(is_cut),
            "has_overlay_text": int(has_overlay),
            "overlay_text_sample": text_found.strip()[:120].replace("\n"," ") if has_overlay else "",
            "brand_color_hit": int(hit),
            "top_colors_rgb": ";".join([f"{tuple(map(int,rgb))}" for rgb in dom_rgb[:3]]),
        })

    total_frames = len(frames) if frames else 1
    summary = {
        "video": video_path.name,
        "frames_sampled": total_frames,
        "frame_interval_seconds": interval_s,
        "estimated_cuts": cuts,
        "cuts_per_minute": round(cuts / max(1e-6, (total_frames*interval_s/60)), 2),
        "overlay_frame_percent": round(100.0*overlay_frames/total_frames, 1),
        "brand_match_frame_percent": round(100.0*brand_hit_frames/total_frames, 1),
        "brand_colors_used": ",".join(brand_colors) if brand_colors else "",
    }
    return summary, per_frame_rows

# ---------- Post CSV parsing & disclosure ----------
DISCLOSURE_PATTERNS = [
    r"#ad\b", r"#sponsored\b", r"#paidpartner\b", r"paid partnership", r"paid promotion",
    r"partnered with", r"advertising", r"ad:", r"promo"
]

def get_text_fields(row: Dict[str, Any]) -> str:
    for key in ("desc","description","caption","text","title","video_desc"):
        if key in row and row[key]: return str(row[key])
    return ""

def disclosure_ok_for_row(row: Dict[str, Any]) -> bool:
    txt = (get_text_fields(row) + " " + str(row.get("hashtags",""))).lower()
    return any(re.search(p, txt) for p in DISCLOSURE_PATTERNS)

def read_posts_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def video_id_from_row(r: Dict[str, Any]) -> str:
    # Try common fields
    for k in ("id","video_id","aweme_id","unique_id"):
        if k in r and r[k]: return str(r[k])
    # Fallback: parse from url
    url = r.get("url") or r.get("share_url") or ""
    m = re.search(r"/video/(\d+)", url)
    return m.group(1) if m else ""

# ---------- Scoring ----------
def grade_band(value: float, green_min: float, amber_min: float, higher_is_better=True):
    # returns ("GREEN"/"AMBER"/"RED")
    if higher_is_better:
        if value >= green_min: return "GREEN"
        if value >= amber_min: return "AMBER"
        return "RED"
    else:
        if value <= green_min: return "GREEN"
        if value <= amber_min: return "AMBER"
        return "RED"

def overall_grade(parts: List[str]) -> str:
    # Simple combiner: any RED -> AMBER unless >=2 RED -> RED; all GREEN -> GREEN
    reds = parts.count("RED"); ambers = parts.count("AMBER")
    if reds >= 2: return "RED"
    if reds == 1 or ambers >= 1: return "AMBER"
    return "GREEN"

def main():
    ap = argparse.ArgumentParser(description="TikTok scoring merged with brand-tone frame analysis.")
    ap.add_argument("--posts", required=True, help="Path to TikTok scraper CSV")
    ap.add_argument("--videos", required=True, help="Folder with video files")
    ap.add_argument("--out", default="analysis_out", help="Output folder")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between frames")
    ap.add_argument("--brand_pdf", default=None, help="Path to 'Cheat Sheet only.pdf'")
    ap.add_argument("--brand_json", default=None, help='JSON like {\"colors\":[\"#003E5A\", ...]}')
    ap.add_argument("--delta_e", type=float, default=12.0, help="ΔE76 threshold for brand match")
    ap.add_argument("--kcolors", type=int, default=5, help="K-means dominant colors K")
    # Scoring thresholds (tune as needed)
    ap.add_argument("--th_brand_green", type=float, default=60.0, help="% frames brand-match for GREEN")
    ap.add_argument("--th_brand_amber", type=float, default=30.0, help="% frames brand-match for AMBER")
    ap.add_argument("--th_overlay_green", type=float, default=40.0, help="% frames with overlay text for GREEN")
    ap.add_argument("--th_overlay_amber", type=float, default=15.0, help="% frames with overlay text for AMBER")
    ap.add_argument("--th_cuts_green", type=float, default=12.0, help="cuts/min for GREEN")
    ap.add_argument("--th_cuts_amber", type=float, default=6.0, help="cuts/min for AMBER")
    args = ap.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    # Load brand palette
    brand_colors = load_brand_palette(args.brand_pdf, args.brand_json) or []
    with open(out_root / "brand_colors_detected.json", "w", encoding="utf-8") as f:
        json.dump({"colors": brand_colors}, f, indent=2)
    if not brand_colors:
        print("[!] No brand colors found. Provide --brand_json or parsable --brand_pdf. Continuing without brand scoring.", file=sys.stderr)

    # Load posts
    posts_path = Path(args.posts)
    posts = read_posts_csv(posts_path)

    # Video file lookup (map by id substring or stem match)
    video_dir = Path(args.videos)
    vid_files = {p.stem: p for p in video_dir.rglob("*") if p.suffix.lower() in (".mp4",".mov",".mkv",".avi",".m4v")}
    all_videos = list(vid_files.values())

    # Analyze each video file once, store cache by file path
    analyzed_cache = {}  # path -> (summary, per_frame_rows)
    video_summary_rows = []
    per_frame_rows_all = []

    def analyze_once(p: Path):
        if p in analyzed_cache: return analyzed_cache[p]
        s, pf = analyze_video(p, out_root, args.interval, brand_colors, args.delta_e, args.kcolors)
        analyzed_cache[p] = (s, pf)
        video_summary_rows.append(s)
        per_frame_rows_all.extend(pf)
        return s, pf

    # Iterate posts, try link to a file
    scoring_rows = []
    for row in posts:
        vid_id = video_id_from_row(row)
        candidate = None

        # Match strategy: exact stem, startswith/endswith, contains id
        if vid_id and vid_id in vid_files:
            candidate = vid_files[vid_id]
        else:
            # fuzzy
            for stem, p in vid_files.items():
                if vid_id and (stem.endswith(vid_id) or stem.startswith(vid_id) or vid_id in stem):
                    candidate = p; break

        s = None
        if candidate:
            s, _ = analyze_once(candidate)

        # Disclosure from caption/hashtags:
        disc_ok = disclosure_ok_for_row(row)

        # Build scoring metrics (if no video found, mark N/A)
        brand_pct = s["brand_match_frame_percent"] if (s and brand_colors) else None
        overlay_pct = s["overlay_frame_percent"] if s else None
        cuts_per_min = s["cuts_per_minute"] if s else None

        # Grade components
        brand_grade = "N/A"
        if brand_pct is not None:
            brand_grade = grade_band(brand_pct, args.th_brand_green, args.th_brand_amber, higher_is_better=True)

        overlay_grade = "N/A"
        if overlay_pct is not None:
            overlay_grade = grade_band(overlay_pct, args.th_overlay_green, args.th_overlay_amber, higher_is_better=True)

        cuts_grade = "N/A"
        if cuts_per_min is not None:
            cuts_grade = grade_band(cuts_per_min, args.th_cuts_green, args.th_cuts_amber, higher_is_better=True)

        disclosure_grade = "GREEN" if disc_ok else "RED"

        parts = [g for g in (brand_grade, overlay_grade, cuts_grade, disclosure_grade) if g!="N/A"]
        overall = overall_grade(parts) if parts else "N/A"

        scoring_rows.append({
            "video_id": vid_id or "",
            "video_file": candidate.name if candidate else "",
            "caption_excerpt": get_text_fields(row)[:140].replace("\n"," "),
            "hashtags": row.get("hashtags",""),
            "disclosure_ok": int(disc_ok),
            "disclosure_grade": disclosure_grade,
            "brand_match_frame_percent": brand_pct if brand_pct is not None else "",
            "brand_grade": brand_grade,
            "overlay_frame_percent": overlay_pct if overlay_pct is not None else "",
            "overlay_grade": overlay_grade,
            "cuts_per_minute": cuts_per_min if cuts_per_min is not None else "",
            "cuts_grade": cuts_grade,
            "overall_grade": overall,
            "overall_reason": (
                "Brand tone strong; overlays used; fast pacing; disclosure present" if overall=="GREEN" else
                "Some elements missing or below target" if overall=="AMBER" else
                "Missing disclosure and/or weak brand tone/overlays/pacing"
            )
        })

    # Write outputs
    # 1) Video summaries
    vs_fp = out_root / "video_summary.csv"
    if video_summary_rows:
        with open(vs_fp, "w", newline="", encoding="utf-8") as f:
            fields = list(video_summary_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in video_summary_rows: w.writerow(r)

    # 2) Per-frame details
    pf_fp = out_root / "per_frame_details.csv"
    if per_frame_rows_all:
        with open(pf_fp, "w", newline="", encoding="utf-8") as f:
            fields = ["video","timestamp_s","frame_file","is_cut","has_overlay_text","overlay_text_sample","brand_color_hit","top_colors_rgb"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in per_frame_rows_all: w.writerow(r)

    # 3) Scoring summary with green/amber/red
    sc_fp = out_root / "scoring_summary.csv"
    if scoring_rows:
        with open(sc_fp, "w", newline="", encoding="utf-8") as f:
            fields = list(scoring_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in scoring_rows: w.writerow(r)

    print("\nDone.")
    print(f" Scoring Summary : {sc_fp}")
    if video_summary_rows: print(f" Video Summary    : {vs_fp}")
    if per_frame_rows_all: print(f" Per-frame Details: {pf_fp}")
    print(f" Brand Colors     : {out_root / 'brand_colors_detected.json'}\n")

if __name__ == "__main__":
    main()
