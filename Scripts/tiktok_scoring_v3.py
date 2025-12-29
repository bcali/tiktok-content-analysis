#!/usr/bin/env python3
# tiktok_scoring_v3.py  (Overlay-focused upgrade)
# Adds: video_url, handle, brand columns + composite_score + optional reuse of frames_manifest
# Now with robust overlay detection (OCR + visual) and stronger scoring tie-in.
# Produces: video_summary.csv, per_frame_details.csv, scoring_summary.csv, top_bottom_links.csv, brand_colors_detected.json

import argparse, csv, json, re, sys, os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

# ---------- Optional deps ----------
OCR_AVAILABLE = False
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

# ---------- Color & brand helpers ----------
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
            # accept {"colors":[...]} or {"brand_colors":[...]} etc.
            for k in ("colors","brand_colors","palette"):
                if k in data and isinstance(data[k], list):
                    colors = data[k]
                    break
        except Exception:
            pass
    if not colors and pdf_path and Path(pdf_path).is_file() and PDF_AVAILABLE:
        try:
            text = extract_text(pdf_path)
            hexes = find_hex_codes_in_text(text)
            colors = hexes[:12]
        except Exception:
            pass
    colors = [c.upper() if c.startswith("#") else ("#"+str(c).upper()) for c in colors]
    colors = [c for c in colors if re.match(r"^#[0-9A-F]{6}$", c)]
    return colors

# ---------- URL / handle / brand helpers ----------
URL_LIKE_COLS = ["url","share_url","shareUrl","webVideoUrl","permalink","video_url","canonical_url","href"]

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

def parse_handle_and_id_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    # Expect ... tiktok.com/@HANDLE/video/ID ...
    handle = None; vid = None
    if not url: return handle, vid
    try:
        m = re.search(r"tiktok\.com/@([^/]+)/video/(\d+)", url)
        if m:
            handle, vid = m.group(1), m.group(2)
    except Exception:
        pass
    return handle, vid

def find_any_video_url_in_row(row: Dict[str, Any]) -> Optional[str]:
    for key in URL_LIKE_COLS:
        if key in row and row[key]:
            val = str(row[key]).strip()
            if "/video/" in val and "tiktok.com/@" in val:
                return val
    # fallback: scan all fields
    for v in row.values():
        try:
            s = str(v)
        except Exception:
            continue
        if "tiktok.com/@" in s and "/video/" in s:
            return s
    return None

def sanitize_video_id_raw(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if re.fullmatch(r"\d{8,}", s):
        return s
    if re.search(r"[eE]\+?\d+$", s):
        return None
    runs = re.findall(r"\d{8,}", s)
    if runs:
        runs.sort(key=len, reverse=True)
        return runs[0]
    return None

def canonical_video_url(handle: Optional[str], vid: Optional[str]) -> Optional[str]:
    if not handle or not vid:
        return None
    return f"https://www.tiktok.com/@{handle}/video/{vid}"

def brand_from_handle(handle: str) -> str:
    h = (handle or "").lower()
    if "anantara" in h: return "Anantara"
    if "avani" in h: return "Avani"
    if "nh" in h and "collection" in h: return "NH Collection"
    if h.startswith("nh") or " nh" in h or "nhhotel" in h: return "NH"
    if "tivoli" in h: return "Tivoli"
    if "oaks" in h: return "Oaks"
    return "Other"

# ---------- Overlay detection ----------
def set_tesseract_cmd(tess_cmd: Optional[str]):
    global OCR_AVAILABLE
    if not tess_cmd:
        return
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
        # sanity run
        _ = pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
    except Exception:
        # keep existing state
        pass

def ocr_text(frame_bgr):
    if not OCR_AVAILABLE: return "", 0
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        txt = pytesseract.image_to_string(gray, config="--psm 6")
        txt = txt.strip()
        return txt, len(txt)
    except Exception:
        return "", 0

def detect_overlay_visual(frame_bgr) -> bool:
    """
    Heuristic subtitle/overlay detector:
      - Look at bottom ~40% of the frame
      - Binarize & open to merge characters
      - Count wide, relatively thin connected components (text bands)
    Returns True if we see a plausible text band.
    """
    try:
        h, w = frame_bgr.shape[:2]
        roi = frame_bgr[int(h*0.60):, :]                 # bottom 40%
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # improve contrast a bit
        gray = cv2.equalizeHist(gray)
        # adaptive threshold to handle varied backgrounds
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 8)
        # open to merge letters into bands
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        # find contours
        cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        band_hits = 0
        for c in cnts:
            x,y,ww,hh = cv2.boundingRect(c)
            # look for fairly wide strips typical for captions/overlay bars
            if ww >= int(0.30*w) and 12 <= hh <= int(0.25*h):
                band_hits += 1
                if band_hits >= 1:
                    return True
        return False
    except Exception:
        return False

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
                  brand_colors: List[str], delta_e_thresh=12.0, k_colors=5,
                  overlay_mode="both"):
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
    cuts = 0; overlay_frames = 0; overlay_chars_total = 0; brand_hit_frames = 0
    overlay_visual_frames = 0

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

        # Overlay detection
        txt, char_n = ocr_text(frame) if (overlay_mode in ("ocr","both")) else ("", 0)
        vis = detect_overlay_visual(frame) if (overlay_mode in ("visual","both")) else False
        has_overlay = (len(txt.strip())>0) or vis
        if has_overlay:
            overlay_frames += 1
            overlay_chars_total += char_n
        if vis:
            overlay_visual_frames += 1

        per_frame_rows.append({
            "video": video_path.name,
            "timestamp_s": round(float(t),2),
            "frame_file": str(fp),
            "is_cut": int(is_cut),
            "has_overlay_text": int(len(txt.strip())>0),
            "has_overlay_visual": int(vis),
            "overlay_text_sample": txt.strip()[:120].replace("\n"," ") if len(txt.strip())>0 else "",
            "overlay_char_count": int(char_n),
            "brand_color_hit": int(hit),
            "top_colors_rgb": ";".join([f"{tuple(map(int,rgb))}" for rgb in dom_rgb[:3]]),
        })

    total_frames = len(frames) if frames else 1
    overlay_pct = round(100.0*overlay_frames/total_frames, 1)
    overlay_chars_per_frame = round((overlay_chars_total / max(1, total_frames)), 1)

    summary = {
        "video": video_path.name,
        "frames_sampled": total_frames,
        "frame_interval_seconds": interval_s,
        "estimated_cuts": cuts,
        "cuts_per_minute": round(cuts / max(1e-6, (total_frames*interval_s/60)), 2),
        "overlay_frame_percent": overlay_pct,
        "overlay_chars_per_frame": overlay_chars_per_frame,
        "overlay_visual_frame_percent": round(100.0*overlay_visual_frames/total_frames, 1),
        "brand_match_frame_percent": round(100.0*brand_hit_frames/total_frames, 1),
    }
    return summary, per_frame_rows

def analyze_from_frames(frame_paths_with_ts, brand_colors, delta_e_thresh=12.0, k_colors=5,
                        overlay_mode="both"):
    # frame_paths_with_ts: list of (path, timestamp_float)
    brand_lab = []
    for hx in brand_colors:
        lab = rgb_to_lab(np.array([hex_to_rgb(hx)]))[0]
        brand_lab.append((hx, lab))

    per_frame_rows = []
    prev_hist = None
    cuts = overlay_frames = brand_hit_frames = 0
    overlay_visual_frames = 0
    overlay_chars_total = 0
    last_ts = 0.0

    for i, (fp, t) in enumerate(frame_paths_with_ts):
        frame = cv2.imread(fp)
        if frame is None:
            continue
        hist = frame_histogram(frame)
        is_cut = False
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            is_cut = diff > 0.42
            if is_cut: cuts += 1
        prev_hist = hist

        dom_rgb, _ = kmeans_dominant_colors(frame, k=k_colors)
        dom_lab = rgb_to_lab(dom_rgb)
        hit = False
        if brand_lab:
            for c_lab in dom_lab:
                if min(delta_e(c_lab, b_lab) for _, b_lab in brand_lab) < delta_e_thresh:
                    hit = True; break
        if hit: brand_hit_frames += 1

        txt, char_n = ocr_text(frame) if (overlay_mode in ("ocr","both")) else ("", 0)
        vis = detect_overlay_visual(frame) if (overlay_mode in ("visual","both")) else False
        has_overlay = (len(txt.strip())>0) or vis
        if has_overlay:
            overlay_frames += 1
            overlay_chars_total += char_n
        if vis:
            overlay_visual_frames += 1

        per_frame_rows.append({
            "video": Path(fp).parent.name + ".mp4",
            "timestamp_s": round(float(t),2),
            "frame_file": fp,
            "is_cut": int(is_cut),
            "has_overlay_text": int(len(txt.strip())>0),
            "has_overlay_visual": int(vis),
            "overlay_text_sample": txt.strip()[:120].replace("\n"," ") if len(txt.strip())>0 else "",
            "overlay_char_count": int(char_n),
            "brand_color_hit": int(hit),
            "top_colors_rgb": ";".join([f"{tuple(map(int,rgb))}" for rgb in dom_rgb[:3]]),
        })
        last_ts = t

    total = max(1, len(per_frame_rows))
    duration_min = max(1e-6, last_ts/60.0)
    overlay_pct = round(100.0*overlay_frames/total, 1)
    overlay_chars_per_frame = round((overlay_chars_total / max(1, total)), 1)

    summary = {
        "video": per_frame_rows[0]["video"] if per_frame_rows else "unknown",
        "frames_sampled": total,
        "frame_interval_seconds": None,
        "estimated_cuts": cuts,
        "cuts_per_minute": round(cuts/duration_min, 2) if last_ts>0 else "",
        "overlay_frame_percent": overlay_pct,
        "overlay_chars_per_frame": overlay_chars_per_frame,
        "overlay_visual_frame_percent": round(100.0*overlay_visual_frames/total, 1),
        "brand_match_frame_percent": round(100.0*brand_hit_frames/total, 1),
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
    # Prefer from any URL if present
    any_url = find_any_video_url_in_row(r)
    if any_url:
        _, vid = parse_handle_and_id_from_url(any_url)
        if vid: return vid
    # Else usual fields
    for k in ("id","video_id","aweme_id","unique_id","video/id"):
        if k in r and r[k]:
            vid = sanitize_video_id_raw(str(r[k]))
            if vid: return vid
    # Fallback: parse from url-like fields
    for k in URL_LIKE_COLS:
        if k in r and r[k]:
            _, vid = parse_handle_and_id_from_url(str(r[k]))
            if vid: return vid
    url = r.get("url") or r.get("share_url") or ""
    m = re.search(r"/video/(\d+)", url)
    return m.group(1) if m else ""

def handle_from_row(r: Dict[str, Any]) -> str:
    any_url = find_any_video_url_in_row(r)
    if any_url:
        h, _ = parse_handle_and_id_from_url(any_url)
        if h: return h
    h = extract_handle_from_input(r.get("input"))
    if h: return h
    # Last resort: parse any url-like field
    for k in URL_LIKE_COLS:
        if k in r and r[k]:
            u = str(r[k])
            m = re.search(r"tiktok\.com/@([^/]+)/video/", u)
            if m: return m.group(1)
    return ""

def video_url_from_row(r: Dict[str, Any]) -> str:
    any_url = find_any_video_url_in_row(r)
    if any_url: return any_url
    vid = video_id_from_row(r); h = handle_from_row(r)
    return canonical_video_url(h, vid) or ""

# ---------- Scoring ----------
def grade_band(value: float, green_min: float, amber_min: float, higher_is_better=True):
    if value is None or value == "": return "N/A"
    if higher_is_better:
        if value >= green_min: return "GREEN"
        if value >= amber_min: return "AMBER"
        return "RED"
    else:
        if value <= green_min: return "GREEN"
        if value <= amber_min: return "AMBER"
        return "RED"

def overall_grade(parts: List[str]) -> str:
    reds = parts.count("RED"); ambers = parts.count("AMBER")
    if reds >= 2: return "RED"
    if reds == 1 or ambers >= 1: return "AMBER"
    return "GREEN"

def compute_composite_score(
    brand_pct, overlay_pct, cuts_per_min, th_cuts_green,
    w_brand=0.40, w_overlay=0.30, w_cuts=0.30,
    penalty_no_disclosure=25, disclosure_ok=1,
    penalty_no_overlay=15, overlay_min_pct_for_no_penalty=5.0
):
    # normalize terms to ~[0..1.2]
    b = (brand_pct or 0)/100.0
    o = (overlay_pct or 0)/100.0
    c = 0.0
    if cuts_per_min is not None and th_cuts_green and th_cuts_green > 0:
        c = min((cuts_per_min / th_cuts_green), 1.2)
    score = 100.0 * (w_brand*b + w_overlay*o + w_cuts*c)

    # penalties
    if not disclosure_ok:
        score -= float(penalty_no_disclosure or 0)
    if (overlay_pct is None) or (overlay_pct < float(overlay_min_pct_for_no_penalty or 0)):
        score -= float(penalty_no_overlay or 0)

    return round(max(0.0, min(100.0, score)), 1)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="TikTok scoring v3 with stronger overlay detection and exec-friendly fields.")
    ap.add_argument("--posts", required=True, help="Path to TikTok scraper CSV")
    ap.add_argument("--videos", required=False, help="Folder with video files (if analyzing raw videos)")
    ap.add_argument("--out", default="analysis_out", help="Output folder")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between frames (if extracting)")
    ap.add_argument("--brand_pdf", default=None, help="Path to 'Cheat Sheet only.pdf'")
    ap.add_argument("--brand_json", default=None, help='JSON like {"colors":["#003E5A", ...]}')
    ap.add_argument("--delta_e", type=float, default=12.0, help="ΔE76 threshold for brand match")
    ap.add_argument("--kcolors", type=int, default=5, help="K-means dominant colors K")
    # Scoring thresholds
    ap.add_argument("--th_brand_green", type=float, default=60.0)
    ap.add_argument("--th_brand_amber", type=float, default=30.0)
    ap.add_argument("--th_overlay_green", type=float, default=40.0)
    ap.add_argument("--th_overlay_amber", type=float, default=15.0)
    ap.add_argument("--th_cuts_green", type=float, default=12.0)
    ap.add_argument("--th_cuts_amber", type=float, default=6.0)
    # Composite weights
    ap.add_argument("--w_brand", type=float, default=0.40)
    ap.add_argument("--w_overlay", type=float, default=0.30)
    ap.add_argument("--w_cuts", type=float, default=0.30)
    ap.add_argument("--penalty_no_disclosure", type=float, default=25.0)
    # NEW: overlay penalty & mode
    ap.add_argument("--penalty_no_overlay", type=float, default=15.0, help="Extra penalty if overlay usage is near zero")
    ap.add_argument("--overlay_min_pct_for_no_penalty", type=float, default=5.0, help="Minimum overlay% to avoid the penalty")
    ap.add_argument("--overlay_mode", choices=["ocr","visual","both"], default="both", help="How to detect overlays")
    # Reuse frames
    ap.add_argument("--frames-manifest", dest="frames_manifest", default=None, help="CSV from fetch_tiktok_assets.py to reuse pre-extracted frames")
    # Tesseract path (Windows friendliness)
    ap.add_argument("--tesseract_cmd", default=None, help="Full path to tesseract.exe if not on PATH")
    args = ap.parse_args()

    # Make OCR workable on Windows if path is given
    if args.tesseract_cmd:
        set_tesseract_cmd(args.tesseract_cmd)

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

    # Optional: load frames manifest (video_id -> [(frame_file, ts), ...])
    frames_manifest_map: Dict[str, List[Tuple[str, float]]] = {}
    if args.frames_manifest and Path(args.frames_manifest).is_file():
        with open(args.frames_manifest, "r", encoding="utf-8", errors="ignore") as f:
            r = csv.DictReader(f)
            for row in r:
                vid = row.get("video_id"); fp = row.get("frame_file"); ts = row.get("timestamp_s")
                if vid and fp:
                    try:
                        t = float(ts) if ts not in (None,"") else 0.0
                    except Exception:
                        t = 0.0
                    frames_manifest_map.setdefault(vid, []).append((fp, t))
        for vid in frames_manifest_map:
            frames_manifest_map[vid].sort(key=lambda x: x[1])

    # Video file lookup (map by id substring or stem match) if videos folder provided
    video_dir = Path(args.videos) if args.videos else None
    vid_files = {}
    if video_dir and video_dir.exists():
        vid_files = {p.stem: p for p in video_dir.rglob("*") if p.suffix.lower() in (".mp4",".mov",".mkv",".avi",".m4v")}
    analyzed_cache = {}  # key: video_id or path -> (summary, per_frame_rows)

    video_summary_rows = []
    per_frame_rows_all = []
    scoring_rows = []

    def analyze_path(p: Path):
        if p in analyzed_cache: return analyzed_cache[p]
        s, pf = analyze_video(p, out_root, args.interval, brand_colors, args.delta_e, args.kcolors, args.overlay_mode)
        analyzed_cache[p] = (s, pf)
        video_summary_rows.append(s)
        per_frame_rows_all.extend(pf)
        return s, pf

    def analyze_vidid_from_manifest(vid: str):
        key = f"vid:{vid}"
        if key in analyzed_cache: return analyzed_cache[key]
        s, pf = analyze_from_frames(frames_manifest_map[vid], brand_colors, args.delta_e, args.kcolors, args.overlay_mode)
        analyzed_cache[key] = (s, pf)
        video_summary_rows.append(s)
        per_frame_rows_all.extend(pf)
        return s, pf

    # Iterate posts
    topcalc_rows = []
    for row in posts:
        vid_id = video_id_from_row(row)
        handle = handle_from_row(row)
        brand = brand_from_handle(handle)
        video_url = video_url_from_row(row)
        disc_ok = disclosure_ok_for_row(row)

        s = None
        # Prefer frames manifest if available for this video_id
        if vid_id and vid_id in frames_manifest_map:
            s, _ = analyze_vidid_from_manifest(vid_id)
        else:
            # Try match to a local video file
            candidate = None
            if vid_id and vid_id in vid_files:
                candidate = vid_files[vid_id]
            else:
                for stem, p in vid_files.items():
                    if vid_id and (stem.endswith(vid_id) or stem.startswith(vid_id) or (vid_id in stem)):
                        candidate = p; break
            if candidate:
                s, _ = analyze_path(candidate)

        # Metrics
        brand_pct = s["brand_match_frame_percent"] if (s and brand_colors) else None
        overlay_pct = s.get("overlay_frame_percent") if s else None
        cuts_per_min = s.get("cuts_per_minute") if s else None

        # Grades
        brand_grade = grade_band(brand_pct, args.th_brand_green, args.th_brand_amber, True) if brand_pct is not None else "N/A"
        overlay_grade = grade_band(overlay_pct, args.th_overlay_green, args.th_overlay_amber, True) if overlay_pct is not None else "N/A"
        cuts_grade = grade_band(cuts_per_min, args.th_cuts_green, args.th_cuts_amber, True) if cuts_per_min is not None else "N/A"
        disclosure_grade = "GREEN" if disc_ok else "RED"
        parts = [g for g in (brand_grade, overlay_grade, cuts_grade, disclosure_grade) if g!="N/A"]
        overall = overall_grade(parts) if parts else "N/A"

        # Composite score (0-100) with stronger overlay tie-in
        comp_score = compute_composite_score(
            brand_pct, overlay_pct, cuts_per_min, args.th_cuts_green,
            w_brand=args.w_brand, w_overlay=args.w_overlay, w_cuts=args.w_cuts,
            penalty_no_disclosure=args.penalty_no_disclosure, disclosure_ok=int(bool(disc_ok)),
            penalty_no_overlay=args.penalty_no_overlay,
            overlay_min_pct_for_no_penalty=args.overlay_min_pct_for_no_penalty
        )

        scoring_rows.append({
            "video_id": vid_id or "",
            "handle": handle,
            "brand": brand,
            "video_url": video_url,
            "caption_excerpt": get_text_fields(row)[:140].replace("\n"," "),
            "hashtags": row.get("hashtags",""),
            "disclosure_ok": int(bool(disc_ok)),
            "disclosure_grade": disclosure_grade,
            "brand_match_frame_percent": brand_pct if brand_pct is not None else "",
            "brand_grade": brand_grade,
            "overlay_frame_percent": overlay_pct if overlay_pct is not None else "",
            "overlay_chars_per_frame": s.get("overlay_chars_per_frame","") if s else "",
            "overlay_visual_frame_percent": s.get("overlay_visual_frame_percent","") if s else "",
            "overlay_grade": overlay_grade,
            "cuts_per_minute": cuts_per_min if cuts_per_min is not None else "",
            "cuts_grade": cuts_grade,
            "overall_grade": overall,
            "composite_score": comp_score
        })

        topcalc_rows.append({
            "video_id": vid_id or "",
            "handle": handle,
            "brand": brand,
            "video_url": video_url,
            "overall_grade": overall,
            "brand_match_frame_percent": brand_pct if brand_pct is not None else 0,
            "overlay_frame_percent": overlay_pct if overlay_pct is not None else 0,
            "cuts_per_minute": cuts_per_min if cuts_per_min is not None else 0,
            "disclosure_ok": int(bool(disc_ok)),
            "composite_score": comp_score
        })

    # Write outputs
    if video_summary_rows:
        vs_fp = out_root / "video_summary.csv"
        with open(vs_fp, "w", newline="", encoding="utf-8") as f:
            fields = list(video_summary_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in video_summary_rows: w.writerow(r)

    if per_frame_rows_all:
        pf_fp = out_root / "per_frame_details.csv"
        with open(pf_fp, "w", newline="", encoding="utf-8") as f:
            fields = ["video","timestamp_s","frame_file","is_cut","has_overlay_text","has_overlay_visual","overlay_text_sample","overlay_char_count","brand_color_hit","top_colors_rgb"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in per_frame_rows_all: w.writerow(r)

    if scoring_rows:
        sc_fp = out_root / "scoring_summary.csv"
        with open(sc_fp, "w", newline="", encoding="utf-8") as f:
            fields = list(scoring_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in scoring_rows: w.writerow(r)

        # Top/Bottom 5 links
        try:
            sortable = [r for r in topcalc_rows if isinstance(r.get("composite_score", None), (int,float)) or str(r.get("composite_score","")).replace('.','',1).isdigit()]
            for r in sortable:
                if not isinstance(r["composite_score"], (int,float)):
                    try: r["composite_score"] = float(r["composite_score"])
                    except: r["composite_score"] = 0.0
            top5 = sorted(sortable, key=lambda x: x["composite_score"], reverse=True)[:5]
            bot5 = sorted(sortable, key=lambda x: x["composite_score"])[:5]
            tb_fp = out_root / "top_bottom_links.csv"
            with open(tb_fp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["rank_type","rank","video_id","handle","brand","video_url","composite_score","reason_hint"])
                w.writeheader()
                for i, r in enumerate(top5, 1):
                    reason = "Strong overlay+brand+pacing, disclosed" if r["disclosure_ok"] else "Strong composite but missing disclosure"
                    w.writerow({"rank_type":"TOP","rank":i, **{k:r[k] for k in ["video_id","handle","brand","video_url","composite_score"]}, "reason_hint":reason})
                for i, r in enumerate(bot5, 1):
                    bad = []
                    if (r["brand_match_frame_percent"] or 0) < 30: bad.append("brand tone low")
                    if (r["overlay_frame_percent"] or 0) < 15: bad.append("overlays low")
                    if (r["cuts_per_minute"] or 0) < 6: bad.append("pacing slow")
                    if not r["disclosure_ok"]: bad.append("missing disclosure")
                    reason = ", ".join(bad) if bad else "low composite"
                    w.writerow({"rank_type":"BOTTOM","rank":i, **{k:r[k] for k in ["video_id","handle","brand","video_url","composite_score"]}, "reason_hint":reason})
        except Exception as e:
            print(f"[top_bottom_links] skipped: {e}", file=sys.stderr)

    print("\nDone.")
    if scoring_rows:
        print(f" Scoring Summary : {out_root / 'scoring_summary.csv'}")
        print(f" Top/Bottom Links: {out_root / 'top_bottom_links.csv'}")
        if video_summary_rows: print(f" Video Summary    : {out_root / 'video_summary.csv'}")
        if per_frame_rows_all: print(f" Per-frame Details: {out_root / 'per_frame_details.csv'}")
    print(f" Brand Colors     : {out_root / 'brand_colors_detected.json'}")
    print(f" OCR available    : {OCR_AVAILABLE}")
    print()

if __name__ == "__main__":
    main()
