#!/usr/bin/env python3
"""
TikTok Hotel Audit — Scoring v1.1 (updated)
- Cover OCR (optional) + transcript-powered detectors
- New categories: Narrative & CTA (0–10), Accessibility & Localization (0–5)
- FIX: Proper dtypes for new columns to remove FutureWarning

Inputs:
  --posts       One or more post CSVs (exported from your scraper)
  --overview    Profile CSV (followers) [optional]
  --comments    Comments CSV [optional]  (not required; sentiment kept simple)
  --covers_dir  Folder with cover images: <video_id>.(jpg|png|webp) [optional]
  --config      JSON config (brand hashtags, banned terms, pillars, etc.) [optional]
  --outdir      Output directory [default: .]
  --no-ocr      Disable OCR even if Tesseract/OpenCV are available

Outputs:
  posts_scored.csv, hotels_summary.csv, exec_summary.md

Dependencies (minimum):
  python -m pip install --user pandas numpy pillow langdetect

Optional OCR (choose one):
  A) Tesseract OCR + pytesseract
     - Install Tesseract on Windows:
         winget install -e --id UB-Mannheim.TesseractOCR   (or)   winget install -e --id tesseract.ocr
     - Then:
         python -m pip install --user pytesseract
  B) OpenCV fallback for text-area proxy (no text content):
         python -m pip install --user opencv-python
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------- Optional libraries --------
try:
    from PIL import Image
except Exception:
    Image = None

_HAS_TESS = False
try:
    import pytesseract  # requires system Tesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

_HAS_CV2 = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

_HAS_LANGDETECT = False
try:
    from langdetect import detect
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False


# ---------------- Default Config ---------------- #
DEFAULT_CONFIG = {
    "brand_hashtags": [
        "anantara", "anantarahotels", "avani", "avanihotels", "minorhotels",
        "nhcollection", "niyamamaldives"
    ],
    "location_hint_keywords": [
        "bangkok","chiangmai","phuket","samui","dubai","abu dhabi","doha",
        "bali","maldives","mauritius","gaborone","windhoek","huahin",
        "pattaya","salalah","angkor","quynhon","khaolak","luangprabang","santorini"
    ],
    "competitor_terms": [
        "marriott","hilton","hyatt","accor","four seasons","shangri-la","ritz-carlton","sofitel","intercontinental"
    ],
    "disclosure_terms": [
        "#ad","sponsored","paid partnership","paid-partnership","in partnership with","ad"
    ],
    "cta_terms": [
        "book","reserve","link in bio","swipe","tap","call","dm","visit our website","visit the website","visit website","enquire","inquire"
    ],
    "claim_terms": [
        "guarantee","best price","best-rate","exclusive","limited time","% off","discount","free stay","complimentary stay","2-for-1","all inclusive","all-inclusive"
    ],
    "currency_symbols": ["฿","$","€","£","AED","USD","THB","QAR","OMR","VND","₫","₭","₨","₺","R","ZAR","MUR"],
    "hashtag_min": 2,
    "hashtag_max": 12,
    "caption_min_chars": 10,
    "caption_max_chars": 140,
    "cadence_target_posts_per_week": 2,
    "preferred_langs_by_country": {
        "TH": ["th","en"], "AE": ["ar","en"], "QA": ["ar","en"], "OM": ["ar","en"],
        "VN": ["vi","en"], "KH": ["km","en"], "LA": ["lo","en"], "ID": ["id","en"],
        "MV": ["dv","en"], "ZA": ["en"], "BW": ["en"], "NA": ["en"], "MU": ["en","fr"],
        "GR": ["el","en"], "MZ": ["pt","en"], "ZM": ["en"]
    },
    "pillars": {
        "rooms": ["suite","room","villa","bedroom","balcony","ocean view","garden view","pool villa"],
        "fnb": ["restaurant","breakfast","brunch","dinner","bar","cocktail","wine","buffet","chef","menu"],
        "spa": ["spa","massage","wellness","yoga","sala","steam","sauna","treatment"],
        "experiences": ["tour","excursion","sunset cruise","snorkel","diving","cooking class","temple","market","safari"],
        "meetings": ["meeting","conference","event","wedding","banquet","ballroom","mice"],
        "family": ["kids","family","child","children","babysitting","playground","family-friendly"]
    }
}


# ---------------- Helpers ---------------- #
def extract_handle_from_url(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    url = url.strip()
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        parts = [x for x in p.path.split("/") if x]
        for part in parts:
            if part.startswith("@"):
                return part[1:].lower()
        if parts:
            return parts[-1].lower()
    except Exception:
        pass
    return None


def collect_hashtags(row: pd.Series) -> List[str]:
    tags = []
    for c in row.index:
        cl = str(c).lower()
        if cl.startswith("hashtags/") and cl.endswith("/name"):
            v = row[c]
            if isinstance(v, str) and v.strip():
                tags.append(v.strip().lstrip("#").lower())
    return tags


def pick_caption_text(row: pd.Series) -> str:
    for key in ["desc", "text", "title", "caption"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    return ""


def detect_non_latin(text: str) -> bool:
    return any(ord(ch) > 127 for ch in (text or ""))


def build_video_url(handle: str, vid: str) -> Optional[str]:
    if not handle or not vid:
        return None
    return f"https://www.tiktok.com/@{handle}/video/{vid}"


def find_columns(columns: List[str], keywords: List[str]) -> List[str]:
    keys = [k.lower() for k in keywords]
    out = []
    for c in columns:
        lc = c.lower()
        if any(k in lc for k in keys):
            out.append(c)
    return out


def get_transcript_text(row: pd.Series, transcript_cols: List[str]) -> str:
    for c in transcript_cols:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                import ast
                lst = ast.literal_eval(v)
                if isinstance(lst, list):
                    return " ".join(str(x) for x in lst if x)
            except Exception:
                pass
    return ""


def parse_duration_seconds(row: pd.Series, duration_cols: List[str]) -> Optional[float]:
    for c in duration_cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        try:
            val = float(s)
            if val > 1e4:
                return val / 1000.0
            return val
        except Exception:
            m = re.match(r"^\s*(\d+):(\d{1,2})\s*$", s)
            if m:
                return int(m.group(1)) * 60 + int(m.group(2))
    return None


# ---------------- OCR & Cover Metrics ---------------- #
def get_cover_path(covers_dir: Path, vid: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = covers_dir / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def cover_basic_metrics(img_path: Path) -> Tuple[Dict[str, float], List[str]]:
    metrics, flags = {}, []
    if Image is None or not img_path or not img_path.exists():
        return metrics, flags
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            metrics["width"], metrics["height"] = w, h
            metrics["aspect_ratio"] = round(w / h, 3) if h else None
            gray = im.convert("L")
            hist = gray.histogram()
            total = sum(hist)
            mean = sum(i * hist[i] for i in range(256)) / max(total, 1)
            metrics["brightness_mean"] = round(mean, 2)
            near_black = sum(hist[i] for i in range(0, 16)) / max(total, 1)
            near_white = sum(hist[i] for i in range(240, 256)) / max(total, 1)
            metrics["pct_near_black"] = round(near_black * 100, 2)
            metrics["pct_near_white"] = round(near_white * 100, 2)
            if mean < 40: flags.append("cover_too_dark")
            if mean > 220: flags.append("cover_too_bright")
            if near_black > 0.25: flags.append("cover_large_dark_regions")
            if near_white > 0.25: flags.append("cover_large_white_regions")
            if metrics["aspect_ratio"] and (metrics["aspect_ratio"] < 0.5 or metrics["aspect_ratio"] > (9/16)*4):
                flags.append("cover_unusual_aspect_ratio")
    except Exception:
        flags.append("cover_unreadable")
    return metrics, flags


def ocr_cover(img_path: Path, no_ocr: bool=False) -> Tuple[str, float, Optional[float]]:
    """
    Returns (text, text_area_pct, min_text_contrast).
    If OCR is disabled/unavailable, returns ("", nan, None).
    """
    if no_ocr or not img_path or not img_path.exists() or Image is None:
        return "", float("nan"), None

    text = ""
    text_area_pct = float("nan")
    min_contrast = None

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            area = w * h

            if _HAS_TESS:
                data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
                n = len(data.get("text", []))
                total_box_area = 0
                contrasts = []
                gray_np = np.array(im.convert("L"))

                for i in range(n):
                    txt = data["text"][i] or ""
                    conf_raw = data.get("conf", ["-1"]*n)[i]
                    try:
                        conf = float(conf_raw)
                    except Exception:
                        conf = -1.0
                    if conf < 40 or len(txt.strip()) == 0:
                        continue
                    x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    total_box_area += max(0, bw*bh)

                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(w, x+bw), min(h, y+bh)
                    inside = gray_np[y0:y1, x0:x1]
                    if inside.size == 0:
                        continue
                    inside_mean = float(inside.mean())

                    bx0, by0 = max(0, x-1), max(0, y-1)
                    bx1, by1 = min(w, x+bw+1), min(h, y+bh+1)
                    border = gray_np[by0:by1, bx0:bx1]
                    if border.size == 0:
                        continue
                    border_mean = float(border.mean())
                    contrast = abs(inside_mean - border_mean)
                    contrasts.append(contrast)

                    text += (" " + txt)

                if area > 0:
                    text_area_pct = (total_box_area / area) * 100.0
                if contrasts:
                    min_contrast = float(min(contrasts))

            else:
                if _HAS_CV2:
                    im_np = np.array(im.convert("L"))
                    edges = cv2.Canny(im_np, 80, 200)
                    edge_density = edges.sum() / 255.0
                    text_area_pct = min(50.0, (edge_density / (im_np.size/20.0)) * 100.0)
                text = ""
                min_contrast = None
    except Exception:
        return "", float("nan"), None

    return text.strip(), text_area_pct, min_contrast


# ---------------- Detectors ---------------- #
HOOK_PATTERNS = [
    r"\bdid you know\b", r"\b(top|best)\s+\d+\b", r"\b\d+\s+things\b",
    r"\bhere'?s why\b", r"\bthis (week|weekend|month)\b", r"\blet'?s\b",
    r"\byou\b", r"\bsecret\b", r"\bhidden gem\b"
]
CTA_PATTERNS = [
    r"\bbook\b", r"\breserve\b", r"\blink in bio\b", r"\bswipe\b", r"\btap\b",
    r"\bcall\b", r"\bdm\b", r"\bvisit (our|the)?\s*website\b", r"\benquir", r"\binquir"
]
DISCLOSE_PATTERNS = [
    r"#ad\b", r"\bsponsored\b", r"paid[-\s]*partnership", r"\bin partnership with\b"
]

def has_any(patterns: List[str], text: str) -> bool:
    t = text.lower() if isinstance(text, str) else ""
    return any(re.search(p, t) for p in patterns)

def detect_pillars(text: str, tags: List[str], cfg: dict) -> List[str]:
    t = (text or "").lower() + " " + " ".join(tags or [])
    found = []
    for k, kws in cfg.get("pillars", {}).items():
        if any(kw in t for kw in kws):
            found.append(k)
    return sorted(set(found))


# ---------------- Scoring ---------------- #
def score_brand_safety(fulltext: str, cover_flags: List[str], cfg: dict, is_ad_or_commerce: bool, disclosure_found: bool) -> Tuple[int, List[str]]:
    base = 35
    flags = []
    t = (fulltext or "").lower()

    severe_terms = ["explicit","nsfw","sexual","nudity","weapon","gun","racist","hate","terror","bomb","drugs","casino","gamble"]
    if any(w in t for w in severe_terms):
        flags.append("safety:severe_term")
        return 0, flags

    if any(w in t for w in cfg.get("competitor_terms", [])):
        base -= 6
        flags.append("safety:competitor_mention")

    currency_hit = any(sym.lower() in t for sym in [s.lower() for s in cfg.get("currency_symbols", [])])
    pct_hit = bool(re.search(r"\d+\s*%|%\s*off", t))
    claim_hit = any(kw in t for kw in [x.lower() for x in cfg.get("claim_terms", [])]) or currency_hit or pct_hit
    if is_ad_or_commerce and claim_hit and not disclosure_found:
        base -= 10
        flags.append("safety:claim_no_disclosure")

    for f in cover_flags:
        if f in ("cover_too_dark","cover_too_bright"): base -= 4
        elif f in ("cover_large_dark_regions","cover_large_white_regions","cover_unusual_aspect_ratio","cover_unreadable"): base -= 2
        flags.append(f"safety:cover:{f}")

    return max(0, min(35, int(base))), flags


def score_alignment(tags: List[str], fulltext: str, cfg: dict) -> Tuple[int, List[str]]:
    base, flags = 0, []
    tagset = set((tags or []))
    brand_hits = [t for t in cfg["brand_hashtags"] if t in tagset or t in fulltext]
    if brand_hits: base += min(8, 2*len(brand_hits))
    else: flags.append("alignment:missing_brand_hashtag")
    loc_hits = [k for k in cfg["location_hint_keywords"] if k in tagset or k in fulltext]
    if loc_hits: base += min(7, 2*len(loc_hits))
    else: flags.append("alignment:missing_location_hint")
    return max(0, min(15, int(base))), flags


def score_content_craft(tags: List[str], caption: str, cover_metrics: Dict[str, float]) -> Tuple[int, List[str]]:
    base, flags = 0, []
    if len(tags) < DEFAULT_CONFIG["hashtag_min"]: flags.append("craft:too_few_hashtags")
    elif len(tags) > DEFAULT_CONFIG["hashtag_max"]: flags.append("craft:too_many_hashtags")
    else: base += 4

    clen = len(caption or "")
    if DEFAULT_CONFIG["caption_min_chars"] <= clen <= DEFAULT_CONFIG["caption_max_chars"]: base += 3
    else: flags.append("craft:caption_length_out_of_range")

    bright = cover_metrics.get("brightness_mean")
    if bright is not None and 60 <= bright <= 200: base += 1

    text_area = cover_metrics.get("text_area_pct")
    if text_area is not None and not (isinstance(text_area, float) and math.isnan(text_area)):
        if text_area <= 25.0: base += 1
        elif text_area > 40.0: flags.append("craft:cover_overtexted")

    min_contrast = cover_metrics.get("min_text_contrast")
    if min_contrast is not None:
        if min_contrast >= 35.0: base += 1
        else: flags.append("craft:low_text_contrast")

    return max(0, min(10, int(base))), flags


def score_narrative_cta(hook_ok: bool, cta_ok: bool) -> Tuple[int, List[str]]:
    base, flags = 0, []
    if hook_ok: base += 6
    else: flags.append("narrative:weak_hook")
    if cta_ok: base += 4
    else: flags.append("narrative:missing_or_unclear_cta")
    return max(0, min(10, int(base))), flags


def score_accessibility_localization(has_captions: bool, lang_code: Optional[str], loc_country: str, caption: str, transcript: str) -> Tuple[int, List[str]]:
    base, flags = 0, []
    if has_captions: base += 2
    else: flags.append("access:no_captions")

    txt = (transcript or "") + " " + (caption or "")
    detected = None
    if _HAS_LANGDETECT:
        try:
            if txt.strip():
                detected = detect(txt)
        except Exception:
            detected = None

    prefs = DEFAULT_CONFIG.get("preferred_langs_by_country", {})
    allowed = prefs.get(str(loc_country).upper(), [])
    if detected:
        if allowed and detected in allowed:
            base += 3
        elif detected == "en" and (not allowed or "en" in allowed):
            base += 2
        else:
            base += 1; flags.append(f"access:lang_mismatch:{detected}")
    else:
        if detect_non_latin(txt):
            base += 2
        else:
            base += 2  # assume English ok

    return max(0, min(5, int(base))), flags


def score_engagement_percentile(er_pct: float, share_pct: float) -> Tuple[int, List[str]]:
    p = max(er_pct or 0.0, share_pct or 0.0)
    if p >= 90: return 20, []
    if p >= 80: return 18, []
    if p >= 70: return 16, []
    if p >= 60: return 14, []
    if p >= 50: return 12, []
    if p >= 40: return 10, []
    if p >= 30: return 8, []
    if p >= 20: return 6, []
    if p >= 10: return 4, []
    return 2, ["engagement:low"]


def score_cadence(posts_per_week: Optional[float], days_since_last: Optional[float], target_ppw: float = 2.0) -> Tuple[int, List[str]]:
    flags, base = [], 0
    if posts_per_week is None or days_since_last is None:
        return 5, ["cadence:insufficient_time_data"]
    if posts_per_week >= target_ppw: base += 6
    elif posts_per_week >= 1.0: base += 4; flags.append("cadence:below_target")
    elif posts_per_week > 0: base += 2; flags.append("cadence:well_below_target")
    else: flags.append("cadence:no_recent_posts")
    if days_since_last <= 7: base += 4
    elif days_since_last <= 14: base += 3
    elif days_since_last <= 21: base += 2; flags.append("cadence:stale_gt_14d")
    else: base += 1; flags.append("cadence:stale_gt_21d")
    return max(0, min(10, int(base))), flags


def percentile_rank(values: List[float], x: float) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return 50.0
    vals = sorted(vals)
    below = sum(1 for v in vals if v <= x)
    return 100.0 * below / len(vals)


# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser(description="Score TikTok hotel posts (v1.1: OCR + transcript).")
    ap.add_argument("--posts", nargs="+", required=True, help="One or more post CSV files.")
    ap.add_argument("--overview", default="", help="Overview/profile CSV (followers).")
    ap.add_argument("--comments", default="", help="Comments CSV (optional).")
    ap.add_argument("--covers_dir", default="covers", help="Folder with <video_id>.(jpg|png|webp).")
    ap.add_argument("--config", default="", help="Config JSON path.")
    ap.add_argument("--outdir", default=".", help="Output directory.")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR processing.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    covers_dir = Path(args.covers_dir)

    # Config
    cfg = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        for k, v in user_cfg.items():
            cfg[k] = v

    # Posts
    frames = []
    for p in args.posts:
        pth = Path(p)
        if not pth.exists():
            print(f"[WARN] missing posts file: {pth}")
            continue
        frames.append(pd.read_csv(pth, dtype=str, low_memory=False))
    if not frames:
        raise SystemExit("No readable posts CSVs.")
    posts = pd.concat(frames, ignore_index=True, sort=False)

    # Normalize core columns
    posts["id"] = posts["id"].apply(lambda x: re.search(r"(\d+)", str(x)).group(1) if pd.notna(x) and re.search(r"(\d+)", str(x)) else None)
    posts["handle"] = posts["input"].apply(extract_handle_from_url) if "input" in posts.columns else None

    # Numbers
    for col in ["playCount", "diggCount", "commentCount", "shareCount"]:
        posts[col] = pd.to_numeric(posts[col], errors="coerce") if col in posts.columns else np.nan

    posts["eng_rate"] = (posts["diggCount"].fillna(0) + posts["commentCount"].fillna(0) + posts["shareCount"].fillna(0)) / posts["playCount"].replace({0: np.nan})
    posts["share_rate"] = posts["shareCount"].fillna(0) / posts["playCount"].replace({0: np.nan})

    # Caption & hashtags
    posts["hashtags_list"] = posts.apply(collect_hashtags, axis=1)
    posts["caption"] = posts.apply(pick_caption_text, axis=1)

    # Location & time
    posts["loc_country"] = posts["locationMeta/countryCode"] if "locationMeta/countryCode" in posts.columns else ""
    posts["loc_city"] = posts["locationMeta/city"] if "locationMeta/city" in posts.columns else ""

    from datetime import datetime, timezone, timedelta
    def parse_dt(row):
        if "createTimeISO" in posts.columns and isinstance(row.get("createTimeISO"), str):
            try:
                return pd.to_datetime(row["createTimeISO"], utc=True, errors="coerce")
            except Exception:
                pass
        ts = row.get("createTime")
        try:
            s = float(ts)
            if s > 1e12: s /= 1000.0
            return datetime.fromtimestamp(s, tz=timezone.utc)
        except Exception:
            return pd.NaT
    posts["create_dt"] = posts.apply(parse_dt, axis=1)

    # Overview (followers)
    followers = {}
    if args.overview and Path(args.overview).exists():
        ov = pd.read_csv(args.overview, dtype=str, low_memory=False)
        ov["handle"] = ov["input"].apply(extract_handle_from_url) if "input" in ov.columns else None
        if "authorMeta/fans" in ov.columns:
            ov["followers"] = pd.to_numeric(ov["authorMeta/fans"], errors="coerce")
            followers = dict(ov[["handle","followers"]].dropna().values)
    posts["followers"] = posts["handle"].map(followers) if followers else np.nan

    # Percentiles within handle
    er_pct_list, sh_pct_list = [], []
    for h, grp in posts.groupby("handle"):
        ers = grp["eng_rate"].astype(float).replace([np.inf,-np.inf], np.nan).tolist()
        shs = grp["share_rate"].astype(float).replace([np.inf,-np.inf], np.nan).tolist()
        for idx, row in grp.iterrows():
            er, sh = row.get("eng_rate"), row.get("share_rate")
            er_pct = percentile_rank(ers, er) if pd.notna(er) else 50.0
            sh_pct = percentile_rank(shs, sh) if pd.notna(sh) else 50.0
            er_pct_list.append((idx, er_pct)); sh_pct_list.append((idx, sh_pct))
    posts["er_pct_within_handle"] = posts.index.to_series().map(pd.Series(dict(er_pct_list)))
    posts["share_pct_within_handle"] = posts.index.to_series().map(pd.Series(dict(sh_pct_list)))

    # Transcript/captions columns auto-detect
    tx_cols = find_columns(posts.columns.tolist(), ["transcript","transcription","asr","auto","speech","captions","caption_text","sticker","textonscreen"])
    dur_cols = find_columns(posts.columns.tolist(), ["duration","video/duration","videoDuration","length"])
    lang_col = "textLanguage" if "textLanguage" in posts.columns else None

    # Cadence per handle
    now = datetime.now(timezone.utc)
    posts["_days_since"] = np.nan
    posts["_ppw"] = np.nan
    if posts["create_dt"].notna().any():
        for h, grp in posts.groupby("handle"):
            g = grp.sort_values("create_dt")
            last_dt = g["create_dt"].dropna().max()
            days_since = (now - last_dt).total_seconds() / 86400.0 if pd.notna(last_dt) else np.nan
            window_start = now - timedelta(days=56)
            cnt = g.loc[g["create_dt"] >= window_start, "id"].nunique()
            ppw = cnt / 8.0
            posts.loc[posts["handle"] == h, "_days_since"] = days_since
            posts.loc[posts["handle"] == h, "_ppw"] = ppw

    # ==== Proper dtypes for new columns (fix FutureWarning) ====
    obj_cols   = ["cover_text","transcript_text","transcript_lang","pillars","claim_flags"]
    float_cols = ["cover_text_area_pct","cover_min_text_contrast","wpm"]
    int_cols   = ["transcript_words"]
    bool_cols  = ["hook_ok","cta_ok","ad_disclosure_ok","is_ad_or_commerce"]

    for c in obj_cols:
        posts[c] = pd.Series(index=posts.index, dtype="object")
    for c in float_cols:
        posts[c] = pd.Series(index=posts.index, dtype="float64")
    for c in int_cols:
        posts[c] = pd.Series(index=posts.index, dtype="Int64")      # nullable int
    for c in bool_cols:
        posts[c] = pd.Series(index=posts.index, dtype="boolean")    # nullable boolean
    # ==========================================================

    # Main loop
    for i, r in posts.iterrows():
        vid = r.get("id")
        caption = r.get("caption") or ""
        tags = r.get("hashtags_list") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        tagset = [str(x).lower() for x in tags]

        # Transcript & duration
        transcript = get_transcript_text(r, tx_cols) if tx_cols else ""
        dur_s = parse_duration_seconds(r, dur_cols) if dur_cols else None
        words = len(re.findall(r"\b\w+\b", transcript)) if transcript else 0
        wpm = (words / (dur_s/60.0)) if transcript and dur_s and dur_s > 0 else None

        # Language
        tlang = r.get(lang_col) if lang_col else None
        if not tlang and _HAS_LANGDETECT:
            try:
                sample = ((transcript or "") + " " + caption)[:5000]
                tlang = detect(sample) if sample.strip() else None
            except Exception:
                tlang = None

        # Cover metrics + OCR
        cover_flags = []
        cover_metrics, cf = cover_basic_metrics(get_cover_path(covers_dir, vid))
        cover_flags.extend(cf)
        ctext, tarea, mcontrast = ocr_cover(get_cover_path(covers_dir, vid), no_ocr=args.no_ocr)
        cover_metrics["text_area_pct"] = tarea
        cover_metrics["min_text_contrast"] = mcontrast

        # Fulltext for scanners
        fulltext = " ".join([
            caption.lower(),
            " ".join("#"+t for t in tagset),
            (transcript or "").lower(),
            (ctext or "").lower()
        ])

        # Ad/commerce + disclosure
        is_ad_or_commerce = bool(str(r.get("isAd", "")).lower() == "true" or str(r.get("authorMeta/commerceUserInfo/commerceUser", "")).lower() == "true")
        disclosure_found = has_any(DISCLOSE_PATTERNS, fulltext)

        # Hook & CTA
        first_text = (transcript or caption or "").lower()
        first_12 = " ".join(first_text.split()[:12])
        hook_ok = has_any(HOOK_PATTERNS, first_12)
        cta_ok = has_any(CTA_PATTERNS, fulltext)

        # Pillars
        pillars = detect_pillars(transcript + " " + caption, tagset, cfg)

        # Captions present?
        has_captions = bool(transcript.strip())

        # Scores
        bs, bs_flags = score_brand_safety(fulltext, cover_flags, cfg, is_ad_or_commerce, disclosure_found)
        align, al_flags = score_alignment(tagset, fulltext, cfg)
        craft, cr_flags = score_content_craft(tagset, caption, {
            "brightness_mean": cover_metrics.get("brightness_mean"),
            "text_area_pct": cover_metrics.get("text_area_pct"),
            "min_text_contrast": cover_metrics.get("min_text_contrast"),
        })
        narr, nr_flags = score_narrative_cta(bool(hook_ok), bool(cta_ok))
        accs, ac_flags = score_accessibility_localization(has_captions, tlang, str(r.get("loc_country") or ""), caption, transcript)

        erp = r.get("er_pct_within_handle") if pd.notna(r.get("er_pct_within_handle")) else 50.0
        shp = r.get("share_pct_within_handle") if pd.notna(r.get("share_pct_within_handle")) else 50.0
        eng, eg_flags = score_engagement_percentile(erp, shp)

        cad, cd_flags = score_cadence(
            posts_per_week=(r.get("_ppw") if pd.notna(r.get("_ppw")) else None),
            days_since_last=(r.get("_days_since") if pd.notna(r.get("_days_since")) else None),
            target_ppw=cfg.get("cadence_target_posts_per_week", 2.0)
        )

        flags = []
        for lst in [bs_flags, al_flags, cr_flags, nr_flags, ac_flags, eg_flags, cd_flags]:
            if lst: flags.extend(lst)

        total = bs + align + eng + craft + narr + accs + cad

        # Assign values (now dtypes are correct)
        posts.at[i, "cover_text"] = ctext
        posts.at[i, "cover_text_area_pct"] = round(tarea, 2) if tarea == tarea else np.nan
        posts.at[i, "cover_min_text_contrast"] = round(mcontrast, 1) if mcontrast is not None else np.nan
        posts.at[i, "hook_ok"] = bool(hook_ok)
        posts.at[i, "cta_ok"] = bool(cta_ok)
        posts.at[i, "ad_disclosure_ok"] = bool(disclosure_found or not is_ad_or_commerce)
        posts.at[i, "is_ad_or_commerce"] = bool(is_ad_or_commerce)
        posts.at[i, "transcript_text"] = transcript
        posts.at[i, "transcript_words"] = int(words)
        posts.at[i, "wpm"] = round(wpm, 1) if wpm else np.nan
        posts.at[i, "transcript_lang"] = tlang if tlang else ""
        posts.at[i, "pillars"] = ",".join(pillars)
        posts.at[i, "claim_flags"] = ",".join([
            flag for flag, cond in [
                ("currency", any(sym.lower() in fulltext for sym in [s.lower() for s in DEFAULT_CONFIG["currency_symbols"]])),
                ("percent_off", bool(re.search(r"\d+\s*%|%\s*off", fulltext))),
                ("claims", any(k in fulltext for k in [x.lower() for x in DEFAULT_CONFIG["claim_terms"]]))
            ] if cond
        ])

        posts.at[i, "score_brand_safety"] = bs
        posts.at[i, "score_alignment"] = align
        posts.at[i, "score_engagement"] = eng
        posts.at[i, "score_craft"] = craft
        posts.at[i, "score_narrative_cta"] = narr
        posts.at[i, "score_accessibility"] = accs
        posts.at[i, "score_cadence"] = cad
        posts.at[i, "score_total"] = total
        posts.at[i, "flags"] = ";".join(flags)

    def rag(row):
        if row["score_brand_safety"] == 0 or row["score_total"] < 50: return "RED"
        if row["score_total"] < 75: return "AMBER"
        return "GREEN"
    posts["RAG"] = posts.apply(rag, axis=1)

    # Save post-level
    out_cols = [
        "id","handle","create_dt","playCount","diggCount","commentCount","shareCount",
        "eng_rate","share_rate","er_pct_within_handle","share_pct_within_handle",
        "score_brand_safety","score_alignment","score_engagement","score_craft",
        "score_narrative_cta","score_accessibility","score_cadence",
        "score_total","RAG","flags",
        "caption","hashtags_list","loc_city","loc_country","followers",
        "is_ad_or_commerce","ad_disclosure_ok","claim_flags","pillars",
        "hook_ok","cta_ok","transcript_lang","transcript_words","wpm",
        "cover_text","cover_text_area_pct","cover_min_text_contrast"
    ]
    posts[out_cols].to_csv(outdir / "posts_scored.csv", index=False)

    # Hotel summary
    def pct(series, cond):
        s = series.dropna()
        return 100.0 * (cond.sum()/len(s)) if len(s) else 0.0

    def has_brand_tags(tags):
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        if not isinstance(tags, list):
            return False
        return any(t in DEFAULT_CONFIG["brand_hashtags"] for t in [str(x).lower() for x in tags])

    hotels = []
    for h, grp in posts.groupby("handle"):
        if not h: continue
        avg_score = float(grp["score_total"].mean()) if len(grp) else np.nan
        count = int(grp["id"].nunique())
        greens = int((grp["RAG"] == "GREEN").sum())
        ambers = int((grp["RAG"] == "AMBER").sum())
        reds = int((grp["RAG"] == "RED").sum())
        brand_pct = pct(grp["hashtags_list"], grp["hashtags_list"].apply(has_brand_tags))
        captions_pct = pct(grp["transcript_text"], grp["transcript_text"].apply(lambda x: isinstance(x, str) and x.strip() != ""))
        disclosure_ok_pct = pct(grp["ad_disclosure_ok"], grp["ad_disclosure_ok"] == True)
        hotels.append({
            "handle": h,
            "posts_count": count,
            "avg_score": round(avg_score, 2) if not np.isnan(avg_score) else None,
            "greens": greens, "ambers": ambers, "reds": reds,
            "brand_hashtag_pct": round(brand_pct, 1),
            "captions_present_pct": round(captions_pct, 1),
            "disclosure_ok_pct": round(disclosure_ok_pct, 1),
        })
    hotels_df = pd.DataFrame(hotels).sort_values(by=["avg_score","posts_count"], ascending=[False, False])
    hotels_df.to_csv(outdir / "hotels_summary.csv", index=False)

    # Exec summary
    lines = []
    lines.append("# TikTok Hotel Audit — Executive Summary (v1.1)\n")
    lines.append(f"- Total posts scored: {len(posts)}")
    lines.append(f"- Total hotels: {hotels_df['handle'].nunique() if not hotels_df.empty else 0}")
    if not hotels_df.empty:
        top = hotels_df.head(10)
        lines.append("\n## Top Hotels (by average score)")
        for _, r in top.iterrows():
            lines.append(f"- **@{r['handle']}** — avg {r['avg_score']}, posts {r['posts_count']} (G/A/R: {r['greens']}/{r['ambers']}/{r['reds']})")

    risky = posts.sort_values(by=["score_total"]).head(10)
    if not risky.empty:
        lines.append("\n## Riskiest Posts (review)")
        for _, r in risky.iterrows():
            url = build_video_url(r["handle"], r["id"]) or ""
            lines.append(f"- @{r['handle']} — score {r['score_total']}: {url}  \n  Flags: {r['flags']}")

    best = posts.sort_values(by=["score_total"], ascending=False).head(10)
    if not best.empty:
        lines.append("\n## Best Posts (replicate)")
        for _, r in best.iterrows():
            url = build_video_url(r["handle"], r["id"]) or ""
            lines.append(f"- @{r['handle']} — score {r['score_total']}: {url}")

    lines.append("\n## Pillar Mix (what’s resonating)")
    pill_counts = posts["pillars"].dropna().str.split(",").explode().value_counts()
    if not pill_counts.empty:
        for k, v in pill_counts.items():
            if k: lines.append(f"- {k}: {int(v)} posts")

    disc_issues = posts[(posts["is_ad_or_commerce"] == True) & (posts["ad_disclosure_ok"] == False)]
    lines.append(f"\n- Posts needing disclosure fixes: {len(disc_issues)}")

    (outdir / "exec_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("[OK] Wrote posts_scored.csv, hotels_summary.csv, exec_summary.md")


if __name__ == "__main__":
    main()
