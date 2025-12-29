#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, base64, shutil, glob
import pandas as pd
import numpy as np
from datetime import datetime
from html import escape

# ==============================
# Utility helpers
# ==============================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def read_csv_safe(p):
    return pd.read_csv(p, dtype=str, low_memory=False)

def to_float(s):
    return pd.to_numeric(s, errors="coerce")

def pick_numeric(df, candidates):
    for c in candidates:
        if c in df.columns:
            return to_float(df[c])
    return pd.Series([np.nan]*len(df))

def boolish(s: pd.Series):
    if s is None:
        return pd.Series([np.nan]*0)
    x = s.astype(str).str.lower().str.strip()
    return x.isin(["1","true","yes","y","t"])

def pct(x):
    try:
        return f"{x:.0f}%"
    except Exception:
        return "-"

def safe_url(u):
    if not isinstance(u, str):
        return None
    u = u.strip()
    return u if u.lower().startswith(("http://","https://")) else None

def safe_text(v, maxlen=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        s = ""
    else:
        s = str(v)
    if maxlen and len(s) > maxlen:
        return s[:maxlen] + "…"
    return s

def file_to_data_uri(path):
    if not isinstance(path, str):
        return None
    p = path.strip().strip('"').strip("'")
    if not os.path.isfile(p):
        return None
    ext = os.path.splitext(p)[1].lower()
    mime = {
        ".jpg":"image/jpeg",".jpeg":"image/jpeg",
        ".png":"image/png",".bmp":"image/bmp",
        ".svg":"image/svg+xml"
    }.get(ext)
    if not mime:
        return None
    try:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

# ==============================
# Thumbnails: covers + frames + urls
# ==============================

def _join_if_relative(path_str, root_dir):
    if not path_str:
        return None
    p = path_str.strip().strip('"').strip("'")
    if os.path.isabs(p):
        return p
    base = root_dir or ""
    return os.path.normpath(os.path.join(base, p))

def load_covers_map(covers_csv, covers_root=None):
    if not covers_csv or not os.path.isfile(covers_csv):
        return {}
    df = read_csv_safe(covers_csv)
    # normalize headers
    df.columns = [c.lower() for c in df.columns]
    vid_col = next((c for c in ["video_id", "aweme_id", "id"] if c in df.columns), None)
    if not vid_col:
        return {}
    path_col = next((c for c in ["cover_path","frame_path","frame_file","path","thumb","image","file"] if c in df.columns), None)
    if not path_col:
        return {}
    root_guess = covers_root or os.path.dirname(covers_csv)
    m = {}
    for _, r in df.dropna(subset=[vid_col, path_col]).iterrows():
        vid = str(r[vid_col])
        pth = _join_if_relative(str(r[path_col]), root_guess)
        if os.path.isfile(pth):
            m[vid] = pth
    return m

def load_frames_map(frames_csv, frames_root=None):
    if not frames_csv or not os.path.isfile(frames_csv):
        return {}
    df = read_csv_safe(frames_csv)
    df.columns = [c.lower() for c in df.columns]
    vid_col = next((c for c in ["video_id", "aweme_id", "id"] if c in df.columns), None)
    path_col = next((c for c in ["frame_path","path","image","file","frame_file"] if c in df.columns), None)
    if not vid_col or not path_col:
        return {}
    root_guess = frames_root or os.path.dirname(frames_csv)
    # pick FIRST frame per video_id
    df = df.dropna(subset=[vid_col, path_col]).copy()
    df = df.sort_values(by=[vid_col])
    m = {}
    for vid, sub in df.groupby(vid_col):
        pth = _join_if_relative(str(sub.iloc[0][path_col]), root_guess)
        if os.path.isfile(pth):
            m[str(vid)] = pth
    return m

def find_frame_glob(video_id, frames_root):
    """Last-resort glob search: frames_root/**/{video_id}*.jpg"""
    if not frames_root or not os.path.isdir(frames_root):
        return None
    pat = os.path.join(frames_root, "**", f"{video_id}*.jpg")
    hits = glob.glob(pat, recursive=True)
    return hits[0] if hits else None

def row_remote_thumb_url(row):
    for k in ["cover_url","thumbnail_url","thumb_url","image_url","cover"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            u = safe_url(v)
            if u:
                return u
    return None

# ==============================
# Brand colors & Fonts
# ==============================

BRAND_BG = "#13213c"   # page background
CARD_BG  = "#ffffff"   # tables/cards background
TEXT_LG  = "#ffffff"   # text on dark bg
TEXT_DK  = "#13213c"   # text on white cards
LINK_CLR = "#87b6ff"

def prepare_fonts(fonts_src_dir, out_dir):
    """
    Detect fonts in fonts_src_dir, copy to <out_dir>/fonts, and return @font-face CSS.
      - Prefer Manuka .woff2 (else .ttf)
      - Prefer Plus Jakarta Sans .ttf (else .woff2)
    """
    if not fonts_src_dir or not os.path.isdir(fonts_src_dir):
        return ""

    fonts_out = os.path.join(out_dir, "fonts")
    os.makedirs(fonts_out, exist_ok=True)

    manuka_woff2 = manuka_ttf = plus_ttf = plus_woff2 = None
    for fn in os.listdir(fonts_src_dir):
        lower = fn.lower()
        p = os.path.join(fonts_src_dir, fn)
        if not os.path.isfile(p):
            continue
        if "manuka" in lower and lower.endswith(".woff2"):
            manuka_woff2 = p
        if "manuka" in lower and lower.endswith(".ttf"):
            manuka_ttf = p
        if "plus" in lower and "jakarta" in lower and lower.endswith(".ttf"):
            plus_ttf = p
        if "plus" in lower and "jakarta" in lower and lower.endswith(".woff2"):
            plus_woff2 = p

    css_parts = []
    if manuka_woff2 or manuka_ttf:
        src = manuka_woff2 or manuka_ttf
        fmt = "woff2" if src.lower().endswith(".woff2") else "truetype"
        dst = os.path.join(fonts_out, os.path.basename(src))
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
        css_parts.append(f"""
@font-face {{
  font-family: 'Manuka';
  src: url('fonts/{escape(os.path.basename(dst))}') format('{fmt}');
  font-weight: 300 800;
  font-style: normal;
  font-display: swap;
}}""")

    if plus_ttf or plus_woff2:
        src = plus_ttf or plus_woff2
        fmt = "truetype" if src.lower().endswith(".ttf") else "woff2"
        dst = os.path.join(fonts_out, os.path.basename(src))
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
        css_parts.append(f"""
@font-face {{
  font-family: 'Plus Jakarta Sans';
  src: url('fonts/{escape(os.path.basename(dst))}') format('{fmt}');
  font-weight: 300 800;
  font-style: normal;
  font-display: swap;
}}""")

    return "\n".join(css_parts)

def base_style_block(font_css):
    return f"""
<style>
{font_css}
:root {{
  --bg:{BRAND_BG};
  --text:{TEXT_LG};
  --card:{CARD_BG};
  --text-card:{TEXT_DK};
  --link:{LINK_CLR};
}}
html,body{{background:var(--bg); color:var(--text); margin:0; padding:0}}
body{{font-family:'Plus Jakarta Sans','Segoe UI',Arial,sans-serif;}}
h1,h2,h3{{font-family:'Manuka','Plus Jakarta Sans','Segoe UI',Arial,sans-serif; margin:0 0 12px 0}}
.muted{{color:#cfd6e3}}
.wrap{{max-width:1100px;margin:0 auto;padding:24px}}
.logo-wrap{{display:flex;justify-content:center;align-items:center;padding:18px 0 4px}}
.logo-wrap img{{max-height:54px}}

.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 24px 0}}
.tile{{background:var(--card); color:var(--text-card); border-radius:12px;padding:14px 16px;min-width:160px;box-shadow:0 8px 24px rgba(0,0,0,.18)}}
.tile .label{{font-size:12px;color:#66758f}}
.tile .value{{font-size:22px;font-weight:700;margin-top:4px;color:var(--text-card)}}

table{{border-collapse:collapse;width:100%;margin:14px 0 28px 0; background:var(--card); color:var(--text-card); box-shadow:0 8px 24px rgba(0,0,0,.18); border-radius:12px; overflow:hidden}}
th,td{{border-bottom:1px solid #f0f3f7;padding:10px 12px;text-align:left;vertical-align:top}}
th{{background:#f7f9fc; color:#2c3b57}}
tr:last-child td{{border-bottom:none}}

.brand-chip{{display:inline-flex;align-items:center;gap:8px;padding:4px 10px;border-radius:999px;border:1px solid #e7edf5;background:#f7f9fc;margin:0 6px 6px 0;color:#2c3b57}}
.brand-swatch{{display:inline-block;width:16px;height:16px;border-radius:4px;border:1px solid rgba(0,0,0,.1)}}

a{{color:var(--link); text-decoration:none}}
a:hover{{text-decoration:underline}}
.section h2{{margin:24px 0 8px}}

.card-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{background:var(--card); color:var(--text-card); border-radius:12px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,.18)}}
.card img{{display:block;width:100%;height:150px;object-fit:cover;background:#eef2f7}}
.card .meta{{padding:10px 12px}}
.badge{{display:inline-block;font-size:11px;border:1px solid #e7edf5;border-radius:6px;padding:2px 6px;margin-right:6px;background:#f7f9fc;color:#2c3b57}}
.nav{{margin:10px 0 18px 0}}
.nav a{{margin-right:10px}}
.small{{font-size:12px;color:#66758f}}
.center{{text-align:center}}
</style>
"""

# ==============================
# Color parsing & descriptors
# ==============================

def hex_to_rgb(hexstr):
    s = hexstr.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch*2 for ch in s)
    r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    return r, g, b

def rgb_to_hsl(r, g, b):
    r/=255.0; g/=255.0; b/=255.0
    m = min(r,g,b); M = max(r,g,b); C = M - m
    L = (M + m)/2.0
    if C == 0:
        H = 0.0; S = 0.0
    else:
        if M == r:
            H = ((g - b)/C) % 6
        elif M == g:
            H = (b - r)/C + 2
        else:
            H = (r - g)/C + 4
        H *= 60
        S = 0 if L in (0,1) else C/(1 - abs(2*L - 1))
    return H, S, L

def describe_hex_color(hex_code):
    try:
        r,g,b = hex_to_rgb(hex_code)
    except Exception:
        return f"{hex_code} (unrecognized)"
    h,s,l = rgb_to_hsl(r,g,b)
    if s < 0.15:
        if l < 0.20: return "Black"
        if l < 0.35: return "Charcoal"
        if l < 0.60: return "Gray"
        if l < 0.85: return "Silver"
        return "White"
    if 210 <= h <= 260 and l < 0.28: return "Navy"
    if (h >= 345 or h < 15): base = "Red"
    elif 15 <= h < 45: base = "Orange"
    elif 45 <= h < 65: base = "Gold"
    elif 65 <= h < 85: base = "Lime"
    elif 85 <= h < 150: base = "Green"
    elif 150 <= h < 190: base = "Teal"
    elif 190 <= h < 210: base = "Cyan"
    elif 210 <= h < 225: base = "Azure"
    elif 225 <= h < 255: base = "Blue"
    elif 255 <= h < 275: base = "Indigo"
    elif 275 <= h < 290: base = "Violet"
    elif 290 <= h < 320: base = "Magenta"
    else: base = "Pink"
    prefix = "Dark " if l < 0.30 else ("Light " if l > 0.70 else "")
    return prefix + base

# ==============================
# Parse brand palettes (JSON may be flat or per-brand)
# ==============================

BRANDS = ["Anantara Hotels & Resorts", "Avani Hotels & Resorts", "NH Hotels", "NH Collection", "Tivoli Hotels & Resorts", "Oaks Hotels, Resorts & Suites"]

def parse_brand_json_flexible(path):
    """
    Returns (global_colors, by_brand) where:
      - global_colors: list[str] hex
      - by_brand: dict[str -> list[str]] hex
    Accepts:
      { "colors": [...] }
      { "brand_colors": [...] }
      { "palette": [...] }
      { "brands": [{"brand":"Anantara", "colors":[...]}] }
      { "Anantara": [...], "Avani": [...], ...}
    """
    if not path or not os.path.isfile(path):
        return [], {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return [], {}
    global_colors = []
    by_brand = {}
    if isinstance(j, list):
        global_colors = [str(x) for x in j]
    elif isinstance(j, dict):
        # common single-list keys
        for k in ["colors", "brand_colors", "palette"]:
            if k in j and isinstance(j[k], list):
                global_colors = [str(x) for x in j[k]]
        # brands as array of objects
        if "brands" in j and isinstance(j["brands"], list):
            for item in j["brands"]:
                if not isinstance(item, dict): continue
                bname = item.get("brand") or item.get("name")
                cols = item.get("colors")
                if bname and isinstance(cols, list):
                    by_brand[str(bname)] = [str(x) for x in cols]
        # direct brand keys
        for k, v in j.items():
            if isinstance(v, list) and any(name.lower() in k.lower() for name in ["anantara","avani","nh","tivoli","oaks","collection"]):
                by_brand[k] = [str(x) for x in v]
    return global_colors, by_brand

# ==============================
# Heuristic brand classification (from your guidelines)
# ==============================

def classify_brands_for_hex(hx):
    """Return best-fit brand(s) based on HSL heuristics derived from the creative cheat sheet."""
    try:
        r,g,b = hex_to_rgb(hx); h,s,l = rgb_to_hsl(r,g,b)
    except Exception:
        return set()
    out = set()
    # Anantara: natural, warm, cinematic -> warm hues & greens, moderate saturation
    if (15 <= h <= 65 and s >= 0.25 and s <= 0.8) or (85 <= h <= 150 and s <= 0.7):
        out.add("Anantara Hotels & Resorts")
    # Avani: vibrant, upbeat -> high saturation, bright
    if s >= 0.6 and l >= 0.45:
        out.add("Avani Hotels & Resorts")
    # NH Hotels: clean, natural, friendly -> moderate sat, fresh blues/cyans/greens
    if s <= 0.55 and ((190 <= h <= 255) or (85 <= h <= 150)) and 0.3 <= l <= 0.8:
        out.add("NH Hotels")
    # NH Collection: refined, polished -> restrained sat, elegant blues/neutrals
    if s <= 0.45 and ((200 <= h <= 260) or s < 0.15) and 0.25 <= l <= 0.8:
        out.add("NH Collection")
    # Tivoli: warm, romantic, heritage -> warm hues + pinks, mid saturation
    if ((15 <= h <= 45) or (290 <= h <= 345)) and 0.25 <= s <= 0.75 and 0.25 <= l <= 0.7:
        out.add("Tivoli Hotels & Resorts")
    # Oaks: bright, inviting, family -> brighter, cheerful
    if s >= 0.4 and l >= 0.5:
        out.add("Oaks Hotels, Resorts & Suites")
    # neutrals are broadly OK for all; bias to refined brands
    if s < 0.15:
        out.update(["Anantara Hotels & Resorts","NH Hotels","NH Collection","Tivoli Hotels & Resorts"])
    return out

def build_brand_palettes(global_colors, by_brand_json):
    """
    Produce a dict brand -> list[(hex, label)] and an 'Unassigned' list.
    If JSON provides explicit per-brand palettes, use them; otherwise classify from global_colors.
    """
    palettes = {b: [] for b in BRANDS}
    unassigned = []
    if by_brand_json:
        # clean & label
        for b, cols in by_brand_json.items():
            target = next((bb for bb in BRANDS if bb.lower().startswith(b.lower()) or b.lower().startswith(bb.lower()[:4])), None)
            key = target or b
            palettes.setdefault(key, [])
            for c in cols:
                hx = str(c).strip()
                if not hx: continue
                palettes[key].append((hx, describe_hex_color(hx)))
        return palettes, unassigned

    # No per-brand mapping: classify global colors
    for c in global_colors:
        hx = str(c).strip()
        if not hx: continue
        brands = classify_brands_for_hex(hx)
        if brands:
            for b in brands:
                palettes.setdefault(b, []).append((hx, describe_hex_color(hx)))
        else:
            unassigned.append((hx, describe_hex_color(hx)))
    return palettes, unassigned

# ==============================
# HTML Rendering helpers (tables, chips, cards)
# ==============================

def render_tiles(stats):
    html = ['<div class="tiles">']
    for label, value in stats:
        html.append(f'<div class="tile"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>')
    html.append('</div>')
    return "".join(html)

def render_brand_chips(palette):
    if not palette:
        return "(no brand palette provided)"
    chips = []
    for hexcode, label in palette[:30]:
        hex_clean = hexcode.strip()
        chips.append(f"<span class='brand-chip'><span class='brand-swatch' style='background:{escape(hex_clean)}'></span>{escape(label)} <span class='small'>({escape(hex_clean)})</span></span>")
    return " ".join(chips)

def render_table(df, link_cols=None):
    link_cols = link_cols or {}
    d = df.fillna("")
    out = ["<table><thead><tr>"]
    for c in d.columns:
        out.append(f"<th>{escape(str(c))}</th>")
    out.append("</tr></thead><tbody>")
    for _, row in d.iterrows():
        out.append("<tr>")
        for c in d.columns:
            v = row[c]
            if c in link_cols and isinstance(v, str) and v.startswith(("http://","https://")):
                disp = link_cols[c] if isinstance(link_cols[c], str) else v
                out.append(f"<td><a href='{escape(v)}' target='_blank' rel='noopener'>{escape(disp)}</a></td>")
            else:
                out.append(f"<td>{escape(safe_text(v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)

def num_from_row(row, candidates):
    for c in candidates:
        if c in row and row[c] is not None:
            try:
                return float(row[c])
            except Exception:
                try:
                    return float(to_float(pd.Series([row[c]])).iloc[0])
                except Exception:
                    pass
    return np.nan

def reason_blurb(row):
    brand_pct  = num_from_row(row, ["brand_match_frame_percent","brand_pct","brand_tone_percent"])
    overlay_pct= num_from_row(row, ["overlay_frame_percent","overlay_pct"])
    cuts_min   = num_from_row(row, ["cuts_per_minute","cuts_min","cuts_per_min"])
    disc_raw   = row.get("disclosure_ok","")
    disc       = str(disc_raw).lower() in ["1","true","yes","y","t"]

    parts = []
    if not np.isnan(brand_pct):
        if brand_pct >= 50: parts.append(f"strong brand {brand_pct:.0f}%")
        elif brand_pct >= 20: parts.append(f"some brand {brand_pct:.0f}%")
        else: parts.append(f"weak brand {brand_pct:.0f}%")
    if not np.isnan(overlay_pct):
        if overlay_pct >= 40: parts.append("clear overlays")
        elif overlay_pct >= 10: parts.append("limited overlays")
        else: parts.append("no overlays")
    if not np.isnan(cuts_min):
        if cuts_min >= 3: parts.append("fast pacing")
        elif cuts_min >= 1: parts.append("moderate pacing")
        else: parts.append("slow pacing")
    parts.append("disclosure ✓" if disc else "no disclosure")
    if len(parts) > 3:
        parts = parts[:3]
    return ", ".join(parts) if parts else "-"

def card_for_video(row, local_data_uri, remote_src, score_val):
    handle = safe_text(row.get("handle","-"))
    brand  = safe_text(row.get("brand","-"))
    url    = safe_url(safe_text(row.get("video_url")))
    score  = score_val if isinstance(score_val, (int,float,np.floating)) and not np.isnan(score_val) else None
    caption= safe_text(row.get("caption_excerpt",""), maxlen=140)

    if local_data_uri:
        img_tag = f"<img src='{local_data_uri}' alt='thumb'>"
    elif remote_src:
        img_tag = f"<img src='{escape(remote_src)}' alt='thumb'>"
    else:
        img_tag = "<div style='height:150px;background:#eef2f7'></div>"

    why = reason_blurb(row)
    meta = []
    meta.append(f"<span class='badge'>Score: {score:.1f}</span>" if score is not None else "<span class='badge'>Score: -</span>")
    meta.append(f"<span class='badge'>{escape(why)}</span>")
    link_html = f"<a href='{url}' target='_blank' rel='noopener'>Open video</a>" if url else "<span class='small'>(no link)</span>"

    return f"""
    <div class="card">
      {img_tag}
      <div class="meta">
        <div><strong>{escape(handle)}</strong> <span class="small">| {escape(brand)}</span></div>
        <div>{' '.join(meta)}</div>
        <div class="small" style="margin:6px 0 8px 0">{escape(caption)}</div>
        <div>{link_html}</div>
      </div>
    </div>
    """

# ==============================
# Drilldown pages
# ==============================

def build_handle_page(handle, df_h, covers_map, frames_map, frames_root, out_dir, font_css):
    safe_handle = handle.replace('@','at_').replace('/','_').replace('\\','_').replace(' ','_')
    fname = f"{safe_handle}.html"
    path  = os.path.join(out_dir, "handles", fname)
    ensure_dir(os.path.dirname(path))

    score = pick_numeric(df_h, ["composite_score","score","final_score"])
    brand_pct  = pick_numeric(df_h, ["brand_match_frame_percent","brand_pct"])
    overlay_pct= pick_numeric(df_h, ["overlay_frame_percent","overlay_pct"])

    posts = len(df_h)
    avg_score = float(np.nanmean(score)) if score.dropna().size else float("nan")
    bp = brand_pct.dropna(); op = overlay_pct.dropna()

    stats = [
        ("Handle", handle),
        ("Posts analyzed", str(posts)),
        ("Average score", "-" if pd.isna(avg_score) else f"{avg_score:.1f}"),
        ("Avg brand tone", f"{np.nanmean(bp):.0f}%" if bp.size else "-"),
        ("Avg overlay",    f"{np.nanmean(op):.0f}%" if op.size else "-"),
    ]

    cards = []
    for _, r in df_h.iterrows():
        vid = str(r.get("video_id",""))
        # 1) covers map
        img_path = covers_map.get(vid)
        # 2) frames map
        if not img_path:
            img_path = frames_map.get(vid)
        # 3) glob search under frames_root
        if not img_path and frames_root:
            hit = find_frame_glob(vid, frames_root)
            if hit:
                img_path = hit
        # choose local vs remote
        data_uri = file_to_data_uri(img_path) if img_path else None
        remote = row_remote_thumb_url(r) if not data_uri else None
        sv = to_float(pd.Series([r.get("composite_score")])).iloc[0]
        cards.append(card_for_video(r, data_uri, remote, score_val=sv))
    cards_html = "<div class='card-grid'>" + "".join(cards) + "</div>"

    html = f"""<html><head><meta charset='utf-8'><title>{escape(handle)} – TikTok Drilldown</title>{base_style_block(font_css)}</head>
<body>
<div class='wrap'>
<div class='nav'><a href='../index.html'>&larr; Back to index</a></div>
<h1>Handle: {escape(handle)}</h1>
<div class='muted'>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
{render_tiles(stats)}
<div class='section'><h2>Videos</h2>{cards_html}</div>
</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return "handles/" + fname

# ==============================
# Handles scorecard table (alphabetical)
# ==============================

def render_handles_scorecard_table(by_handle_with_links):
    d = by_handle_with_links.copy()
    d = d.sort_values("Handle", key=lambda s: s.astype(str).str.lower())
    # format columns
    for col in ["avg_score","brand_pct","overlay_pct","disclosure_rate"]:
        if col in d.columns:
            if col == "avg_score":
                d[col] = d[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.1f}")
            else:
                d[col] = d[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.0f}%")
    out = ["<table><thead><tr><th>Handle</th><th>Posts</th><th>Avg score</th><th>Brand tone</th><th>Overlay</th><th>Disclosure</th></tr></thead><tbody>"]
    for _, r in d.iterrows():
        out.append("<tr>")
        out.append(f"<td><a href='{escape(r['rel'])}'>{escape(r['Handle'])}</a></td>")
        out.append(f"<td>{int(r['posts'])}</td>")
        out.append(f"<td>{escape(str(r.get('avg_score','-')))}</td>")
        out.append(f"<td>{escape(str(r.get('brand_pct','-')))}</td>")
        out.append(f"<td>{escape(str(r.get('overlay_pct','-')))}</td>")
        out.append(f"<td>{escape(str(r.get('disclosure_rate','-')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)

# ==============================
# Brand palettes section
# ==============================

def render_palettes_by_brand(palettes, unassigned):
    parts = []
    for b in BRANDS:
        chips = render_brand_chips(palettes.get(b, []))
        parts.append(f"<div class='section'><h2>{escape(b)} – Palette</h2>{chips or '(no colors mapped)'}</div>")
    if unassigned:
        parts.append(f"<div class='section'><h2>Unassigned / Off-guideline</h2>{render_brand_chips(unassigned)}</div>")
    return "\n".join(parts)

# ==============================
# Main
# ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--brand_json", default=None)
    ap.add_argument("--covers", default=None, help="CSV mapping video_id->cover_path")
    ap.add_argument("--frames_manifest", default=None, help="CSV of extracted frames (video_id->frame_path)")
    ap.add_argument("--covers_root", default=None, help="Root folder for relative paths in covers CSV")
    ap.add_argument("--frames_root", default=None, help="Root folder for relative paths in frames CSV or for glob fallback")
    ap.add_argument("--out_dir", required=True, help="Folder where index.html + handle pages are written")
    ap.add_argument("--logo", default=None, help="Path to logo image (png/jpg/svg)")
    ap.add_argument("--fonts_dir", default=None, help="Folder containing Manuka (.woff2 or .ttf) and Plus Jakarta Sans (.ttf or .woff2)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    sc = read_csv_safe(args.scoring)
    score      = pick_numeric(sc, ["composite_score","score","final_score"])
    brand_pct  = pick_numeric(sc, ["brand_match_frame_percent","brand_pct","brand_tone_percent"])
    overlay_pct= pick_numeric(sc, ["overlay_frame_percent","overlay_pct"])
    cuts_min   = pick_numeric(sc, ["cuts_per_minute","cuts_min","cuts_per_min"])
    disclosure = sc["disclosure_ok"] if "disclosure_ok" in sc.columns else None

    posts     = len(sc)
    avg_score = np.nanmean(score) if score.dropna().size else np.nan
    med_score = np.nanmedian(score) if score.dropna().size else np.nan
    disc_rate = boolish(disclosure).mean()*100 if disclosure is not None and len(disclosure) else np.nan
    brand_cov = (brand_pct.notna()).mean()*100 if len(brand_pct) else np.nan
    overlay_cov = (overlay_pct.notna()).mean()*100 if len(overlay_pct) else np.nan
    cuts_cov  = (cuts_min.notna()).mean()*100 if len(cuts_min) else np.nan

    # brand pivot
    brand_col = sc["brand"] if "brand" in sc.columns else sc.get("handle", pd.Series([None]*len(sc)))
    dfb = pd.DataFrame({
        "Brand": brand_col,
        "score": score,
        "brand_pct": brand_pct,
        "overlay_pct": overlay_pct,
        "cuts_min": cuts_min,
        "disc": boolish(disclosure) if disclosure is not None else pd.Series([np.nan]*len(sc))
    })
    gb = dfb.groupby("Brand", dropna=False)
    by_brand = pd.DataFrame({
        "Brand": gb.size().index,
        "posts": gb.size().values,
        "avg_score": gb["score"].mean().values,
        "brand_pct": gb["brand_pct"].mean().values,
        "overlay_pct": gb["overlay_pct"].mean().values,
        "cuts_min": gb["cuts_min"].mean().values,
        "disclosure_rate": (gb["disc"].mean().values*100)
    }).sort_values(["avg_score","posts"], ascending=[False, False])

    # per-handle KPI table
    handle_series = sc.get("handle", pd.Series([None]*len(sc))).fillna("(unknown)")
    dfh = pd.DataFrame({
        "Handle": handle_series,
        "score": score,
        "brand_pct": brand_pct,
        "overlay_pct": overlay_pct,
        "disc": boolish(disclosure) if disclosure is not None else pd.Series([np.nan]*len(sc))
    })
    gh = dfh.groupby("Handle", dropna=False)
    by_handle = pd.DataFrame({
        "posts": gh.size()
    })
    by_handle["avg_score"] = gh["score"].mean()
    by_handle["brand_pct"] = gh["brand_pct"].mean()
    by_handle["overlay_pct"] = gh["overlay_pct"].mean()
    by_handle["disclosure_rate"] = gh["disc"].mean()*100
    by_handle = by_handle.reset_index()

    # rank videos
    sc2 = sc.copy()
    sc2["_score_float"] = score
    sc2 = sc2.dropna(subset=["_score_float"]).sort_values("_score_float", ascending=False)
    top5 = sc2.head(5).copy()
    bottom5 = sc2.tail(5).copy().sort_values("_score_float", ascending=True)

    def add_why(df_in):
        df = df_in.copy()
        df["why"] = [reason_blurb(r) for _, r in df.iterrows()]
        return df

    top_disp = add_why(top5)[["handle","brand","_score_float","why","video_url"]].rename(columns={"_score_float":"score"})
    bot_disp = add_why(bottom5)[["handle","brand","_score_float","why","video_url"]].rename(columns={"_score_float":"score"})

    # Thumbnails: build maps
    covers_map = load_covers_map(args.covers, args.covers_root)
    frames_map = load_frames_map(args.frames_manifest, args.frames_root)

    # Fonts + logo
    font_css = prepare_fonts(args.fonts_dir, args.out_dir)
    logo_uri = file_to_data_uri(args.logo) if args.logo else None

    # per-handle drilldowns
    unique_handles = sorted(list(pd.Series(handle_series).dropna().unique()))
    links = []
    for h in unique_handles:
        df_sub = sc[handle_series == h].copy()
        rel = build_handle_page(h, df_sub, covers_map, frames_map, args.frames_root, args.out_dir, font_css)
        links.append((h, rel, len(df_sub)))

    # attach rel path to by_handle for link table
    link_map = {h: rel for (h, rel, _) in links}
    by_handle["rel"] = by_handle["Handle"].map(lambda h: link_map.get(h, ""))

    # Brand palettes per brand
    global_colors, by_brand_json = parse_brand_json_flexible(args.brand_json)
    palettes, unassigned = build_brand_palettes(global_colors, by_brand_json)

    stats = [
        ("Posts analyzed", str(posts)),
        ("Avg composite", "-" if np.isnan(avg_score) else f"{avg_score:.1f}"),
        ("Median composite", "-" if np.isnan(med_score) else f"{med_score:.1f}"),
        ("Disclosure rate", pct(disc_rate)),
        ("Brand tone availability", pct(brand_cov)),
        ("Overlay availability", pct(overlay_cov)),
        ("Cuts/min availability", pct(cuts_cov)),
    ]

    index_html = f"""<html><head><meta charset='utf-8'>
<title>TikTok Executive Summary</title>
{base_style_block(font_css)}
</head><body>
<div class="wrap">
  <div class="logo-wrap">
    {f"<img src='{escape(logo_uri)}' alt='logo'>" if logo_uri else ""}
  </div>

  <h1 class="center">TikTok Executive Summary</h1>
  <div class='center muted'>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

  {render_tiles(stats)}

  <div class='section'><h2>Brand Performance</h2>
  {render_table(by_brand.assign(
      avg_score=by_brand["avg_score"].map(lambda v: '-' if pd.isna(v) else f'{v:.1f}'),
      brand_pct=by_brand["brand_pct"].map(lambda v: '-' if pd.isna(v) else f'{v:.0f}%'),
      overlay_pct=by_brand["overlay_pct"].map(lambda v: '-' if pd.isna(v) else f'{v:.0f}%'),
      cuts_min=by_brand["cuts_min"].map(lambda v: '-' if pd.isna(v) else f'{v:.2f}'),
      disclosure_rate=by_brand["disclosure_rate"].map(lambda v: '-' if pd.isna(v) else f'{v:.0f}%')
  ))}
  </div>

  <div class='section'>
    <h2>Top 5 Videos (what good looks like)</h2>
    {render_table(top_disp, link_cols={"video_url":"open"})}
    <h2>Bottom 5 Videos (what to improve)</h2>
    {render_table(bot_disp, link_cols={"video_url":"open"})}
  </div>

  <div class='section'>
    <h2>Handles (alphabetical scorecards)</h2>
    {render_handles_scorecard_table(by_handle.assign(posts=by_handle["posts"].fillna(0)))}
  </div>

  <div class='section'>
    <h2>Brand Palettes by Brand (guided)</h2>
    {render_palettes_by_brand(palettes, unassigned)}
  </div>

</div>
</body></html>"""

    with open(os.path.join(args.out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print("Wrote:", os.path.join(args.out_dir, "index.html"))
    print("Drilldowns:", os.path.join(args.out_dir, "handles", "<handle>.html"))

if __name__ == "__main__":
    main()
