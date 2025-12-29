#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, base64, pandas as pd, numpy as np
from datetime import datetime
from html import escape

# ---------- utils ----------
def ensure_dir(path): 
    if path: os.makedirs(path, exist_ok=True)

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
    if not isinstance(u, str): return None
    u = u.strip()
    # allow http(s) only for external links; for internal links we inject <a> directly
    return u if u.lower().startswith(("http://","https://")) else None

def safe_text(v, maxlen=None):
    """Stringify + handle NaN before escaping."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        s = ""
    else:
        s = str(v)
    if maxlen and len(s) > maxlen:
        return s[:maxlen] + "…"
    return s

def file_to_data_uri(path):
    if not isinstance(path, str): return None
    p = path.strip().strip('"').strip("'")
    if not os.path.isfile(p): return None
    ext = os.path.splitext(p)[1].lower()
    mime = {
        ".jpg":"image/jpeg",".jpeg":"image/jpeg",
        ".png":"image/png",".bmp":"image/bmp"
    }.get(ext)
    if not mime: return None
    try:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def load_covers_map(covers_csv):
    if not covers_csv or not os.path.isfile(covers_csv): 
        return {}
    df = read_csv_safe(covers_csv)
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower
    if "video_id" not in df.columns:
        return {}
    path_col = next((c for c in ["cover_path","frame_file","path","thumb","image","file"] if c in df.columns), None)
    if not path_col: 
        return {}
    m = {}
    for _, r in df.dropna(subset=["video_id", path_col]).iterrows():
        m[str(r["video_id"])] = str(r[path_col])
    return m

# ---------- brand color translation ----------
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
    def tone_prefix(L):
        if L < 0.30: return "Dark "
        if L > 0.70: return "Light "
        return ""
    if 210 <= h <= 260 and l < 0.28:
        return "Navy"
    if (h >= 345 or h < 15): name = "Red"
    elif 15 <= h < 45: name = "Orange"
    elif 45 <= h < 65: name = "Gold"
    elif 65 <= h < 85: name = "Lime"
    elif 85 <= h < 150: name = "Green"
    elif 150 <= h < 190: name = "Teal"
    elif 190 <= h < 210: name = "Cyan"
    elif 210 <= h < 225: name = "Azure"
    elif 225 <= h < 255: name = "Blue"
    elif 255 <= h < 275: name = "Indigo"
    elif 275 <= h < 290: name = "Violet"
    elif 290 <= h < 320: name = "Magenta"
    else: name = "Pink"
    return f"{tone_prefix(l)}{name}".strip()

def parse_brand_palette(brand_json_path):
    if not brand_json_path or not os.path.isfile(brand_json_path):
        return []
    try:
        with open(brand_json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        colors = []
        for key in ["colors","palette","brand_colors"]:
            if key in j and isinstance(j[key], list):
                colors = j[key]; break
        chips = []
        for c in colors:
            if not isinstance(c, str): 
                continue
            h = c.strip()
            if not h: 
                continue
            try:
                label = describe_hex_color(h)
            except Exception:
                label = h
            chips.append((h, label))
        return chips
    except Exception:
        return []

# ---------- HTML ----------
STYLE = """
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111}
h1{margin:0 0 12px 0}
.muted{color:#666}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 24px 0}
.tile{border:1px solid #eee;border-radius:10px;padding:14px 16px;min-width:160px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.tile .label{font-size:12px;color:#666}
.tile .value{font-size:22px;font-weight:600;margin-top:4px}
table{border-collapse:collapse;width:100%;margin:14px 0 28px 0}
th,td{border:1px solid #eee;padding:8px 10px;text-align:left;vertical-align:top}
th{background:#fafafa}
.brand-chip{display:inline-flex;align-items:center;gap:8px;padding:2px 10px;border-radius:999px;border:1px solid #eee;background:#f7f7f7;margin:0 6px 6px 0}
.brand-swatch{display:inline-block;width:16px;height:16px;border-radius:4px;border:1px solid rgba(0,0,0,.1)}
a{color:#0b62d6;text-decoration:none}
a:hover{text-decoration:underline}
.section h2{margin:24px 0 8px}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{border:1px solid #eee;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.card img{display:block;width:100%;height:150px;object-fit:cover;background:#fafafa}
.card .meta{padding:10px 12px}
.badge{display:inline-block;font-size:11px;border:1px solid #eee;border-radius:6px;padding:2px 6px;margin-right:6px;background:#fbfbfb}
.nav{margin:10px 0 18px 0}
.nav a{margin-right:10px}
.small{font-size:12px;color:#555}
</style>
"""

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

# --- concise reasons for good/bad videos ---
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
    # brand
    if not np.isnan(brand_pct):
        if brand_pct >= 50: parts.append(f"strong brand {brand_pct:.0f}%")
        elif brand_pct >= 20: parts.append(f"some brand {brand_pct:.0f}%")
        else: parts.append(f"weak brand {brand_pct:.0f}%")
    # overlay
    if not np.isnan(overlay_pct):
        if overlay_pct >= 40: parts.append("clear overlays")
        elif overlay_pct >= 10: parts.append("limited overlays")
        else: parts.append("no overlays")
    # pacing
    if not np.isnan(cuts_min):
        if cuts_min >= 3: parts.append("fast pacing")
        elif cuts_min >= 1: parts.append("moderate pacing")
        else: parts.append("slow pacing")
    # disclosure
    parts.append("disclosure ✓" if disc else "no disclosure")

    # Keep it tight: max 3 phrases
    if len(parts) > 3:
        parts = parts[:3]
    return ", ".join(parts) if parts else "-"

def card_for_video(row, data_uri, score_val):
    handle = safe_text(row.get("handle","-"))
    brand  = safe_text(row.get("brand","-"))
    url    = safe_url(safe_text(row.get("video_url")))
    score  = score_val if isinstance(score_val, (int,float,np.floating)) and not np.isnan(score_val) else None
    caption= safe_text(row.get("caption_excerpt",""), maxlen=140)

    img_tag = f"<img src='{data_uri}' alt='thumb'>" if data_uri else "<div style='height:150px;background:#f2f2f2'></div>"
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

# ---------- pages ----------
def build_handle_page(handle, df_h, covers_map, out_dir):
    safe_handle = handle.replace('@','at_').replace('/','_').replace('\\','_').replace(' ','_')
    fname = f"{safe_handle}.html"
    path  = os.path.join(out_dir, "handles", fname)
    ensure_dir(os.path.dirname(path))

    score = pick_numeric(df_h, ["composite_score","score","final_score"])
    brand_pct  = pick_numeric(df_h, ["brand_match_frame_percent","brand_pct"])
    overlay_pct= pick_numeric(df_h, ["overlay_frame_percent","overlay_pct"])

    posts = len(df_h)
    avg_score = float(np.nanmean(score)) if score.dropna().size else float("nan")

    stats = [
        ("Handle", handle),
        ("Posts analyzed", str(posts)),
        ("Average score", "-" if pd.isna(avg_score) else f"{avg_score:.1f}"),
    ]
    bp = brand_pct.dropna()
    if bp.size: stats.append(("Avg brand tone", f"{np.nanmean(bp):.0f}%"))
    op = overlay_pct.dropna()
    if op.size: stats.append(("Avg overlay", f"{np.nanmean(op):.0f}%"))

    # video cards (thumbs)
    cards = []
    for _, r in df_h.iterrows():
        vid = str(r.get("video_id",""))
        data_uri = file_to_data_uri(covers_map.get(vid))
        sv = to_float(pd.Series([r.get("composite_score")])).iloc[0]
        cards.append(card_for_video(r, data_uri, score_val=sv))
    cards_html = "<div class='card-grid'>" + "".join(cards) + "</div>"

    html = f"""<html><head><meta charset='utf-8'><title>{escape(handle)} – TikTok Drilldown</title>{STYLE}</head>
<body>
<div class='nav'><a href='../index.html'>&larr; Back to index</a></div>
<h1>Handle: {escape(handle)}</h1>
<div class='muted'>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
{render_tiles(stats)}
<div class='section'><h2>Videos</h2>{cards_html}</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return "handles/" + fname

def render_handles_table(links):
    """links: list of (handle, rel_path, n_posts). Returns alphabetical table with clickable handle."""
    links_sorted = sorted(links, key=lambda x: str(x[0]).lower())
    out = ["<table><thead><tr><th>Handle</th><th>Posts</th></tr></thead><tbody>"]
    for h, rel, n in links_sorted:
        out.append(f"<tr><td><a href='{escape(rel)}'>{escape(h)}</a></td><td>{int(n)}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--brand_json", default=None)
    ap.add_argument("--covers", default=None, help="CSV mapping video_id->cover_path")
    ap.add_argument("--out_dir", required=True, help="Folder where index.html + handle pages are written")
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

    # handle list + drilldowns
    handle_series = sc.get("handle", pd.Series([None]*len(sc))).fillna("(unknown)")
    unique_handles = sorted(list(pd.Series(handle_series).dropna().unique()))

    sc2 = sc.copy()
    sc2["_score_float"] = score
    sc2 = sc2.dropna(subset=["_score_float"]).sort_values("_score_float", ascending=False)
    top5 = sc2.head(5).copy()
    bottom5 = sc2.tail(5).copy().sort_values("_score_float", ascending=True)

    # add "why" blurbs to top/bottom
    def add_why(df_in):
        df = df_in.copy()
        whys = []
        for _, r in df.iterrows():
            whys.append(reason_blurb(r))
        df["why"] = whys
        return df

    top_disp = add_why(top5)[["handle","brand","_score_float","why","video_url"]].rename(columns={"_score_float":"score"})
    bot_disp = add_why(bottom5)[["handle","brand","_score_float","why","video_url"]].rename(columns={"_score_float":"score"})

    covers_map = load_covers_map(args.covers)

    links = []
    for h in unique_handles:
        df_h = sc[handle_series == h].copy()
        rel = build_handle_page(h, df_h, covers_map, args.out_dir)
        links.append((h, rel, len(df_h)))

    palette = parse_brand_palette(args.brand_json)

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
{STYLE}
</head><body>
<h1>TikTok Executive Summary</h1>
<div class='muted'>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

<div style='margin:8px 0'>Brand palette: {" ".join([f"<span class='brand-chip'><span class='brand-swatch' style='background:{escape(h)}'></span>{escape(label)} <span class='small'>({escape(h)})</span></span>" for h,label in palette]) or "(no brand palette provided)"}</div>

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
  <h2>Handles</h2>
  {render_handles_table(links)}
</div>

</body></html>"""

    with open(os.path.join(args.out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print("Wrote:", os.path.join(args.out_dir, "index.html"))
    print("Drilldowns:", os.path.join(args.out_dir, "handles", "<handle>.html"))

if __name__ == "__main__":
    main()
