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
    if not isinstance(path, str): return None
    p = path.strip().strip('"').strip("'")
    if not os.path.isfile(p): return None
    ext = os.path.splitext(p)[1].lower()
    mime = {
        ".jpg":"image/jpeg",".jpeg":"image/jpeg",
        ".png":"image/png",".bmp":"image/bmp",
        ".svg":"image/svg+xml"
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

# ---------- brand colors & fonts ----------
BRAND_BG = "#13213c"      # page background
CARD_BG  = "#ffffff"      # cards/tables background
TEXT_LG  = "#ffffff"      # primary text on dark bg
TEXT_DK  = "#13213c"      # text on white cards
LINK_CLR = "#87b6ff"      # link color on dark bg

def build_font_css(fonts_dir):
    """Return @font-face CSS if .woff2 files exist; else empty (fallbacks kick in)."""
    if not fonts_dir:
        return ""
    manuka = None
    pjs = None
    try:
        # try some common filenames
        for fn in os.listdir(fonts_dir):
            f = fn.lower()
            if f.startswith("manuka") and f.endswith(".woff2") and not manuka:
                manuka = os.path.join(fonts_dir, fn)
            if "plus" in f and "jakarta" in f and f.endswith(".woff2") and not pjs:
                pjs = os.path.join(fonts_dir, fn)
    except Exception:
        return ""
    css = []
    if manuka and os.path.isfile(manuka):
        manuka_rel = os.path.basename(manuka)
        css.append(f"""
@font-face {{
  font-family: 'Manuka';
  src: url('fonts/{escape(manuka_rel)}') format('woff2');
  font-weight: 400 800; font-style: normal; font-display: swap;
}}""")
    if pjs and os.path.isfile(pjs):
        pjs_rel = os.path.basename(pjs)
        css.append(f"""
@font-face {{
  font-family: 'Plus Jakarta Sans';
  src: url('fonts/{escape(pjs_rel)}') format('woff2');
  font-weight: 300 800; font-style: normal; font-display: swap;
}}""")
    return "\n".join(css)

# ---------- HTML base styles (dark theme + white cards) ----------
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
    if len(parts) > 3: parts = parts[:3]
    return ", ".join(parts) if parts else "-"

def card_for_video(row, data_uri, score_val):
    handle = safe_text(row.get("handle","-"))
    brand  = safe_text(row.get("brand","-"))
    url    = safe_url(safe_text(row.get("video_url")))
    score  = score_val if isinstance(score_val, (int,float,np.floating)) and not np.isnan(score_val) else None
    caption= safe_text(row.get("caption_excerpt",""), maxlen=140)

    img_tag = f"<img src='{data_uri}' alt='thumb'>" if data_uri else "<div style='height:150px;background:#eef2f7'></div>"
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
def build_handle_page(handle, df_h, covers_map, out_dir, font_css):
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
        data_uri = file_to_data_uri(covers_map.get(vid))
        sv = to_float(pd.Series([r.get("composite_score")])).iloc[0]
        cards.append(card_for_video(r, data_uri, score_val=sv))
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

def describe_hex_color(hex_code):
    # (short version included earlier in analysis; omitted here for brevity)
    try:
        s = hex_code.strip().lstrip("#")
        if len(s)==3: s="".join(ch*2 for ch in s)
        r=int(s[0:2],16); g=int(s[2:4],16); b=int(s[4:6],16)
    except Exception:
        return f"{hex_code} (unrecognized)"
    def rgb_to_hsl(r,g,b):
        r/=255; g/=255; b/=255
        m=min(r,g,b); M=max(r,g,b); C=M-m; L=(M+m)/2
        if C==0: return 0,0,L
        H = 60*(((g-b)/C)%6 if M==r else (b-r)/C+2 if M==g else (r-g)/C+4)
        S = 0 if L in (0,1) else C/(1-abs(2*L-1))
        return H,S,L
    h,s,l=rgb_to_hsl(r,g,b)
    if s<0.15:
        if l<0.20: return "Black"
        if l<0.35: return "Charcoal"
        if l<0.60: return "Gray"
        if l<0.85: return "Silver"
        return "White"
    if 210<=h<=260 and l<0.28: return "Navy"
    if (h>=345 or h<15): base="Red"
    elif 15<=h<45: base="Orange"
    elif 45<=h<65: base="Gold"
    elif 65<=h<85: base="Lime"
    elif 85<=h<150: base="Green"
    elif 150<=h<190: base="Teal"
    elif 190<=h<210: base="Cyan"
    elif 210<=h<225: base="Azure"
    elif 225<=h<255: base="Blue"
    elif 255<=h<275: base="Indigo"
    elif 275<=h<290: base="Violet"
    elif 290<=h<320: base="Magenta"
    else: base="Pink"
    prefix = "Dark " if l<0.30 else ("Light " if l>0.70 else "")
    return prefix+base

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
            label = describe_hex_color(h)
            chips.append((h, label))
        return chips
    except Exception:
        return []

def render_handles_scorecard_table(by_handle_with_links):
    d = by_handle_with_links.copy()
    d = d.sort_values("Handle", key=lambda s: s.str.lower())
    # format columns
    for col in ["avg_score","brand_pct","overlay_pct","disclosure_rate"]:
        if col in d.columns:
            if col == "avg_score":
                d[col] = d[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.1f}")
            else:
                d[col] = d[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.0f}%")
    # manual render so we can keep <a> for handle
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--brand_json", default=None)
    ap.add_argument("--covers", default=None, help="CSV mapping video_id->cover_path")
    ap.add_argument("--out_dir", required=True, help="Folder where index.html + handle pages are written")
    ap.add_argument("--logo", default=None, help="Path to logo image (png/jpg/svg)")
    ap.add_argument("--fonts_dir", default=None, help="Folder containing Manuka*.woff2 and PlusJakartaSans*.woff2")
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

    # handle list + drilldowns + per-handle KPIs
    handle_series = sc.get("handle", pd.Series([None]*len(sc))).fillna("(unknown)")
    unique_handles = sorted(list(pd.Series(handle_series).dropna().unique()))

    # per-handle KPI table (scorecard)
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

    covers_map = load_covers_map(args.covers)

    # fonts + assets
    font_css = build_font_css(args.fonts_dir)
    logo_uri = file_to_data_uri(args.logo) if args.logo else None

    # per-handle drilldowns
    links = []
    for h in unique_handles:
        df_sub = sc[handle_series == h].copy()
        rel = build_handle_page(h, df_sub, covers_map, args.out_dir, font_css)
        links.append((h, rel, len(df_sub)))

    # attach rel path to by_handle for link table
    link_map = {h: rel for (h, rel, _) in links}
    by_handle["rel"] = by_handle["Handle"].map(lambda h: link_map.get(h, ""))

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
{base_style_block(font_css)}
</head><body>
<div class="wrap">
  <div class="logo-wrap">
    {f"<img src='{escape(logo_uri)}' alt='logo'>" if logo_uri else ""}
  </div>

  <h1 class="center">TikTok Executive Summary</h1>
  <div class='center muted'>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

  <div style='margin:12px 0' class='center'>Brand palette: {" ".join([f"<span class='brand-chip'><span class='brand-swatch' style='background:{escape(h)}'></span>{escape(label)} <span class='small'>({escape(h)})</span></span>" for h,label in palette]) or "(no brand palette provided)"}</div>

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

</div>
</body></html>"""

    with open(os.path.join(args.out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print("Wrote:", os.path.join(args.out_dir, "index.html"))
    print("Drilldowns:", os.path.join(args.out_dir, "handles", "<handle>.html"))

if __name__ == "__main__":
    main()
