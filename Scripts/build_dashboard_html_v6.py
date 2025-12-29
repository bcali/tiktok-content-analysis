#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, base64, shutil, glob
import pandas as pd
import numpy as np
from datetime import datetime
from html import escape

# ==============================
# Utility helpers (Inherited from v5)
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
        if np.isnan(x): return "-"
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
# Thumbnail loaders (Inherited from v5)
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
    df.columns = [c.lower() for c in df.columns]
    vid_col = next((c for c in ["video_id", "aweme_id", "id"] if c in df.columns), None)
    if not vid_col: return {}
    path_col = next((c for c in ["cover_path","frame_path","frame_file","path","thumb","image","file"] if c in df.columns), None)
    if not path_col: return {}
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
    if not vid_col or not path_col: return {}
    root_guess = frames_root or os.path.dirname(frames_csv)
    df = df.dropna(subset=[vid_col, path_col]).copy()
    df = df.sort_values(by=[vid_col])
    m = {}
    for vid, sub in df.groupby(vid_col):
        pth = _join_if_relative(str(sub.iloc[0][path_col]), root_guess)
        if os.path.isfile(pth):
            m[str(vid)] = pth
    return m

def find_frame_glob(video_id, frames_root):
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
            if u: return u
    return None

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
    parts.append("disclosure \u2713" if disc else "no disclosure")
    if len(parts) > 3: parts = parts[:3]
    return ", ".join(parts) if parts else "-"

def num_from_row(row, candidates):
    for c in candidates:
        if c in row and row[c] is not None:
            try:
                v = float(row[c])
                if not np.isnan(v): return v
            except Exception:
                pass
    return np.nan

# ==============================
# HTML Template Construction (Modern Design)
# ==============================

def get_html_template(data_json):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TikTok Executive Summary | Minor Hotels</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        .hero-overlay {{
            background: linear-gradient(to bottom, rgba(19, 33, 60, 0.4), rgba(19, 33, 60, 0.95));
        }}
        .gold-accent {{ color: hsl(38, 70%, 55%); }}
        .bg-card-custom {{ background: #ffffff; }}
        .text-muted-custom {{ color: #66758f; }}
        .border-custom {{ border-color: #eef2f7; }}
    </style>
</head>
<body class="bg-[#f8fafc] text-[#13213c]">

    <header class="relative h-80 overflow-hidden bg-[#13213c]">
        <img src="{data_json['hero_image']}" alt="Resort Backdrop" class="absolute inset-0 w-full h-full object-cover">
        <div class="hero-overlay absolute inset-0"></div>
        <div class="relative z-10 h-full flex flex-col items-center justify-center px-4">
            <div class="mb-4">
                <span class="text-3xl font-bold tracking-widest gold-accent">MINOR HOTELS</span>
            </div>
            <h1 class="text-4xl md:text-5xl font-extrabold text-white text-center tracking-tight">
                TikTok Executive Summary
            </h1>
            <p class="text-gray-300 mt-2 text-sm">
                Generated: {datetime.now().strftime("%B %d, %Y")}
            </p>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-8 -mt-12 relative z-20">
        <!-- Stats and Chart Section -->
        <section class="grid grid-cols-1 md:grid-cols-4 lg:grid-cols-5 gap-4 mb-10">
            <div class="col-span-1 md:col-span-2 row-span-2 bg-white p-6 rounded-2xl shadow-xl border border-gray-100">
                <h3 class="text-lg font-bold mb-4 flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-blue-500"></span> Score Distribution
                </h3>
                <div class="h-64 relative">
                    <canvas id="scoreChart"></canvas>
                </div>
            </div>
            
            <div class="bg-white p-5 rounded-2xl shadow-md border border-gray-100 flex flex-col justify-center">
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Posts Analyzed</span>
                <span class="text-3xl font-extrabold mt-1">{data_json['summaryStats']['postsAnalyzed']}</span>
            </div>
            <div class="bg-white p-5 rounded-2xl shadow-md border border-gray-100 flex flex-col justify-center">
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Avg Composite</span>
                <span class="text-3xl font-extrabold mt-1">{data_json['summaryStats']['avgComposite']}</span>
            </div>
            <div class="bg-white p-5 rounded-2xl shadow-md border border-gray-100 flex flex-col justify-center">
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Median Composite</span>
                <span class="text-3xl font-extrabold mt-1">{data_json['summaryStats']['medianComposite']}</span>
            </div>
            <div class="bg-white p-5 rounded-2xl shadow-md border border-gray-100 flex flex-col justify-center">
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Disclosure Rate</span>
                <span class="text-3xl font-extrabold mt-1 text-red-500">{data_json['summaryStats']['disclosureRate']}</span>
            </div>
            <div class="bg-white p-5 rounded-2xl shadow-md border border-gray-100 flex flex-col justify-center">
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Brand Tone</span>
                <span class="text-3xl font-extrabold mt-1 text-green-600">{data_json['summaryStats']['brandToneAvailability']}</span>
            </div>
            <div class="bg-white p-5 rounded-2xl shadow-md border border-gray-100 flex flex-col justify-center">
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Overlay Avail.</span>
                <span class="text-3xl font-extrabold mt-1 text-green-600">{data_json['summaryStats']['overlayAvailability']}</span>
            </div>
        </section>

        <!-- Brand Performance Table -->
        <section class="mb-12">
            <div class="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
                <div class="p-6 border-b border-gray-50">
                    <h2 class="text-xl font-bold flex items-center gap-2">
                        <span class="w-2 h-6 rounded-full bg-blue-900"></span> Brand Performance
                    </h2>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm text-left">
                        <thead class="bg-gray-50 text-gray-500 font-semibold uppercase text-[10px] tracking-widest">
                            <tr>
                                <th class="px-6 py-4">Brand</th>
                                <th class="px-6 py-4">Posts</th>
                                <th class="px-6 py-4 text-center">Avg Score</th>
                                <th class="px-6 py-4 text-center">Brand Tone</th>
                                <th class="px-6 py-4 text-center">Overlay</th>
                                <th class="px-6 py-4 text-center">Cuts/Min</th>
                                <th class="px-6 py-4 text-center">Disclosure</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-50">
                            {render_brand_rows(data_json['brandPerformance'])}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Top 5 Videos -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold mb-6 flex items-center gap-3">
                <span class="w-10 h-10 rounded-xl bg-green-100 flex items-center justify-center text-green-600">★</span>
                Top 5 Videos — What Good Looks Like
            </h2>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
                {render_video_cards(data_json['topVideos'], "excellent")}
            </div>
        </section>

        <!-- Bottom 5 Videos -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold mb-6 flex items-center gap-3">
                <span class="w-10 h-10 rounded-xl bg-red-100 flex items-center justify-center text-red-600">↓</span>
                Bottom 5 Videos — What to Improve
            </h2>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
                {render_video_cards(data_json['bottomVideos'], "low")}
            </div>
        </section>

        <footer class="text-center py-12 border-t border-gray-200 mt-12">
            <p class="text-sm text-gray-400 font-medium">Minor Hotels TikTok Analytics Dashboard</p>
        </footer>
    </main>

    <script>
        const ctx = document.getElementById('scoreChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: ['Score > 60', 'Score 50-60', 'Score 40-50', 'Score < 40'],
                datasets: [{{
                    data: [{data_json['scoreDistribution']['above60']}, {data_json['scoreDistribution']['above50']}, {data_json['scoreDistribution']['above40']}, {data_json['scoreDistribution']['below40']}],
                    backgroundColor: ['#22c55e', '#eab308', '#f97316', '#ef4444'],
                    borderWidth: 0,
                    hoverOffset: 10
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ padding: 20, usePointStyle: true, font: {{ size: 11 }} }} }}
                }},
                cutout: '70%'
            }}
        }});
    </script>
</body>
</html>
"""

def render_brand_rows(rows):
    html = []
    for r in rows:
        html.append(f"""
        <tr class="hover:bg-gray-50 transition-colors">
            <td class="px-6 py-4 font-bold text-gray-900">{r['brand']}</td>
            <td class="px-6 py-4 font-medium text-gray-500">{r['posts']}</td>
            <td class="px-6 py-4 text-center">
                <span class="inline-block px-3 py-1 rounded-lg bg-blue-50 text-blue-700 font-bold">{r['avgScore']}</span>
            </td>
            <td class="px-6 py-4 text-center font-medium text-gray-600">{r['brandPct']}</td>
            <td class="px-6 py-4 text-center font-medium text-gray-600">{r['overlayPct']}</td>
            <td class="px-6 py-4 text-center font-medium text-gray-600">{r['cutsMin']:.2f}</td>
            <td class="px-6 py-4 text-center">
                <span class="px-2 py-1 rounded-md text-[10px] font-bold uppercase { 'bg-green-100 text-green-700' if r['disclosureRate'] != '0%' else 'bg-red-100 text-red-700'}">
                    {r['disclosureRate']}
                </span>
            </td>
        </tr>
        """)
    return "".join(html)

def render_video_cards(videos, status_type):
    html = []
    border_color = "border-green-200" if status_type == "excellent" else "border-red-200"
    badge_bg = "bg-green-500" if status_type == "excellent" else "bg-red-500"
    
    for v in videos:
        img_src = v.get('thumb') or "https://images.unsplash.com/photo-1616469829581-73993eb86b02?q=80&w=200&h=300&auto=format&fit=crop"
        html.append(f"""
        <div class="bg-white rounded-2xl shadow-md border {border_color} overflow-hidden flex flex-col group hover:-translate-y-1 transition-all duration-300">
            <div class="relative h-48 bg-gray-100">
                <img src="{img_src}" alt="Video Thumb" class="w-full h-full object-cover">
                <div class="absolute top-3 right-3">
                    <span class="{badge_bg} text-white text-[10px] font-bold px-2 py-1 rounded-lg shadow-sm">
                        SCORE: {v['score']}
                    </span>
                </div>
            </div>
            <div class="p-4 flex flex-col flex-grow">
                <div class="flex justify-between items-start mb-2">
                    <span class="text-xs font-bold text-gray-400">@{v['handle']}</span>
                    <span class="text-[10px] font-bold bg-gray-100 text-gray-600 px-2 py-0.5 rounded-md">{v['brand']}</span>
                </div>
                <p class="text-xs text-gray-600 line-clamp-2 italic mb-4 flex-grow">"{v['reason']}"</p>
                <a href="{v['videoUrl']}" target="_blank" class="w-full text-center py-2 rounded-xl bg-blue-900 text-white text-xs font-bold hover:bg-blue-800 transition-colors">
                    View on TikTok
                </a>
            </div>
        </div>
        """)
    return "".join(html)

# ==============================
# Main Generator Logic
# ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--brand_json", default=None)
    ap.add_argument("--covers", default=None)
    ap.add_argument("--frames_manifest", default=None)
    ap.add_argument("--covers_root", default=None)
    ap.add_argument("--frames_root", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hero_image", default="https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?q=80&w=1200&auto=format&fit=crop")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    sc = read_csv_safe(args.scoring)
    score      = pick_numeric(sc, ["composite_score","score","final_score"])
    brand_pct  = pick_numeric(sc, ["brand_match_frame_percent","brand_pct","brand_tone_percent"])
    overlay_pct= pick_numeric(sc, ["overlay_frame_percent","overlay_pct"])
    cuts_min   = pick_numeric(sc, ["cuts_per_minute","cuts_min","cuts_per_min"])
    disclosure = sc["disclosure_ok"] if "disclosure_ok" in sc.columns else None

    # Overall Stats
    posts_total = len(sc)
    avg_score = round(np.nanmean(score), 1) if score.dropna().size else 0.0
    med_score = round(np.nanmedian(score), 1) if score.dropna().size else 0.0
    disc_rate = pct(boolish(disclosure).mean()*100) if disclosure is not None and len(disclosure) else "0%"
    brand_cov = pct((brand_pct.notna()).mean()*100) if len(brand_pct) else "0%"
    overlay_cov = pct((overlay_pct.notna()).mean()*100) if len(overlay_pct) else "0%"
    cuts_cov  = pct((cuts_min.notna()).mean()*100) if len(cuts_min) else "0%"

    # Score Distribution
    dist = {
        "above60": int((score >= 60).sum()),
        "above50": int(((score >= 50) & (score < 60)).sum()),
        "above40": int(((score >= 40) & (score < 50)).sum()),
        "below40": int((score < 40).sum()),
        "total": posts_total
    }

    # Brand Performance
    brand_col = sc["brand"] if "brand" in sc.columns else sc.get("handle", pd.Series(["Other"]*len(sc)))
    dfb = pd.DataFrame({
        "brand": brand_col,
        "score": score,
        "brand_pct": brand_pct,
        "overlay_pct": overlay_pct,
        "cuts_min": cuts_min,
        "disc": boolish(disclosure) if disclosure is not None else pd.Series([0]*len(sc))
    })
    gb = dfb.groupby("brand")
    brand_perf = []
    for brand, sub in gb:
        brand_perf.append({
            "brand": brand,
            "posts": len(sub),
            "avgScore": round(sub["score"].mean(), 1),
            "brandPct": pct(sub["brand_pct"].mean()),
            "overlayPct": pct(sub["overlay_pct"].mean()),
            "cutsMin": round(sub["cuts_min"].mean(), 2),
            "disclosureRate": pct(sub["disc"].mean()*100)
        })
    brand_perf = sorted(brand_perf, key=lambda x: x["avgScore"], reverse=True)

    # Videos ranking
    sc2 = sc.copy()
    sc2["_score_float"] = score
    sc2 = sc2.dropna(subset=["_score_float"]).sort_values("_score_float", ascending=False)
    
    covers_map = load_covers_map(args.covers, args.covers_root)
    frames_map = load_frames_map(args.frames_manifest, args.frames_root)

    def get_vid_data(subset):
        vids = []
        for _, r in subset.iterrows():
            vid = str(r.get("video_id",""))
            img_pth = covers_map.get(vid) or frames_map.get(vid)
            if not img_pth and args.frames_root: img_pth = find_frame_glob(vid, args.frames_root)
            
            vids.append({
                "handle": r.get("handle","unknown"),
                "brand": r.get("brand","Other"),
                "score": r.get("_score_float", 0.0),
                "reason": reason_blurb(r),
                "videoUrl": r.get("video_url","#"),
                "thumb": file_to_data_uri(img_pth) if img_pth else None
            })
        return vids

    top_videos = get_vid_data(sc2.head(5))
    bottom_videos = get_vid_data(sc2.tail(5).sort_values("_score_float", ascending=True))

    data_json = {
        "hero_image": args.hero_image,
        "summaryStats": {
            "postsAnalyzed": posts_total,
            "avgComposite": avg_score,
            "medianComposite": med_score,
            "disclosureRate": disc_rate,
            "brandToneAvailability": brand_cov,
            "overlayAvailability": overlay_cov,
            "cutsMinAvailability": cuts_cov
        },
        "scoreDistribution": dist,
        "brandPerformance": brand_perf,
        "topVideos": top_videos,
        "bottomVideos": bottom_videos
    }

    final_html = get_html_template(data_json)
    out_file = os.path.join(args.out_dir, "index.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    print(f"Success: Modern dashboard generated at {out_file}")

if __name__ == "__main__":
    main()

