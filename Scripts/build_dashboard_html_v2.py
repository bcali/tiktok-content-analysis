#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, pandas as pd, numpy as np
from datetime import datetime
from html import escape

def ensure_dir(path): 
    if path: os.makedirs(path, exist_ok=True)

def read_csv_safe(p): return pd.read_csv(p, dtype=str, low_memory=False)
def to_float(s): return pd.to_numeric(s, errors="coerce")

def pick_numeric(df, candidates):
    for c in candidates:
        if c in df.columns: return to_float(df[c])
    return pd.Series([np.nan]*len(df))

def boolish(s: pd.Series):
    if s is None: return pd.Series([np.nan]*0)
    x = s.astype(str).str.lower().str.strip()
    return x.isin(["1","true","yes","y","t"])

def pct(x):
    try: return f"{x:.0f}%"
    except Exception: return "-"

def safe_url(u):
    if not isinstance(u, str): return None
    u = u.strip()
    return u if u.lower().startswith(("http://","https://")) else None

def render_table(df, cols):
    d = df[cols].copy() if cols else df.copy()
    # escape cells
    d = d.fillna("")
    html = ["<table border='1' class='dataframe'>"]
    # header
    html.append("<thead><tr>")
    for c in d.columns: html.append(f"<th>{escape(str(c))}</th>")
    html.append("</tr></thead>")
    # body
    html.append("<tbody>")
    for _, row in d.iterrows():
        html.append("<tr>")
        for c in d.columns:
            html.append(f"<td>{escape(str(row[c]))}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--brand_json", default=None, help="Optional brand.json to show palette chips")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    ensure_dir(out_dir)

    sc = read_csv_safe(args.scoring)

    # Flexible fields
    score      = pick_numeric(sc, ["composite_score","score","final_score"])
    brand_pct  = pick_numeric(sc, ["brand_match_frame_percent","brand_pct","brand_tone_percent"])
    overlay_pct= pick_numeric(sc, ["overlay_frame_percent","overlay_pct"])
    cuts_min   = pick_numeric(sc, ["cuts_per_minute","cuts_min","cuts_per_min"])
    disclosure = sc["disclosure_ok"] if "disclosure_ok" in sc.columns else None

    posts     = len(sc)
    avg_score = np.nanmean(score) if len(score) else np.nan
    med_score = np.nanmedian(score) if len(score) else np.nan
    disc_rate = boolish(disclosure).mean()*100 if disclosure is not None and len(disclosure) else np.nan
    brand_cov = (brand_pct.notna()).mean()*100 if len(brand_pct) else np.nan
    overlay_cov = (overlay_pct.notna()).mean()*100 if len(overlay_pct) else np.nan
    cuts_cov  = (cuts_min.notna()).mean()*100 if len(cuts_min) else np.nan

    # Brand pivot
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

    # Handle table (top 50 by avg score)
    handle_col = sc.get("handle", pd.Series([None]*len(sc)))
    dfh = pd.DataFrame({
        "Brand": brand_col,
        "Handle": handle_col,
        "score": score
    })
    gh = dfh.groupby(["Brand","Handle"], dropna=False).agg(
        posts=("score","size"),
        avg_score=("score","mean")
    ).reset_index().sort_values(["avg_score","posts"], ascending=[False, False]).head(50)

    # Top / Bottom 5
    sc2 = sc.copy()
    sc2["_score_float"] = score
    sc2 = sc2.dropna(subset=["_score_float"]).sort_values("_score_float", ascending=False)
    top5 = sc2.head(5).copy()
    bottom5 = sc2.tail(5).copy().sort_values("_score_float", ascending=True)

    def disp_topbot(df):
        out = pd.DataFrame({
            "rank": list(range(1, len(df)+1)),
            "handle": df.get("handle","-"),
            "brand": df.get("brand","-"),
            "score": df["_score_float"]
        })
        # Attach URLs if any
        urls = df.get("video_url")
        if urls is not None:
            out["video_url"] = urls.map(lambda u: safe_url(u) or "")
        return out

    top_disp = disp_topbot(top5)
    bot_disp = disp_topbot(bottom5)

    # Palette chips (optional)
    chips = ""
    if args.brand_json and os.path.isfile(args.brand_json):
        try:
            with open(args.brand_json, "r", encoding="utf-8") as f:
                bj = json.load(f)
            colors = []
            for key in ["colors","palette","brand_colors"]:
                if key in bj and isinstance(bj[key], list):
                    colors = bj[key]; break
            if colors:
                chips = " ".join([f"<span class='brand-chip'>{escape(str(c))}</span>" for c in colors[:30]])
        except Exception:
            pass

    # HTML
    html = f"""<html><head><meta charset='utf-8'><title>TikTok Exec Summary</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111}}
h1{{margin:0 0 12px 0}}
.muted{{color:#666}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 24px 0}}
.tile{{border:1px solid #eee;border-radius:10px;padding:14px 16px;min-width:160px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.tile .label{{font-size:12px;color:#666}}
.tile .value{{font-size:22px;font-weight:600;margin-top:4px}}
table{{border-collapse:collapse;width:100%;margin:14px 0 28px 0}}
th,td{{border:1px solid #eee;padding:8px 10px;text-align:left}}
th{{background:#fafafa}}
.brand-chip{{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #eee;background:#f7f7f7;margin-right:4px}}
a{{color:#0b62d6;text-decoration:none}}
a:hover{{text-decoration:underline}}
.section h2{{margin:24px 0 8px}}
</style>
</head><body>
<h1>TikTok Executive Summary</h1>
<div class='muted'>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
<div style='margin-top:6px'>Brand palette: {chips or '(none)'} </div>

<div class="tiles">
  <div class="tile"><div class="label">Posts analyzed</div><div class="value">{posts}</div></div>
  <div class="tile"><div class="label">Avg composite</div><div class="value">{'-' if np.isnan(avg_score) else f'{avg_score:.1f}'}</div></div>
  <div class="tile"><div class="label">Median composite</div><div class="value">{'-' if np.isnan(med_score) else f'{med_score:.1f}'}</div></div>
  <div class="tile"><div class="label">Disclosure rate</div><div class="value">{pct(disc_rate)}</div></div>
  <div class="tile"><div class="label">Brand tone availability</div><div class="value">{pct(brand_cov)}</div></div>
  <div class="tile"><div class="label">Overlay availability</div><div class="value">{pct(overlay_cov)}</div></div>
  <div class="tile"><div class="label">Cuts/min availability</div><div class="value">{pct(cuts_cov)}</div></div>
</div>

<div class='section'><h2>Brand Performance</h2>
{render_table(by_brand.assign(
    avg_score=by_brand["avg_score"].map(lambda v: '-' if pd.isna(v) else f'{v:.1f}'),
    brand_pct=by_brand["brand_pct"].map(lambda v: '-' if pd.isna(v) else f'{v:.0f}%'),
    overlay_pct=by_brand["overlay_pct"].map(lambda v: '-' if pd.isna(v) else f'{v:.0f}%'),
    cuts_min=by_brand["cuts_min"].map(lambda v: '-' if pd.isna(v) else f'{v:.2f}'),
    disclosure_rate=by_brand["disclosure_rate"].map(lambda v: '-' if pd.isna(v) else f'{v:.0f}%')
), ["Brand","posts","avg_score","brand_pct","overlay_pct","cuts_min","disclosure_rate"])}
</div>

<div class='section'><h2>Hotel / Handle Performance (Top 50 by Avg Score)</h2>
{render_table(gh.assign(
    avg_score=gh["avg_score"].map(lambda v: '-' if pd.isna(v) else f'{v:.1f}')
), ["Brand","Handle","posts","avg_score"])}
</div>

<div class='section'><h2>Examples for Training</h2>

<h3>Top 5 Videos (what good looks like)</h3>
{render_table(top_disp, ["rank","handle","brand","score","video_url"])}

<h3>Bottom 5 Videos (what to improve)</h3>
{render_table(bot_disp, ["rank","handle","brand","score","video_url"])}

</div>
</body></html>"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
