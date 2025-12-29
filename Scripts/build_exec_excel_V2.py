#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, pandas as pd, numpy as np
from datetime import datetime

def ensure_dir(path: str):
    if path: os.makedirs(path, exist_ok=True)

def read_csv_safe(p: str) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, low_memory=False)

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
    try:
        return f"{x:.0f}%"
    except Exception:
        return "-"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--out", required=True, help="Output .xlsx path")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    ensure_dir(out_dir)

    sc = read_csv_safe(args.scoring)

    # Flexible numeric fields
    score      = pick_numeric(sc, ["composite_score","score","final_score"])
    brand_pct  = pick_numeric(sc, ["brand_match_frame_percent","brand_pct","brand_tone_percent"])
    overlay_pct= pick_numeric(sc, ["overlay_frame_percent","overlay_pct"])
    cuts_min   = pick_numeric(sc, ["cuts_per_minute","cuts_min","cuts_per_min"])

    # Booleans
    disclosure = sc["disclosure_ok"] if "disclosure_ok" in sc.columns else None
    disc_rate = boolish(disclosure).mean()*100 if disclosure is not None and len(disclosure) else np.nan

    # Overview metrics
    posts = len(sc)
    avg_score   = np.nanmean(score) if len(score) else np.nan
    med_score   = np.nanmedian(score) if len(score) else np.nan
    brand_cov   = (brand_pct.notna()).mean()*100 if len(brand_pct) else np.nan
    overlay_cov = (overlay_pct.notna()).mean()*100 if len(overlay_pct) else np.nan
    cuts_cov    = (cuts_min.notna()).mean()*100 if len(cuts_min) else np.nan

    # Build pivots
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
        "posts": gb.size()
    })
    by_brand["avg_score"] = gb["score"].mean()
    by_brand["brand_pct"] = gb["brand_pct"].mean()
    by_brand["overlay_pct"] = gb["overlay_pct"].mean()
    by_brand["cuts_min"] = gb["cuts_min"].mean()
    by_brand["disclosure_rate"] = gb["disc"].mean()*100
    by_brand = by_brand.reset_index().sort_values(["avg_score","posts"], ascending=[False, False])

    by_hotel = None
    if "hotel" in sc.columns:
        dfh = pd.DataFrame({
            "Hotel": sc["hotel"],
            "score": score,
            "brand_pct": brand_pct,
            "overlay_pct": overlay_pct,
            "cuts_min": cuts_min,
            "disc": boolish(disclosure) if disclosure is not None else pd.Series([np.nan]*len(sc))
        })
        gh = dfh.groupby("Hotel", dropna=False)
        by_hotel = pd.DataFrame({
            "posts": gh.size()
        })
        by_hotel["avg_score"] = gh["score"].mean()
        by_hotel["brand_pct"] = gh["brand_pct"].mean()
        by_hotel["overlay_pct"] = gh["overlay_pct"].mean()
        by_hotel["cuts_min"] = gh["cuts_min"].mean()
        by_hotel["disclosure_rate"] = gh["disc"].mean()*100
        by_hotel = by_hotel.reset_index().sort_values(["avg_score","posts"], ascending=[False, False])

    # Top/Bottom 5
    sc2 = sc.copy()
    sc2["_score_float"] = score
    sc2 = sc2.dropna(subset=["_score_float"])
    sc2 = sc2.sort_values("_score_float", ascending=False)
    top5 = sc2.head(5)
    bottom5 = sc2.tail(5).sort_values("_score_float", ascending=True)

    # Write Excel
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as xw:
        wb  = xw.book

        # Overview sheet
        ov = pd.DataFrame({
            "Metric": [
                "Generated",
                "Posts analyzed",
                "Average score",
                "Median score",
                "Disclosure rate",
                "Brand tone availability",
                "Overlay availability",
                "Cuts/min availability",
            ],
            "Value": [
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                posts,
                f"{avg_score:.1f}" if not np.isnan(avg_score) else "-",
                f"{med_score:.1f}" if not np.isnan(med_score) else "-",
                pct(disc_rate),
                pct(brand_cov),
                pct(overlay_cov),
                pct(cuts_cov),
            ]
        })
        ov.to_excel(xw, sheet_name="Overview", index=False)

        # All rows
        sc.to_excel(xw, sheet_name="All Rows", index=False)
        ws_all = xw.sheets["All Rows"]
        ws_all.freeze_panes(1, 0)
        ws_all.autofilter(0, 0, len(sc), max(0, sc.shape[1]-1))

        # By Brand
        disp_brand = by_brand.copy()
        for c in ["brand_pct","overlay_pct","disclosure_rate"]:
            if c in disp_brand.columns:
                disp_brand[c] = disp_brand[c].map(lambda v: None if pd.isna(v) else float(v))
        disp_brand.to_excel(xw, sheet_name="By Brand", index=False)

        # By Hotel
        if by_hotel is not None:
            disp_hotel = by_hotel.copy()
            for c in ["brand_pct","overlay_pct","disclosure_rate"]:
                if c in disp_hotel.columns:
                    disp_hotel[c] = disp_hotel[c].map(lambda v: None if pd.isna(v) else float(v))
            disp_hotel.to_excel(xw, sheet_name="By Hotel", index=False)

        # Top 5 / Bottom 5
        top5.drop(columns=["_score_float"], errors="ignore").to_excel(xw, sheet_name="Top 5", index=False)
        bottom5.drop(columns=["_score_float"], errors="ignore").to_excel(xw, sheet_name="Bottom 5", index=False)

    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
