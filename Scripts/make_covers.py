import pandas as pd, os
from pathlib import Path

# Get the directory of the current script to calculate relative paths
BASE_DIR = Path(__file__).resolve().parent.parent

frames = BASE_DIR / "frames_manifest.csv"
out    = BASE_DIR / "assets" / "covers_manifest.csv"

if not frames.exists():
    print(f"Error: Could not find frames manifest at {frames}")
    exit(1)

df = pd.read_csv(frames)
# pick a path-like column
cands = [c for c in ["first_frame_path","frame_path","cover","thumb","path", "frame_file"] if c in df.columns]
if not cands:
    raise SystemExit("No frame path column found in frames_manifest.csv")
pcol = cands[0]

def clean(p):
    if pd.isna(p): return None
    s = str(p).strip().strip('"').strip("'")
    # If the path is absolute and contains bclark, try to make it relative to BASE_DIR
    if "bclark" in s:
        if "frames" in s:
            rel_part = s.split("frames")[-1].lstrip("\\").lstrip("/")
            return str(BASE_DIR / "frames" / rel_part)
    return s.replace("/", "\\")  # normalize slashes

df["cover_path"] = df[pcol].map(clean)
covers = (df.dropna(subset=["video_id","cover_path"])
            .sort_values(["video_id","cover_path"])
            .groupby("video_id", as_index=False)
            .first()[["video_id","cover_path"]])

# only keep existing local image files
covers = covers[covers["cover_path"].map(
    lambda p: os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg",".jpeg",".png",".bmp"]
)]

os.makedirs(os.path.dirname(out), exist_ok=True)
covers.to_csv(out, index=False)
print("Wrote:", out, "rows:", len(covers))
