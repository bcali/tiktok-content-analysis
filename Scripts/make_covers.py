import pandas as pd, os

frames = r"D:\Users\bclark\Tiktok\frames_manifest.csv"
out    = r"D:\Users\bclark\Tiktok\assets\covers_manifest.csv"

df = pd.read_csv(frames)
# pick a path-like column
cands = [c for c in ["first_frame_path","frame_path","cover","thumb","path"] if c in df.columns]
if not cands:
    raise SystemExit("No frame path column found in frames_manifest.csv")
pcol = cands[0]

def clean(p):
    if pd.isna(p): return None
    s = str(p).strip().strip('"').strip("'")
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
