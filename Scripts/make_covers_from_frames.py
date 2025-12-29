import pandas as pd, os

FRAMES = r"D:\Users\bclark\Tiktok\frames_manifest.csv"
OUT    = r"D:\Users\bclark\Tiktok\assets\covers_manifest.csv"

df = pd.read_csv(FRAMES)

# Expect columns: video_id, frame_file, timestamp_s
if not {"video_id","frame_file"}.issubset(df.columns):
    raise SystemExit("frames_manifest.csv must have video_id and frame_file columns")

def clean(p):
    if pd.isna(p): return None
    s = str(p).strip().strip('"').strip("'")
    return s.replace("/", "\\")  # normalize slashes for Windows

df["cover_path"] = df["frame_file"].map(clean)

# pick earliest frame per video (smallest timestamp) to use as cover
if "timestamp_s" in df.columns:
    df = df.sort_values(["video_id","timestamp_s","cover_path"])
else:
    df = df.sort_values(["video_id","cover_path"])

covers = (df.dropna(subset=["video_id","cover_path"])
            .groupby("video_id", as_index=False)
            .first()[["video_id","cover_path"]])

# keep only existing local image files
def is_img(p):
    if not isinstance(p, str): return False
    ext = os.path.splitext(p)[1].lower()
    return os.path.isfile(p) and ext in [".jpg",".jpeg",".png",".bmp"]

covers = covers[covers["cover_path"].map(is_img)]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
covers.to_csv(OUT, index=False)
print(f"Wrote: {OUT} rows: {len(covers)}")
