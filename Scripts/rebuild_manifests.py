import pandas as pd
import os
from pathlib import Path

def make_relative(p, base):
    if pd.isna(p): return p
    s = str(p)
    # If the path is already relative, return as is
    if not os.path.isabs(s):
        return s
    
    # Try to find a common part like 'frames' or 'assets'
    for folder in ['frames', 'assets', 'out_dense']:
        if f'{os.sep}{folder}{os.sep}' in s:
            return folder + s.split(f'{os.sep}{folder}')[-1]
        if f'/{folder}/' in s:
            return folder + s.split(f'/{folder}/')[-1]
            
    # Fallback: if it's within the base directory, make it relative
    try:
        return os.path.relpath(s, base)
    except ValueError:
        return s

def rebuild_csv(csv_path, base_dir):
    if not os.path.isfile(csv_path):
        print(f"Skipping {csv_path}: not found")
        return
        
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    path_cols = [c for c in df.columns if any(x in c.lower() for x in ['file', 'path', 'thumb', 'cover'])]
    
    for col in path_cols:
        df[col] = df[col].apply(lambda x: make_relative(x, base_dir))
        
    df.to_csv(csv_path, index=False)
    print(f"Done. Updated {len(path_cols)} columns in {csv_path}")

if __name__ == "__main__":
    # Now in Scripts/ folder, so root is parent
    ROOT = Path(__file__).resolve().parent.parent
    
    # List of CSVs that likely have absolute paths
    target_csvs = [
        'frames_manifest.csv',
        'covers_manifest.csv',
        'out_dense/scoring_summary.csv',
        'out_dense/per_frame_details.csv'
    ]
    
    for csv in target_csvs:
        rebuild_csv(ROOT / csv, ROOT)
