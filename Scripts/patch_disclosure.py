import pandas as pd, numpy as np, re

p_in  = r"D:\Users\bclark\Tiktok\out_dense\scoring_summary.csv"
p_out = r"D:\Users\bclark\Tiktok\out_dense\scoring_summary_with_disclosure.csv"

df = pd.read_csv(p_in, dtype=str, low_memory=False)

def col(name):
    return df[name] if name in df.columns else pd.Series([""]*len(df))

# Look for common disclosure signals in captions/hashtags
text = (col("caption").fillna("") + " " +
        col("caption_excerpt").fillna("") + " " +
        col("hashtags").fillna("")).str.lower()

pattern = r"(#ad|#spon|#sponsored|#gifted|paid partnership|paid\s+partner|sponsored|ad\s*:)"
detected = text.str.contains(pattern, regex=True)

# If disclosure_ok already exists, only fill where empty; else create it
if "disclosure_ok" in df.columns:
    base = df["disclosure_ok"].astype(str).str.lower().str.strip()
    empty = base.isna() | (base == "") | (base == "nan")
    df.loc[empty, "disclosure_ok"] = np.where(detected[empty], "1", "0")
else:
    df["disclosure_ok"] = np.where(detected, "1", "0")

df.to_csv(p_out, index=False)
print("Wrote:", p_out)
