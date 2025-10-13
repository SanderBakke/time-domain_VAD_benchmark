#!/usr/bin/env python
import argparse, numpy as np, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--out_npz", required=True)
ap.add_argument("--use_prob", action="store_true")
args = ap.parse_args()

col = "prob" if args.use_prob else "score"
dtypes = {"label_frame":"int8", col:"float32"}
df = pd.read_csv(args.in_csv, usecols=["label_frame", col], dtype=dtypes, engine="c", memory_map=True, low_memory=False)
y = df["label_frame"].to_numpy()
s = df[col].to_numpy()
mask = np.isfinite(s)
np.savez_compressed(args.out_npz, y=y[mask], s=s[mask])
print(f"Cached → {args.out_npz}  (rows={mask.sum():,})")
