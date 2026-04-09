import pandas as pd
from pathlib import Path

csv_dir = Path("data/digits/")

df = pd.read_csv(csv_dir / "train.csv", header=None, names=["image", "label"], dtype={"image": str, "label": str})
df_sorted = df.sort_values(by="image", key=lambda x: x.str.lower())

df_sorted.to_csv(csv_dir / "sorted.csv", index=False, header=False)