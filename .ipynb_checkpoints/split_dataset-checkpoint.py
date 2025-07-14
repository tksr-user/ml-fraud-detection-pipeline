import pandas as pd
import numpy as np
import os

# === CONFIGURATION ===
RAW_DATA_PATH = "C:/Users/HP/ml_pipeline_project/data/raw/creditcard.csv" # Adjust this if your file is elsewhere
OUTPUT_DIR = "data/splits"
NUM_SPLITS = 10

# === STEP 1: LOAD ORIGINAL DATASET ===
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Loaded dataset with {len(df)} rows from: {RAW_DATA_PATH}")
except FileNotFoundError:
    print(f"❌ File not found: {RAW_DATA_PATH}")
    exit(1)

# === STEP 2: SHUFFLE THE DATA ===
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("🔀 Dataset shuffled")

# === STEP 3: SPLIT INTO D1 TO D10 ===
splits = np.array_split(df, NUM_SPLITS)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === STEP 4: SAVE EACH SPLIT ===
for i, split_df in enumerate(splits, start=1):
    filename = os.path.join(OUTPUT_DIR, f"D{i}.csv")
    split_df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename} ({len(split_df)} rows)")

print("\n Done. All splits saved in 'data/splits' folder.")
