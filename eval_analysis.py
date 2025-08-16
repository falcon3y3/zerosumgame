#analysis pipeline for text length versus LLM prediction accuracy using generated predictions 
#python retro_length_analysis.py --predictions results/zero_shot_predictions.csv --output_dir results/length_analysis_zero_shot
#python retro_length_analysis.py --predictions results/fewshot_predictions.csv --output_dir results/length_analysis_few_shot
# retro_length_analysis_multi.py
# Merge many prediction CSVs, analyze whether text length impacts accuracy,
# and compute bootstrap CIs for accuracy by length bucket.
import os
import glob
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ===== ARGUMENT PARSER =====
parser = argparse.ArgumentParser()
parser.add_argument("--predictions_dir", type=str, required=True,
                    help="Folder containing CSV prediction files")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Folder to save length analysis results")
parser.add_argument("--length_unit", choices=["chars", "words"], default="chars",
                    help="Unit for comment length (chars or words)")
args = parser.parse_args()

# ===== CREATE OUTPUT FOLDER =====
os.makedirs(args.output_dir, exist_ok=True)

# ===== LOAD AND CONCATENATE CSV FILES =====
all_files = glob.glob(os.path.join(args.predictions_dir, "*.csv"))
if not all_files:
    raise ValueError(f"No CSV files found in {args.predictions_dir}")

dfs = []
for f in all_files:
    df = pd.read_csv(f)
    if "predicted_zero_sum" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError(f"CSV file {f} missing required columns")
    # Ensure numeric columns
    df["predicted_zero_sum"] = pd.to_numeric(df["predicted_zero_sum"], errors="coerce")
    df["ground_truth"] = pd.to_numeric(df["ground_truth"], errors="coerce")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Drop rows with NaN in predictions or ground truth
df = df.dropna(subset=["predicted_zero_sum", "ground_truth"])
df["predicted_zero_sum"] = df["predicted_zero_sum"].astype(int)
df["ground_truth"] = df["ground_truth"].astype(int)

# ===== CALCULATE COMMENT LENGTH =====
if args.length_unit == "chars":
    df["comment_length"] = df["text"].str.len()
else:
    df["comment_length"] = df["text"].str.split().apply(len)

# ===== BUCKET COMMENTS BY LENGTH (optional) =====
# Example: 10 buckets
df["length_bucket"] = pd.qcut(df["comment_length"], q=10, duplicates='drop')

# ===== SPEARMAN CORRELATION =====
spearman_corr, p_value = spearmanr(df["comment_length"], df["predicted_zero_sum"])
print(f"Spearman correlation between comment length and predicted label: {spearman_corr:.4f} (p={p_value:.4f})")

# ===== SAVE ANALYSIS =====
analysis_path = os.path.join(args.output_dir, "length_vs_prediction.csv")
df.to_csv(analysis_path, index=False)
print(f"Length analysis saved to {analysis_path}")
