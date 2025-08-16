#analysis pipeline for text length versus LLM prediction accuracy using generated predictions 
#python retro_length_analysis.py --predictions results/zero_shot_predictions.csv --output_dir results/length_analysis_zero_shot
#python retro_length_analysis.py --predictions results/fewshot_predictions.csv --output_dir results/length_analysis_few_shot
# retro_length_analysis_multi.py
# Merge many prediction CSVs, analyze whether text length impacts accuracy,
# and compute bootstrap CIs for accuracy by length bucket.

# retro_length_analysis.py
import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ===== ARGUMENT PARSER =====
parser = argparse.ArgumentParser(description="Analyze prediction accuracy vs comment length")
parser.add_argument("--predictions_dir", type=str, required=True,
                    help="Folder containing CSV prediction files")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Folder to save analysis results")
parser.add_argument("--length_unit", choices=["chars", "words"], default="chars",
                    help="Unit for comment length (default: chars)")
parser.add_argument("--n_buckets", type=int, default=4,
                    help="Number of length buckets (default: 4)")
args = parser.parse_args()

# ===== CREATE OUTPUT FOLDER =====
os.makedirs(args.output_dir, exist_ok=True)

# ===== LOAD AND CONCAT CSV FILES =====
all_files = glob.glob("results/Zero_Shot_10runs/*.csv")
if not all_files:
    raise ValueError(f"No CSV files found in {args.predictions_dir}")

dfs = [pd.read_csv(f) for f in all_files]
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} predictions from {len(all_files)} CSV files.")

# ===== COMPUTE LENGTH =====
if args.length_unit == "chars":
    df["length"] = df["text"].astype(str).apply(len)
else:  # words
    df["length"] = df["text"].astype(str).apply(lambda x: len(x.split()))

# ===== BUCKETIZE =====
df["length_bucket"] = pd.qcut(df["length"], q=args.n_buckets)
metrics_per_bucket = []

for bucket, group in df.groupby("length_bucket"):
    y_true = group["ground_truth"]
    y_pred = group["predicted_zero_sum"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics_per_bucket.append({
        "bucket": str(bucket),
        "median_length": group["length"].median(),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

metrics_df = pd.DataFrame(metrics_per_bucket)

# ===== SPEARMAN CORRELATION =====
for metric in ["accuracy", "precision", "recall", "f1"]:
    rho, pval = spearmanr(metrics_df["median_length"], metrics_df[metric])
    print(f"{metric} vs length: Spearman rho={rho:.3f}, p={pval:.3f}")

# ===== SAVE RESULTS =====
metrics_csv_path = os.path.join(args.output_dir, f"length_metrics_{args.length_unit}.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Metrics per length bucket saved to {metrics_csv_path}")

# ===== PLOT =====
plt.figure(figsize=(8, 5))
metrics_df.set_index("median_length")[["accuracy", "precision", "recall", "f1"]].plot(marker='o')
plt.xlabel(f"Median Comment Length ({args.length_unit})")
plt.ylabel("Metric")
plt.title("Model Performance vs Comment Length")
plt.grid(True)
plot_path = os.path.join(args.output_dir, f"length_metrics_{args.length_unit}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to {plot_path}")
