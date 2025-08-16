#analysis pipeline for text length versus LLM prediction accuracy using generated predictions 
#python retro_length_analysis.py --predictions results/zero_shot_predictions.csv --output_dir results/length_analysis_zero_shot
#python retro_length_analysis.py --predictions results/fewshot_predictions.csv --output_dir results/length_analysis_few_shot
# retro_length_analysis_multi.py
# Merge many prediction CSVs, analyze whether text length impacts accuracy,
# and compute bootstrap CIs for accuracy by length bucket.

import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ----------------------------
# Bootstrap helper
# ----------------------------
def bootstrap_ci_binary_accuracy(values, n_boot=2000, ci=95, rng=None):
    """
    values: array-like of 0/1 correctness for a given bucket
    n_boot: number of bootstrap resamples
    ci: confidence level (e.g., 95)
    rng: np.random.Generator for reproducibility (optional)
    Returns: mean, lower, upper
    """
    values = np.asarray(values)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()

    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(sample.mean())
    means = np.array(means)

    mean_point = float(values.mean())
    alpha = (100 - ci) / 2
    lower = np.percentile(means, alpha)
    upper = np.percentile(means, 100 - alpha)
    return mean_point, float(lower), float(upper)

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Analyze accuracy vs text length across multiple prediction CSVs.")
parser.add_argument("--predictions_dir", type=str, required=True,
                    help="Directory containing predictions CSVs (each run saved separately).")
parser.add_argument("--output_dir", type=str, default="results/length_analysis",
                    help="Directory to save analysis outputs.")
parser.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap resamples.")
parser.add_argument("--ci", type=float, default=95.0, help="Confidence interval level (e.g., 95).")
parser.add_argument("--length_unit", choices=["chars", "words"], default="chars",
                    help="Measure length by characters or words.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ----------------------------
# Load all runs
# ----------------------------
all_files = glob.glob("results/Zero_Shot_10runs/*.csv")
if not all_files:
    raise ValueError(f"No CSV files found in {args.predictions_dir}")

dfs = []
for f in all_files:
    df_run = pd.read_csv(f)
    # Minimal schema check
    missing = {"text", "ground_truth", "predicted_zero_sum"} - set(df_run.columns)
    if missing:
        raise ValueError(f"{os.path.basename(f)} is missing required columns: {missing}")
    df_run["run_id"] = os.path.basename(f).replace(".csv", "")
    dfs.append(df_run)

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} predictions from {len(all_files)} files.")

# ----------------------------
# Prepare
# ----------------------------
df = df.dropna(subset=["predicted_zero_sum"])
# Ensure ints
df["ground_truth"] = df["ground_truth"].astype(int)
df["predicted_zero_sum"] = df["predicted_zero_sum"].astype(int)
df["correct"] = (df["predicted_zero_sum"] == df["ground_truth"]).astype(int)

# Text length
if args.length_unit == "words":
    df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
else:
    df["text_length"] = df["text"].str.len()

# Create quantile buckets (5 bins by default); drop duplicates if needed
df["length_bucket"] = pd.qcut(df["text_length"], q=5, duplicates="drop")

# ----------------------------
# Overall accuracy by bucket + bootstrap CIs
# ----------------------------
bucket_group = df.groupby("length_bucket", observed=True)
bucket_rows = []
rng = np.random.default_rng()  # one RNG for all CIs

for bucket, sub in bucket_group:
    mean_acc, lo, hi = bootstrap_ci_binary_accuracy(
        sub["correct"].values, n_boot=args.n_boot, ci=args.ci, rng=rng
    )
    bucket_rows.append({
        "length_bucket": str(bucket),
        "n": len(sub),
        "accuracy": mean_acc,
        f"ci_lower_{int(args.ci)}": lo,
        f"ci_upper_{int(args.ci)}": hi
    })

bucket_ci_df = pd.DataFrame(bucket_rows)
bucket_ci_path = os.path.join(args.output_dir, "accuracy_by_length_bucket_with_CI.csv")
bucket_ci_df.to_csv(bucket_ci_path, index=False)

# ----------------------------
# Per-run accuracy by bucket (no CI here, distribution shown via box/violin)
# ----------------------------
per_run_acc = df.groupby(["run_id", "length_bucket"], observed=True)["correct"].mean().reset_index()
per_run_acc_path = os.path.join(args.output_dir, "per_run_accuracy_by_length_bucket.csv")
per_run_acc.to_csv(per_run_acc_path, index=False)

# ----------------------------
# Correlations (length vs correctness)
# ----------------------------
pearson_corr, pearson_p = pearsonr(df["text_length"], df["correct"])
spearman_corr, spearman_p = spearmanr(df["text_length"], df["correct"])
corr_results = pd.DataFrame([{
    "length_unit": args.length_unit,
    "Pearson_r": pearson_corr,
    "Pearson_p": pearson_p,
    "Spearman_rho": spearman_corr,
    "Spearman_p": spearman_p
}])
corr_path = os.path.join(args.output_dir, "correlation_results.csv")
corr_results.to_csv(corr_path, index=False)

print("\n=== Correlation Results ===")
print(corr_results.to_string(index=False))

# ----------------------------
# PLOTS
# ----------------------------

# 1) Scatter: length vs correctness
plt.figure(figsize=(8, 6))
plt.scatter(df["text_length"], df["correct"], alpha=0.3)
plt.xlabel(f"Comment Length ({args.length_unit})")
plt.ylabel("Correct (1) / Incorrect (0)")
plt.title("Prediction Accuracy vs Comment Length (All Runs)")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "accuracy_vs_length.png"), dpi=300)
plt.close()

# 2) Per-run distribution by bucket (boxplot + violin)
plt.figure(figsize=(10, 6))
sns.boxplot(data=per_run_acc, x="length_bucket", y="correct")
plt.xticks(rotation=45)
plt.xlabel("Text Length Bucket")
plt.ylabel("Accuracy")
plt.title("Per-Run Accuracy by Text Length Bucket (Boxplot)")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "per_run_accuracy_boxplot.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.violinplot(data=per_run_acc, x="length_bucket", y="correct", inner="quartile")
plt.xticks(rotation=45)
plt.xlabel("Text Length Bucket")
plt.ylabel("Accuracy")
plt.title("Per-Run Accuracy by Text Length Bucket (Violin)")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "per_run_accuracy_violin.png"), dpi=300)
plt.close()

# 3) Accuracy by bucket with bootstrap CIs (error bars)
# Sort buckets by their left edge for nicer ordering on x-axis
# Convert interval string back to an orderable key
def bucket_sort_key(s):
    # s looks like "(a, b]" or "[a, b]"
    # Extract numbers safely
    txt = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    left = txt.split(",")[0].strip()
    try:
        return float(left)
    except:
        return 0.0

plot_df = bucket_ci_df.copy()
plot_df["sort_key"] = plot_df["length_bucket"].apply(bucket_sort_key)
plot_df = plot_df.sort_values("sort_key", ascending=True)

x = np.arange(len(plot_df))
y = plot_df["accuracy"].values
yerr_lower = y - plot_df[f"ci_lower_{int(args.ci)}"].values
yerr_upper = plot_df[f"ci_upper_{int(args.ci)}"].values - y
yerr = np.vstack([yerr_lower, yerr_upper])

plt.figure(figsize=(10, 6))
plt.bar(x, y)
plt.errorbar(x, y, yerr=yerr, fmt='none', capsize=4, linewidth=1)
plt.xticks(x, plot_df["length_bucket"], rotation=45, ha="right")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.xlabel("Text Length Bucket")
plt.title(f"Accuracy by Length Bucket with {int(args.ci)}% Bootstrap CIs")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "accuracy_by_length_bucket_with_CI.png"), dpi=300)
plt.close()

print("\n✅ Analysis complete.")
print(f"- Overall CI CSV: {bucket_ci_path}")
print(f"- Per-run bucket CSV: {per_run_acc_path}")
print(f"- Correlations CSV: {corr_path}")
print(f"- Plots saved in: {args.output_dir}")
