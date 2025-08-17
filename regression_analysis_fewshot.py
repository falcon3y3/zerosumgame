#regression_analysis_fewshot.py
# regression_analysis_fewshot_by_order.py
# Infer few-shot k by chronological order of files (e.g., first 10 => k=8, next 10 => k=20, next 10 => k=143)

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# Global styling
sns.set_theme(style="whitegrid", context="talk")  
sns.set_palette("colorblind")  # Options: "Set2", "muted", "colorblind", "Paired", "pastel"

# ---------- helpers ----------
F1_CANDIDATES = ["F1 Score", "F1", "f1", "f1_score", "F1_Score"]

def find_f1_col(df: pd.DataFrame) -> str:
    for c in F1_CANDIDATES:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for alias in ["f1 score", "f1", "f1_score", "f1score"]:
        if alias in lower_map:
            return lower_map[alias]
    raise KeyError(f"No F1 column found. Tried: {', '.join(F1_CANDIDATES)}")

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def parse_ts_from_name(path: str):
    """Try to extract timestamp from filename; fallback to mtime."""
    base = os.path.basename(path)
    # Patterns: YYYY-MM-DD_HH-MM-SS or YYYYMMDD_HHMMSS
    m1 = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", base)
    if m1:
        try:
            return datetime.strptime(m1.group(0), "%Y-%m-%d_%H-%M-%S").timestamp()
        except Exception:
            pass
    m2 = re.search(r"\d{8}_\d{6}", base)
    if m2:
        try:
            return datetime.strptime(m2.group(0), "%Y%m%d_%H%M%S").timestamp()
        except Exception:
            pass
    # Fallback: file modification time
    return os.path.getmtime(path)

def assign_k_by_order(n_files: int, ks: list, counts: list):
    """Return list of k values length n_files based on block counts."""
    if len(ks) != len(counts):
        raise ValueError("ks and counts must have same length")
    seq = []
    for k, c in zip(ks, counts):
        seq.extend([k] * c)
    # If there are fewer/more files than sum(counts), trim or extend last k
    if n_files <= len(seq):
        return seq[:n_files]
    else:
        # extend with last k value
        seq.extend([ks[-1]] * (n_files - len(seq)))
        return seq

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Regression of F1 vs few-shot k inferred by run order.")
parser.add_argument("--metrics_dir", type=str, required=True,
                    help="Directory containing per-run metrics CSVs.")
parser.add_argument("--pattern", type=str, default="*metrics*.csv",
                    help="Glob pattern to match metrics files (default '*metrics*.csv').")
parser.add_argument("--output_dir", type=str, default="results/fewshot_regression_analysis",
                    help="Directory to save outputs.")
parser.add_argument("--ks", type=int, nargs="+", default=[8, 20, 143],
                    help="Few-shot k values in sequence (default: 8 20 143).")
parser.add_argument("--counts", type=int, nargs="+", default=[10, 10, 10],
                    help="Number of runs per k, in same order as --ks (default: 10 10 10).")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------- load files and sort by time ----------
files = sorted(glob.glob(os.path.join(args.metrics_dir, args.pattern)))
if not files:
    raise ValueError(f"No files matched: {os.path.join(args.metrics_dir, args.pattern)}")

# Sort by parsed timestamp (or mtime fallback)
files_with_time = [(f, parse_ts_from_name(f)) for f in files]
files_sorted = [f for f, _ in sorted(files_with_time, key=lambda x: x[1])]

# Assign k by order
k_sequence = assign_k_by_order(len(files_sorted), ks=args.ks, counts=args.counts)

# ---------- collect rows ----------
rows = []
for f, k in zip(files_sorted, k_sequence):
    try:
        dfm = pd.read_csv(f)
    except Exception as e:
        print(f"[skip read error] {f}: {e}")
        continue
    if dfm.empty:
        print(f"[skip empty] {f}")
        continue

    try:
        f1_col = find_f1_col(dfm)
    except KeyError as e:
        print(f"[skip missing F1] {f}: {e}")
        continue

    f1_val = coerce_numeric(dfm[f1_col]).iloc[0]
    if pd.isna(f1_val):
        print(f"[skip non-numeric F1] {f}")
        continue

    # Optional: capture other metrics if present
    acc = dfm.filter(regex=r"^Accuracy$", axis=1)
    prec = dfm.filter(regex=r"^Precision$", axis=1)
    rec = dfm.filter(regex=r"^Recall$", axis=1)

    rows.append({
        "run_file": os.path.basename(f),
        "few_shot_k": int(k),
        "F1": float(f1_val),
        "Accuracy": float(coerce_numeric(acc.iloc[0,0])) if not acc.empty else np.nan,
        "Precision": float(coerce_numeric(prec.iloc[0,0])) if not prec.empty else np.nan,
        "Recall": float(coerce_numeric(rec.iloc[0,0])) if not rec.empty else np.nan,
    })

data = pd.DataFrame(rows)
if data.empty:
    raise RuntimeError("No valid runs collected after filtering; check files and pattern.")

# Keep only valid ranges
data = data[(data["few_shot_k"] > 0) & (data["F1"].between(0, 1, inclusive="both"))]
data["log_k"] = np.log(data["few_shot_k"])

# ---------- regression & ANOVA ----------
ols_model = smf.ols("F1 ~ log_k", data=data).fit()
anova_model = smf.ols("F1 ~ C(few_shot_k)", data=data).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)



print("\n=== OLS: F1 ~ log(k) ===")
print(ols_model.summary())
print("\n=== ANOVA: F1 ~ C(k) ===")
print(anova_table)

# Save to txt file
with open(os.path.join(args.output_dir, "anova_results.txt"), "w") as f:
    f.write("=== ANOVA: F1 ~ C(few_shot_k) ===\n")
    f.write(str(anova_table))
    
# --- Post-hoc pairwise comparisons: Tukey HSD ---
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import os

# Run Tukey HSD on F1 across the k groups
tukey = pairwise_tukeyhsd(endog=data["F1"], groups=data["few_shot_k"], alpha=0.05)
print("\n=== Tukey HSD: pairwise differences in F1 by k ===")
print(tukey.summary())

# (Optional) save a tidy CSV of the Tukey results
tukey_df = pd.DataFrame(
    tukey._results_table.data[1:],  # skip header row
    columns=tukey._results_table.data[0]
)
tukey_csv_path = os.path.join(args.output_dir if "args" in globals() else ".", "tukey_f1_by_k.csv")
tukey_df.to_csv(tukey_csv_path, index=False)
print(f"Tukey results saved to: {tukey_csv_path}")

# ---------- save tables ----------
data_path = os.path.join(args.output_dir, "fewshot_runs_f1_inferred_by_order.csv")
anova_path = os.path.join(args.output_dir, "anova_f1_by_k.csv")
ols_summary_path = os.path.join(args.output_dir, "ols_summary.txt")

data.to_csv(data_path, index=False)
anova_table.to_csv(anova_path, index=False)
with open(ols_summary_path, "w") as fh:
    fh.write(ols_model.summary().as_text())

# ---------- plots ----------
# Scatter + OLS fit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x="few_shot_k", y="F1", hue="few_shot_k", s=100)
k_sorted = np.sort(data["few_shot_k"].unique())
fit_df = pd.DataFrame({"few_shot_k": k_sorted, "log_k": np.log(k_sorted)})
fit_df["F1_fit"] = ols_model.predict(fit_df)
plt.plot(k_sorted, fit_df["F1_fit"], color="black", label="OLS fit (F1 ~ log k)")
plt.xscale("log")
plt.xticks(k_sorted, [str(k) for k in k_sorted])
plt.xlabel("Few-shot Examples (log scale)")
plt.ylabel("F1")
plt.title("F1 vs Few-shot Example Conditions")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "f1_vs_k_regression.png"), dpi=300)
plt.close()

# Boxplot per k
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x="few_shot_k", y="F1", palette="Set2")
sns.stripplot(data=data, x="few_shot_k", y="F1", color="black", alpha=0.5)
plt.xlabel("Few-shot Examples")
plt.ylabel("F1")
plt.title("F1 Distribution by Few-shot Example Conditions")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "f1_by_k_boxplot.png"), dpi=300)
plt.close()

print(f"\n✅ Done. Outputs in: {args.output_dir}")
print(f"- Cleaned data: {data_path}")
print(f"- OLS summary:  {ols_summary_path}")
print(f"- ANOVA table:  {anova_path}")
