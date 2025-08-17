# compare_f1_fewshot.py
# Compare F1 scores across several runs of few-shot prompting.

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- helpers ----------
def find_f1_column(df):
    candidates = ["F1 Score", "F1", "f1", "f1_score", "F1_Score"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No F1 column found. Expected one of: " + ", ".join(candidates))

def find_model_type_column(df):
    for c in ["Model_Type", "model_type", "mode", "Mode"]:
        if c in df.columns:
            return c
    return None

def coerce_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    return s

def bootstrap_ci(values, n_boot=5000, ci=95, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    n = len(values)
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    mean = float(values.mean())
    alpha = (100 - ci) / 2
    lower = np.percentile(boots, alpha)
    upper = np.percentile(boots, 100 - alpha)
    return mean, float(lower), float(upper)

def parse_timestamp_from_filename(fname):
    """
    Try to extract a timestamp like 2025-08-16_15-42-01 or 20250816_154201 from filename.
    Returns string or None.
    """
    base = os.path.basename(fname)
    # patterns: YYYY-MM-DD_HH-MM-SS OR YYYYMMDD_HHMMSS
    m1 = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", base)
    if m1:
        return m1.group(0)
    m2 = re.search(r"\d{8}_\d{6}", base)
    if m2:
        return m2.group(0)
    return None

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Compare F1 scores over multiple few-shot runs.")
parser.add_argument("--metrics_dir", type=str, required=True,
                    help="Directory containing per-run metrics CSV files.")
parser.add_argument("--pattern", type=str, default="*fewshot*metrics*.csv",
                    help="Glob pattern to match metrics files in metrics_dir "
                         "(default: '*fewshot*metrics*.csv').")
parser.add_argument("--output_dir", type=str, default="results/f1_comparison",
                    help="Directory to write combined CSV, summary, and plots.")
parser.add_argument("--require_fewshot_tag", action="store_true",
                    help="If set, will only keep rows where Model_Type == 'few-shot' (if column exists).")
parser.add_argument("--n_boot", type=int, default=5000, help="Bootstrap resamples for CI (default 5000).")
parser.add_argument("--ci", type=float, default=95.0, help="Confidence interval percent (default 95).")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------- load files ----------
search_glob = os.path.join(args.metrics_dir, args.pattern)
files = sorted(glob.glob(search_glob))
if not files:
    raise ValueError(f"No files matched pattern:\n  {search_glob}")

rows = []
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Skipping {f} (read error): {e}")
        continue

    # optionally filter to few-shot based on a model type column
    mt_col = find_model_type_column(df)
    if args.require_fewshot_tag and mt_col:
        df = df[df[mt_col].astype(str).str.lower() == "few-shot"]

    # get F1 column
    try:
        f1_col = find_f1_column(df)
    except ValueError as e:
        print(f"Skipping {f}: {e}")
        continue

    # take the first (or only) row as per-run metrics
    # (your pipeline writes a single-row metrics CSV per run)
    if df.empty:
        print(f"Skipping {f}: empty metrics file.")
        continue

    # normalize / coerce F1 numeric
    f1_vals = coerce_numeric(df[f1_col])
    if f1_vals.isna().all():
        print(f"Skipping {f}: F1 column not numeric.")
        continue
    f1 = float(f1_vals.iloc[0])

    # get timestamp if exists as a column or from filename
    ts = None
    for c in ["timestamp", "Timestamp", "run_timestamp"]:
        if c in df.columns and isinstance(df[c].iloc[0], str):
            ts = df[c].iloc[0]
            break
    if ts is None:
        ts = parse_timestamp_from_filename(f) or ""

    rows.append({
        "run_id": os.path.splitext(os.path.basename(f))[0],
        "file": f,
        "timestamp": ts,
        "F1": f1
    })

if not rows:
    raise ValueError("No valid F1 rows collected. Check your pattern or files.")

combined = pd.DataFrame(rows)

# ---------- summary stats + CI ----------
mean_f1, lo_f1, hi_f1 = bootstrap_ci(combined["F1"].values, n_boot=args.n_boot, ci=args.ci)

summary = pd.DataFrame([{
    "n_runs": len(combined),
    "F1_mean": mean_f1,
    f"F1_CI{int(args.ci)}_lower": lo_f1,
    f"F1_CI{int(args.ci)}_upper": hi_f1,
    "F1_std": float(np.std(combined["F1"].values, ddof=1)) if len(combined) > 1 else 0.0
}])

# ---------- save CSVs ----------
combined_csv = os.path.join(args.output_dir, "fewshot_f1_runs.csv")
summary_csv = os.path.join(args.output_dir, "fewshot_f1_summary.csv")
combined.to_csv(combined_csv, index=False)
summary.to_csv(summary_csv, index=False)
print(f"Saved per-run F1: {combined_csv}")
print(f"Saved summary:    {summary_csv}")

# ---------- plots ----------
# Boxplot
plt.figure(figsize=(6, 5))
sns.boxplot(y=combined["F1"])
sns.stripplot(y=combined["F1"], alpha=0.5, jitter=0.15)
plt.ylabel("F1")
plt.title("Few-Shot F1 Distribution Across Runs")
plt.ylim(0, 1)
boxplot_path = os.path.join(args.output_dir, "fewshot_f1_boxplot.png")
plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved boxplot:     {boxplot_path}")

# Line plot by timestamp if available; otherwise by run index
plot_df = combined.copy()
# Use timestamp for sorting if non-empty; else use index
if plot_df["timestamp"].astype(str).str.len().max() > 0:
    sort_key = plot_df["timestamp"].astype(str)
else:
    sort_key = pd.Series(range(len(plot_df)), name="order")

plot_df = plot_df.assign(_sort=sort_key).sort_values("_sort").reset_index(drop=True)
# Line plot ordered by run index (1, 2, ... N)
plt.figure(figsize=(9, 4))
plt.plot(range(1, len(combined) + 1), combined["F1"], marker="o")
plt.xticks(range(1, len(combined) + 1))  # tick labels = 1..N
plt.ylabel("F1")
plt.xlabel("Run")
plt.ylim(0, 1)
plt.title("Few-Shot F1 Across Runs (by index)")
lineplot_path = os.path.join(args.output_dir, "fewshot_f1_line.png")
plt.tight_layout()
plt.savefig(lineplot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved line plot:   {lineplot_path}")

# Bar + CI overlay (from bootstrap)
plt.figure(figsize=(5, 4))
plt.bar([0], [mean_f1], width=0.5)
plt.errorbar([0], [mean_f1], 
             yerr=[[mean_f1 - lo_f1], [hi_f1 - mean_f1]],
             fmt="none", capsize=6, linewidth=1.5)
plt.xticks([0], [f"Few-Shot (n={len(combined)})"])
plt.ylim(0, 1)
plt.ylabel("F1")
plt.title(f"Few-Shot Mean F1 with {int(args.ci)}% CI")
ci_plot_path = os.path.join(args.output_dir, "fewshot_f1_mean_ci.png")
plt.savefig(ci_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved CI plot:     {ci_plot_path}")
