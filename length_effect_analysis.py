# length_effect_analysis.py
# Analyze whether Reddit comment length affects performance (correctness/F1)
# across ZERO vs FEW conditions and within FEW by k ∈ {8,20,143}.
# Inputs: a directory of prediction CSVs from your runs.

import os
import re
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import f1_score

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Test effect of comment length on performance.")
parser.add_argument("--predictions_dir", type=str, required=True,
                    help="Folder containing per-run prediction CSVs.")
parser.add_argument("--pattern", type=str, default="*.csv",
                    help="Glob pattern to match prediction files (default: *.csv).")
parser.add_argument("--output_dir", type=str, default="results/length_effect",
                    help="Folder to write outputs.")
parser.add_argument("--length_unit", choices=["words", "chars"], default="words",
                    help="Measure length by words (default) or characters.")
parser.add_argument("--bins", type=int, default=5,
                    help="Number of quantile bins for F1-by-length plots (default: 5).")
parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility of qcut ties).")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------- Styling ----------
sns.set_theme(style="whitegrid", context="talk")
sns.set_palette("Set2")

# ---------- Helpers ----------
PRED_COL_CANDIDATES = ["predicted_zero_sum", "llm_zero_sum", "prediction", "pred"]
GT_COL_CANDIDATES   = ["ground_truth", "zero_sum", "label", "target"]
TEXT_COL_CANDIDATES = ["text", "comment", "body"]

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Required column not found. Looked for: {candidates}")

def infer_condition_from(df, fname):
    # Prefer an explicit column, else infer from filename
    for c in ["Model_Type", "model_type", "mode", "condition"]:
        if c in df.columns:
            val = str(df[c].iloc[0]).strip().lower()
            if "zero" in val:
                return "zero"
            if "few" in val:
                return "few"
    base = os.path.basename(fname).lower()
    if "zero" in base:
        return "zero"
    if "few" in base:
        return "few"
    # fallback: treat as few-shot if 'k' parseable, else 'unknown'
    return "few" if infer_k_from_name(fname) is not None else "unknown"

def infer_k_from_name(fname):
    base = os.path.basename(fname).lower()
    # patterns: fewshot_8, k20, shots143, etc.
    patterns = [
        r"fewshot[_\-]?(\d+)",
        r"[_\-]k(?:=)?(\d+)",
        r"shots?(?:=)?(\d+)",
        r"nshots?(?:=)?(\d+)",
        r"examples?(?:=)?(\d+)",
        r"[_\-](8|20|143)(?:[_\-]|\.csv$)"
    ]
    for pat in patterns:
        m = re.search(pat, base)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    # check columns later if present
    return None

def ensure_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

def lr_test(smaller, bigger, df_diff=None):
    """Likelihood Ratio test comparing two nested models (statsmodels results)."""
    llf_small = smaller.llf
    llf_big   = bigger.llf
    if df_diff is None:
        df_diff = bigger.df_model - smaller.df_model
    chi2_stat = 2 * (llf_big - llf_small)
    p = 1 - chi2.cdf(chi2_stat, df_diff)
    return chi2_stat, df_diff, p

# ---------- Load & concatenate ----------
files = sorted(glob.glob(os.path.join(args.predictions_dir, args.pattern)))
if not files:
    raise ValueError(f"No files matched: {os.path.join(args.predictions_dir, args.pattern)}")

rows = []
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[skip read error] {f}: {e}")
        continue
    if df.empty:
        print(f"[skip empty] {f}")
        continue

    try:
        pred_col = find_col(df, PRED_COL_CANDIDATES)
        gt_col   = find_col(df, GT_COL_CANDIDATES)
        text_col = find_col(df, TEXT_COL_CANDIDATES)
    except KeyError as e:
        print(f"[skip missing columns] {f}: {e}")
        continue

    # Normalize columns
    df = df[[text_col, gt_col, pred_col]].copy()
    df.rename(columns={text_col: "text", gt_col: "ground_truth", pred_col: "predicted"}, inplace=True)
    df["predicted"]    = ensure_numeric_series(df["predicted"])
    df["ground_truth"] = ensure_numeric_series(df["ground_truth"])

    # Drop bad rows
    df = df.dropna(subset=["predicted", "ground_truth", "text"])
    if df.empty:
        print(f"[skip no valid rows] {f}")
        continue

    # Cast to ints
    df["predicted"]    = df["predicted"].astype(int)
    df["ground_truth"] = df["ground_truth"].astype(int)

    # Add metadata
    cond = infer_condition_from(df, f)
    k = None
    if cond == "few":
        # Try to infer k from name; if not, look for a column
        k = infer_k_from_name(f)
        if k is None:
            for kc in ["few_shot_k", "k", "n_shots", "examples"]:
                if kc in df.columns:
                    try:
                        k = int(pd.to_numeric(df[kc].iloc[0], errors="coerce"))
                        break
                    except Exception:
                        pass
    df["condition"] = cond
    df["k"] = k

    # Track source file
    df["run_file"] = os.path.basename(f)
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
if data.empty:
    raise RuntimeError("No valid data after loading.")

# ---------- Length & correctness ----------
if args.length_unit == "words":
    data["length"] = data["text"].astype(str).apply(lambda x: len(x.split()))
else:
    data["length"] = data["text"].astype(str).str.len()

# log1p length to reduce skew; also keep raw for binning/plots
data["length_log"] = np.log1p(data["length"])
data["correct"] = (data["predicted"] == data["ground_truth"]).astype(int)

# Filter to only known conditions
data = data[data["condition"].isin(["zero", "few"])].copy()
if data.empty:
    raise RuntimeError("No rows with recognized condition ('zero' or 'few').")

# ---------- MODEL 1: Across zero vs few (interaction) ----------
# Additive model (no interaction): correct ~ length_log + C(condition)
m_add = smf.logit("correct ~ length_log + C(condition)", data=data).fit(disp=False)
# Full model (interaction): correct ~ length_log * C(condition)
m_int = smf.logit("correct ~ length_log * C(condition)", data=data).fit(disp=False)

# LRT for interaction
chi2_stat, df_diff, p_int = lr_test(m_add, m_int)
# LRT for main length effect: compare C(condition)-only vs additive
m_cond_only = smf.logit("correct ~ C(condition)", data=data).fit(disp=False)
chi2_len, df_len, p_len = lr_test(m_cond_only, m_add)

# Save summaries
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(os.path.join(args.output_dir, f"length_effect_zero_vs_few_{ts}.txt"), "w") as fh:
    fh.write("=== ZERO vs FEW: Logistic regression on correctness ===\n\n")
    fh.write("[Additive model] correct ~ length_log + C(condition)\n")
    fh.write(m_add.summary().as_text() + "\n\n")
    fh.write("[Interaction model] correct ~ length_log * C(condition)\n")
    fh.write(m_int.summary().as_text() + "\n\n")
    fh.write(f"[LRT: interaction] chi2={chi2_stat:.3f}, df={int(df_diff)}, p={p_int:.6f}\n")
    fh.write(f"[LRT: length main effect] chi2={chi2_len:.3f}, df={int(df_len)}, p={p_len:.6f}\n")

# ---------- MODEL 2: Within FEW only, by k ∈ {8,20,143} ----------
few = data[(data["condition"] == "few") & (data["k"].isin([8, 20, 143]))].copy()
if few.empty:
    print("[warn] No few-shot rows with k in {8,20,143} found; skipping within-few analysis.")
else:
    # Additive: correct ~ length_log + C(k)
    mf_add = smf.logit("correct ~ length_log + C(k)", data=few).fit(disp=False)
    # Interaction: correct ~ length_log * C(k)
    mf_int = smf.logit("correct ~ length_log * C(k)", data=few).fit(disp=False)
    # No-length: correct ~ C(k)
    mf_konly = smf.logit("correct ~ C(k)", data=few).fit(disp=False)

    chi2_int, df_int, p_int_k = lr_test(mf_add, mf_int)
    chi2_len_k, df_len_k, p_len_k = lr_test(mf_konly, mf_add)

    with open(os.path.join(args.output_dir, f"length_effect_within_few_{ts}.txt"), "w") as fh:
        fh.write("=== FEW only (k ∈ {8,20,143}): Logistic regression on correctness ===\n\n")
        fh.write("[Additive] correct ~ length_log + C(k)\n")
        fh.write(mf_add.summary().as_text() + "\n\n")
        fh.write("[Interaction] correct ~ length_log * C(k)\n")
        fh.write(mf_int.summary().as_text() + "\n\n")
        fh.write(f"[LRT: interaction] chi2={chi2_int:.3f}, df={int(df_int)}, p={p_int_k:.6f}\n")
        fh.write(f"[LRT: length main effect] chi2={chi2_len_k:.3f}, df={int(df_len_k)}, p={p_len_k:.6f}\n")

# ---------- F1 by length bins (overall & stratified) ----------
rng = np.random.default_rng(args.seed)

def f1_by_bins(df, group_cols, bins=args.bins):
    out = []
    # Quantile bins on raw length within the *full* df for comparability
    # (Alternatively, compute bins within each subgroup; here we keep global bins)
    df = df.copy()
    df["length_bin"] = pd.qcut(df["length"], q=bins, duplicates="drop")
    for keys, sub in df.groupby(group_cols + ["length_bin"], dropna=False):
        y_true = sub["ground_truth"]
        y_pred = sub["predicted"]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        out.append({
            **({k: v for k, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}),
            "length_bin": str(sub["length_bin"].iloc[0]),
            "n": len(sub),
            "F1": f1,
            "median_len": sub["length"].median()
        })
    return pd.DataFrame(out)

# Overall by condition
f1_cond = f1_by_bins(data, ["condition"], bins=args.bins)
f1_cond.to_csv(os.path.join(args.output_dir, f"f1_by_length_bins_condition_{ts}.csv"), index=False)

# FEW only by k
if not few.empty:
    f1_k = f1_by_bins(few, ["k"], bins=args.bins)
    f1_k.to_csv(os.path.join(args.output_dir, f"f1_by_length_bins_k_{ts}.csv"), index=False)

# ---------- PLOTS ----------
# 1) Logistic curves: zero vs few
plt.figure(figsize=(9, 6))
sns.regplot(
    data=data[data["condition"]=="zero"], x="length", y="correct",
    logistic=True, ci=None, scatter_kws={"alpha":0.25}, line_kws={"lw":3}, label="Zero-shot"
)
sns.regplot(
    data=data[data["condition"]=="few"], x="length", y="correct",
    logistic=True, ci=None, scatter_kws={"alpha":0.25}, line_kws={"lw":3}, label="Few-shot"
)
plt.xlabel(f"Comment length ({args.length_unit})")
plt.ylabel("P(correct)")
plt.title("Probability of Correct Classification vs Length\nZero-shot vs Few-shot")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f"logit_zero_vs_few_{ts}.png"), dpi=300)
plt.close()

# 2) Logistic curves within FEW by k
if not few.empty:
    plt.figure(figsize=(9, 6))
    for k_val, sub in few.groupby("k"):
        sns.regplot(
            data=sub, x="length", y="correct",
            logistic=True, ci=None, scatter_kws={"alpha":0.20},
            line_kws={"lw":3}, label=f"k={k_val}"
        )
    plt.xlabel(f"Comment length ({args.length_unit})")
    plt.ylabel("P(correct)")
    plt.title("Probability of Correct Classification vs Length (Few-shot by k)")
    plt.legend(title="Few-shot k")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"logit_few_by_k_{ts}.png"), dpi=300)
    plt.close()

# 3) F1 by length bins (condition)
plt.figure(figsize=(9, 6))
for cond, sub in f1_cond.groupby("condition"):
    sub = sub.sort_values("median_len")
    plt.plot(sub["median_len"], sub["F1"], marker="o", linewidth=2, label=cond.title())
plt.xlabel(f"Median length of bin ({args.length_unit})")
plt.ylabel("F1")
plt.title("F1 by Length Bins (Zero vs Few)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f"f1_bins_condition_{ts}.png"), dpi=300)
plt.close()

# 4) F1 by length bins (few-shot k)
if not few.empty:
    plt.figure(figsize=(9, 6))
    for k_val, sub in f1_k.groupby("k"):
        sub = sub.sort_values("median_len")
        plt.plot(sub["median_len"], sub["F1"], marker="o", linewidth=2, label=f"k={k_val}")
    plt.xlabel(f"Median length of bin ({args.length_unit})")
    plt.ylabel("F1")
    plt.title("F1 by Length Bins (Few-shot by k)")
    plt.legend(title="Few-shot k")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"f1_bins_k_{ts}.png"), dpi=300)
    plt.close()

# ---------- Save a quick summary CSV of tests ----------
summary_rows = [{
    "test": "ZERO vs FEW: interaction (length × condition)",
    "chi2": chi2_stat, "df": int(df_diff), "p_value": p_int
},{
    "test": "ZERO vs FEW: length main effect",
    "chi2": chi2_len, "df": int(df_len), "p_value": p_len
}]
if not few.empty:
    summary_rows += [{
        "test": "FEW only: interaction (length × k)",
        "chi2": chi2_int, "df": int(df_int), "p_value": p_int_k
    },{
        "test": "FEW only: length main effect",
        "chi2": chi2_len_k, "df": int(df_len_k), "p_value": p_len_k
    }]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(args.output_dir, f"significance_tests_{ts}.csv"), index=False)

print("\n✅ Done. Key outputs saved to:", args.output_dir)
print("- Logistic model summaries: *_zero_vs_few_*.txt and *_within_few_*.txt")
print("- Significance table: significance_tests_*.csv")
print("- Plots: logit_* and f1_bins_* PNGs")
