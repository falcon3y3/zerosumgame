import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from glob import glob

# ===== ARGUMENTS =====
parser = argparse.ArgumentParser()
parser.add_argument("--predictions_dir", type=str, required=True, help="Directory containing prediction CSVs")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
parser.add_argument("--length_unit", choices=["chars", "words"], default="chars", help="Use character or word length")
args = parser.parse_args()

# ===== CREATE OUTPUT FOLDER =====
os.makedirs(args.output_dir, exist_ok=True)

# ===== LOAD & CONCAT CSVs =====
all_files = glob(os.path.join(args.predictions_dir, "*.csv"))
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# ===== FILTER OUT INVALIDS =====
df = df.dropna(subset=["text", "ground_truth", "predicted_zero_sum"])

# ===== COMPUTE TEXT LENGTH =====
if args.length_unit == "chars":
    df["text_length"] = df["text"].astype(str).str.len()
else:
    df["text_length"] = df["text"].astype(str).str.split().apply(len)

# ===== COMPUTE F1 PER COMMENT =====
def compute_f1(row):
    if row["ground_truth"] == 1 and row["predicted_zero_sum"] == 1:
        return 1.0
    elif row["ground_truth"] == 0 and row["predicted_zero_sum"] == 0:
        return 1.0
    elif row["ground_truth"] != row["predicted_zero_sum"]:
        return 0.0
    return None

df["F1_score"] = df.apply(compute_f1, axis=1)
df = df.dropna(subset=["F1_score"])

# ===== REGRESSION =====
X = sm.add_constant(df["text_length"])
y = df["F1_score"]
model = sm.OLS(y, X).fit()

# ===== SAVE SUMMARY TO TXT =====
summary_path = os.path.join(args.output_dir, "regression_summary.txt")
with open(summary_path, "w") as f:
    f.write(model.summary().as_text())
print(f"Regression summary saved to {summary_path}")

# ===== PLOT =====
plt.figure(figsize=(8, 5))
sns.regplot(x="text_length", y="F1_score", data=df, scatter_kws={"alpha": 0.5})
plt.title(f"F1 Score vs. Text Length ({args.length_unit})")
plt.xlabel(f"Text Length ({args.length_unit})")
plt.ylabel("F1 Score")
plt.tight_layout()
plot_path = os.path.join(args.output_dir, "f1_vs_length.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Plot saved to {plot_path}")
