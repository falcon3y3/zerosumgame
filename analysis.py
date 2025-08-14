#analyzes the results of the few shot prompting
import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

# --- Load predictions ---
predictions_path = "results/predictions.csv"
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"Predictions file not found at {predictions_path}")

df = pd.read_csv(predictions_path)

# --- Label parser ---
def extract_pred_label(text):
    try:
        obj = json.loads(text)
        label = str(obj.get("zero_sum", "UNKNOWN")).strip().upper()
        if label in ["Y", "YES"]:
            return "Y"
        elif label in ["N", "NO"]:
            return "N"
        else:
            return "UNKNOWN"
    except Exception:
        return "UNKNOWN"

df["predicted_label"] = df["llm_output"].apply(extract_pred_label)

# --- Handle UNKNOWNs ---
unknown_count = (df["predicted_label"] == "UNKNOWN").sum()
if unknown_count > 0:
    print(f"Warning: {unknown_count} predictions could not be parsed into Y/N.")

# --- Filter for valid labels ---
valid_df = df[df["predicted_label"].isin(["Y", "N"])].copy()

# --- Metrics ---
y_true = valid_df["ground_truth"]
y_pred = valid_df["predicted_label"]

accuracy = np.mean(y_true == y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

report = classification_report(y_true, y_pred, zero_division=0)
conf_mat = confusion_matrix(y_true, y_pred, labels=["Y", "N"])

# --- Save results ---
os.makedirs("results", exist_ok=True)
with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 (macro): {f1_macro:.4f}\n")
    f.write(f"F1 (weighted): {f1_weighted:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
    f.write(str(conf_mat))

print("Metrics saved to results/metrics.txt")
