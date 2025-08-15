<<<<<<< HEAD
# zero_shot_pipeline_with_metrics_updated.py
=======
>>>>>>> 8aa06bea61e312342a17ea041ae5eba26cfb8888
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import OpenAI
import json
from scipy.stats import chi2_contingency
from datetime import datetime

# ===== CONFIG =====
MODEL_NAME = "gpt-4.1" #model used
TEMPERATURE = 0
DATA_PATH = "data/groundtruth_cleaned_anon.xlsx"
<<<<<<< HEAD
RESULTS_PATH = "results/predictions_zero_shot3.csv"
USE_SEED = False  # set True for reproducibility
=======
USE_SEED = False  #set True for reproducibility
>>>>>>> 8aa06bea61e312342a17ea041ae5eba26cfb8888
SEED = 42

# ===== CREATE RESULTS FOLDER =====
os.makedirs("results", exist_ok=True)

# ===== TIMESTAMP FOR FILENAMES =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = f"results/predictions_zero_shot_{timestamp}.csv"
METRICS_PATH = f"results/metrics_zero_shot_{timestamp}.csv"
CM_PATH = f"results/confusion_matrix_zero_shot_{timestamp}.png"

# ===== API KEY SETUP =====
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")
client = OpenAI(api_key=api_key)

# ===== LOAD DATA =====
df = pd.read_excel(DATA_PATH)
df.rename(columns={
    "Zero-Sum Y/N": "zero_sum",
    "annotation_justification_combo": "justification"
}, inplace=True)

# Ensure binary ground truth
df["zero_sum"] = df["zero_sum"].astype(int)

# ===== SPLIT DATA 70/15/15 =====
random_state = SEED if USE_SEED else None
train_val, test = train_test_split(df, test_size=0.15, random_state=random_state, shuffle=True)
train, validate = train_test_split(train_val, test_size=0.1765, random_state=random_state, shuffle=True)
print(f"Train size: {len(train)}, Validate size: {len(validate)}, Test size: {len(test)}")

# ===== ZERO-SHOT PROMPT BUILDER =====
def build_prompt_zero_shot(text):
    return f"""
You are an expert linguistic researcher.
You are tasked with reviewing Reddit comments to determine whether they contain
the illusion of a zero-sum game.

Definition:
A phenomenon in which people assign a strict gain/loss framework to a conflict,
such that any gains accomplished by one side must be accompanied by an equivalent
loss on the other. This is often seen in political or controversial discourse.

Instructions:
- Respond ONLY with valid JSON.
- JSON must have exactly two keys:
  - "zero_sum": integer 1 if zero-sum framing is present, otherwise integer 0
  - "justification": concise 1–2 sentence reason for the classification

Comment: "{text}"
Output:
""".strip()

# ===== CLASSIFY FUNCTION =====
def classify_with_justification(text):
    prompt = build_prompt_zero_shot(text)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    msg_content = response.choices[0].message.content
    try:
        return json.loads(msg_content)
    except json.JSONDecodeError:
        return {"zero_sum": None, "justification": msg_content}

# ===== RUN CLASSIFICATION =====
def run_classification(df_subset):
    results = []
    for _, row in df_subset.iterrows():
        pred = classify_with_justification(row["text"])
        results.append({
            "ID": row["ID"],
            "text": row["text"],
            "ground_truth": row["zero_sum"],
            "ground_justification": row["justification"],
            "llm_zero_sum": pred.get("zero_sum"),
            "llm_justification": pred.get("justification")
        })
    return pd.DataFrame(results)

# ===== CLASSIFY TEST SET =====
predictions_df = run_classification(test)
predictions_df.to_csv(RESULTS_PATH, index=False)
print(f"Predictions saved to {RESULTS_PATH}")

# ===== FILTER VALID PREDICTIONS =====
y_true = predictions_df["ground_truth"].astype(int)
y_pred = predictions_df["llm_zero_sum"].fillna(-1).astype(int)
<<<<<<< HEAD
valid_idx = y_pred.isin([0,1])
=======

# Filter invalid predictions (-1)
valid_idx = y_pred.isin([0, 1])
>>>>>>> 8aa06bea61e312342a17ea041ae5eba26cfb8888
y_true = y_true[valid_idx]
y_pred = y_pred[valid_idx]

# ===== METRICS =====
acc = accuracy_score(y_true, y_pred)
<<<<<<< HEAD
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("\n=== EVALUATION METRICS ===")
=======
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# ===== CHI-SQUARED TEST =====
contingency_table = pd.crosstab(y_true, y_pred)
chi2, p, dof, expected = chi2_contingency(contingency_table)

# ===== NORMALIZED CONFUSION MATRIX =====
cm_normalized = cm.astype("float") / cm.sum()

# ===== SAVE METRICS & CONFUSION MATRIX VALUES TO CSV =====
metrics_df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1,
    "Chi-squared Statistic": chi2,
    "P-value": p,
    "Degrees of Freedom": dof,
    "ConfusionMatrix_TN": cm[0, 0],
    "ConfusionMatrix_FP": cm[0, 1],
    "ConfusionMatrix_FN": cm[1, 0],
    "ConfusionMatrix_TP": cm[1, 1],
    "ConfusionMatrix_TN_Percent": cm_normalized[0, 0],
    "ConfusionMatrix_FP_Percent": cm_normalized[0, 1],
    "ConfusionMatrix_FN_Percent": cm_normalized[1, 0],
    "ConfusionMatrix_TP_Percent": cm_normalized[1, 1]
}])
metrics_df.to_csv(METRICS_PATH, index=False)
print(f"Metrics saved to {METRICS_PATH}")

# ===== SAVE CONFUSION MATRIX PLOT =====
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Zero Shot")
plt.savefig(CM_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to {CM_PATH}")

# ===== PRINT METRICS TO TERMINAL =====
print("\n=== METRICS ===")
>>>>>>> 8aa06bea61e312342a17ea041ae5eba26cfb8888
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
<<<<<<< HEAD
print("\nConfusion Matrix:")
print(cm)
=======
print("Confusion Matrix:\n", cm)

print("\n=== STATISTICAL SIGNIFICANCE ===")
print("Chi-squared statistic:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)
>>>>>>> 8aa06bea61e312342a17ea041ae5eba26cfb8888
