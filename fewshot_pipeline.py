# few shot pipeline
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import OpenAI
import math
import json
from scipy.stats import chi2_contingency
from datetime import datetime

# ===== CONFIG =====
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0
DATA_PATH = "data/groundtruth_cleaned_anon.xlsx"
SEED = 42
K_FEW_SHOT = 8
MAX_PROMPT_TOKENS = None

# ===== CREATE RESULTS FOLDER =====
os.makedirs("results", exist_ok=True)

# ===== TIMESTAMP =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = f"results/predictions_few_shot_{timestamp}.csv"
METRICS_PATH = f"results/metrics_few_shot_{timestamp}.csv"
CM_PATH = f"results/confusion_matrix_few_shot_{timestamp}.png"

# ===== API KEY SETUP =====
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")
client = OpenAI(api_key=api_key)

# ===== LOAD DATA =====
df = pd.read_excel(DATA_PATH)

# ===== SPLIT DATA 70/15/15 =====
train_val, test = train_test_split(df, test_size=0.15, random_state=SEED, shuffle=True)
train, validate = train_test_split(train_val, test_size=0.1765, random_state=SEED, shuffle=True)

print(f"Train size: {len(train)}, Validate size: {len(validate)}, Test size: {len(test)}")

# ===== TOKEN APPROXIMATION =====
def approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))

# ===== FEW-SHOT SAMPLING =====
def sample_few_shot(train_df, k=K_FEW_SHOT, seed=42, max_tokens=None):
    pos = train_df[train_df["zero_sum"] == 1]
    neg = train_df[train_df["zero_sum"] == 0]

    k_pos = min(len(pos), k // 2)
    k_neg = min(len(neg), k - k_pos)

    few_shot_df = pd.concat([
        pos.sample(k_pos, random_state=seed),
        neg.sample(k_neg, random_state=seed)
    ]).sample(frac=1.0, random_state=seed)

    if max_tokens:
        chosen = []
        used = 0
        for _, r in few_shot_df.iterrows():
            ex_str = f'Comment: "{r["text"]}"\nOutput: {{"zero_sum": {int(r["zero_sum"])}, "justification": "{r["annotation_justification_combo"]}"}}\n'
            t = approx_tokens(ex_str)
            if used + t > max_tokens:
                break
            chosen.append(r)
            used += t
        few_shot_df = pd.DataFrame(chosen)

    return few_shot_df

FEW_SHOT_SET = sample_few_shot(train, k=K_FEW_SHOT, seed=SEED, max_tokens=MAX_PROMPT_TOKENS)

# ===== PROMPT BUILDER =====
def build_prompt(text, examples_df):
    examples_str = ""
    for _, row in examples_df.iterrows():
        examples_str += f'Comment: "{row["text"]}"\nOutput: {{"zero_sum": {int(row["zero_sum"])}, "justification": "{row["annotation_justification_combo"]}"}}\n\n'

    return f"""
You are an expert linguistic researcher.
You are tasked with reviewing Reddit comments on controversial opinions
to determine whether the text contains the illusion of a zero-sum game.

Definition:
A phenomenon in which people assign a strict gain/loss framework to a conflict,
such that any gains accomplished by one side must be accompanied by an equivalent loss on the other.
Often seen in political or controversial discourse.

Instructions:
- Respond ONLY with valid JSON.
- JSON must have exactly two keys:
  - "zero_sum": integer 1 if zero-sum framing is present, otherwise integer 0
  - "justification": concise 1–2 sentence reason for the classification

Here are some examples:
{examples_str}
Now classify the following:

Comment: "{text}"
Output:
""".strip()

# ===== CLASSIFY =====
def classify_with_justification(text):
    prompt = build_prompt(text, FEW_SHOT_SET)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content

# ===== RUN CLASSIFICATION =====
def run_classification(df_subset):
    results = []
    for _, row in df_subset.iterrows():
        prediction_text = classify_with_justification(row["text"])
        try:
            parsed = json.loads(prediction_text)
            pred_label = int(parsed.get("zero_sum", 0))
            pred_just = parsed.get("justification", "")
        except json.JSONDecodeError:
            pred_label = None
            pred_just = prediction_text

        results.append({
            "ID": row["ID"],
            "text": row["text"],
            "ground_truth": int(row["zero_sum"]),
            "ground_justification": row["annotation_justification_combo"],
            "predicted_zero_sum": pred_label,
            "predicted_justification": pred_just,
            "raw_llm_output": prediction_text
        })
    return pd.DataFrame(results)

# ===== EXECUTE ON TEST SET =====
predictions_df = run_classification(test)
predictions_df.to_csv(RESULTS_PATH, index=False)
print(f"Predictions saved to {RESULTS_PATH}")

# ===== METRICS =====
valid_preds = predictions_df.dropna(subset=["predicted_zero_sum"])
y_true = valid_preds["ground_truth"].astype(int)
y_pred = valid_preds["predicted_zero_sum"].astype(int)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype(float) / cm.sum()

# ===== CHI-SQUARE =====
contingency_table = pd.crosstab(y_true, y_pred)
chi2, p, dof, expected = chi2_contingency(contingency_table)

# ===== SAVE METRICS CSV =====
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
plt.title("Confusion Matrix - Few Shot")
plt.savefig(CM_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to {CM_PATH}")

# ===== PRINT TO CONSOLE =====
print("\n--- Evaluation Metrics ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:\n", cm)

print("\n=== Statistical Significance ===")
print("Chi-squared statistic:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)
