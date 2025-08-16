# few-shot pipeline with timestamped outputs + cumulative metrics logging
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import OpenAI
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ===== ARGUMENT PARSER =====
parser = argparse.ArgumentParser()
parser.add_argument("--random", action="store_true", help="Use random splits each run (no fixed seed)")
args = parser.parse_args()

# ===== CONFIG =====
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0
DATA_PATH = "data/groundtruth_cleaned_anon.xlsx"
SEED = 42
K_FEW_SHOT = 143
MAX_PROMPT_TOKENS = None
MODEL_TYPE = "few-shot"

# ===== CREATE RESULTS FOLDERS =====
os.makedirs("results", exist_ok=True)
os.makedirs("results/history", exist_ok=True)

# ===== TIMESTAMP =====
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ===== API KEY SETUP =====
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")
client = OpenAI(api_key=api_key)

# ===== LOAD DATA =====
df = pd.read_excel(DATA_PATH)

# ===== SPLIT DATA =====
if args.random:
    print("🔀 Using random splits each run.")
    train_val, test = train_test_split(df, test_size=0.15, shuffle=True)
    train, validate = train_test_split(train_val, test_size=0.1765, shuffle=True)
else:
    print(f"📌 Using fixed seed: {SEED}")
    train_val, test = train_test_split(df, test_size=0.15, random_state=SEED, shuffle=True)
    train, validate = train_test_split(train_val, test_size=0.1765, random_state=SEED, shuffle=True)

print(f"Train size: {len(train)}, Validate size: {len(validate)}, Test size: {len(test)}")

# ===== TOKEN APPROXIMATION =====
def approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))

# ===== FEW-SHOT SAMPLING =====
def sample_few_shot(train_df, k=K_FEW_SHOT, max_tokens=None):
    pos = train_df[train_df["zero_sum"] == 1]
    neg = train_df[train_df["zero_sum"] == 0]

    k_pos = min(len(pos), k // 2)
    k_neg = min(len(neg), k - k_pos)

    if args.random:
        few_shot_df = pd.concat([
            pos.sample(k_pos),
            neg.sample(k_neg)
        ]).sample(frac=1.0)
    else:
        few_shot_df = pd.concat([
            pos.sample(k_pos, random_state=SEED),
            neg.sample(k_neg, random_state=SEED)
        ]).sample(frac=1.0, random_state=SEED)

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

FEW_SHOT_SET = sample_few_shot(train, k=K_FEW_SHOT, max_tokens=MAX_PROMPT_TOKENS)

# ===== Prompt Building =====
def build_prompt(text, examples_df):
    examples_str = ""
    for _, row in examples_df.iterrows():
        examples_str += f'Comment: "{row["text"]}"\nOutput: {{"zero_sum": {int(row["zero_sum"])}, "justification": "{row["annotation_justification_combo"]}"}}\n\n'

    return f"""
You are an expert linguistic researcher. 
You are tasked with reviewing Reddit comments on controversial opinions to determine whether the text contains 
a specific linguistic feature called the illusion of a zero-sum game. 
The definition of the illusion of a zero-sum game is as follows: 
a phenomenon in which people assign a strict gain/loss framework to a given conflict, 
such that any gains accomplished by one side must necessarily be accompanied by an 
equivalent loss on the part of the other. 

This language often appears in discourse on political or controversial topics 
wherein people may identify strongly with one group over another. 
Please follow the instructions below and think carefully before generating an output.

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

# ===== CLASSIFICATION =====
def classify_with_justification(text):
    prompt = build_prompt(text, FEW_SHOT_SET)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content

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
            "raw_llm_output": prediction_text,
            "timestamp": run_timestamp
        })
    return pd.DataFrame(results)

# ===== EXECUTE ON TEST SET =====
predictions_df = run_classification(test)
predictions_path = f"results/fewshot_predictions_{run_timestamp}.csv"
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")

# ===== METRICS =====
valid_preds = predictions_df.dropna(subset=["predicted_zero_sum"])
y_true = valid_preds["ground_truth"]
y_pred = valid_preds["predicted_zero_sum"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

metrics_df = pd.DataFrame([{
    "Model_Type": MODEL_TYPE,
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1,
    "timestamp": run_timestamp
}])
metrics_path = f"results/fewshot_metrics_{run_timestamp}.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics saved to {metrics_path}")

# ===== APPEND TO GLOBAL HISTORY =====
all_metrics_path = "results/history/all_metrics.csv"
if os.path.exists(all_metrics_path):
    existing_df = pd.read_csv(all_metrics_path)
    combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
else:
    combined_df = metrics_df
combined_df.to_csv(all_metrics_path, index=False)
print(f"Updated metrics history at {all_metrics_path}")

# ===== NORMALIZED CONFUSION MATRIX =====
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Not Zero-Sum", "Zero-Sum"],
            yticklabels=["Not Zero-Sum", "Zero-Sum"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Normalized Confusion Matrix (Few-Shot)\n{run_timestamp}")
conf_matrix_path = f"results/fewshot_confusion_matrix_{run_timestamp}.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved to {conf_matrix_path}")
