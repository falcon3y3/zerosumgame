# zero-shot pipeline, add --random when running if looking for multiple samples
# has timestamps
#uses claude-3-5-haiku-20241022
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from anthropic import Anthropic

# ===== ARGUMENT PARSER =====
parser = argparse.ArgumentParser()
parser.add_argument("--random", action="store_true", help="Use random splits each run (no fixed seed)")
args = parser.parse_args()

# ===== CONFIG =====
MODEL_NAME = "claude-3-5-haiku-20241022"  # update if you want a faster/cheaper variant
TEMPERATURE = 0
DATA_PATH = "data/groundtruth_cleaned_anon.xlsx"
SEED = 42

# ===== CREATE RESULTS FOLDER =====
os.makedirs("results", exist_ok=True)

# ===== TIMESTAMP =====
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ===== API KEY SETUP =====
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set!")
client = Anthropic(api_key=api_key)

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

# ===== PROMPT BUILDING =====
def build_prompt_zero_shot(text):
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

Now classify the following comment:

Comment: "{text}"
Output:
""".strip()

# ===== CLASSIFICATION =====
def classify_zero_shot(text):
    prompt = build_prompt_zero_shot(text)
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        temperature=TEMPERATURE,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text

def run_classification(df_subset):
    results = []
    for _, row in df_subset.iterrows():
        prediction_text = classify_zero_shot(row["text"])
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
predictions_csv_path = f"results/claude_zero_shot_predictions_{run_timestamp}.csv"
predictions_df.to_csv(predictions_csv_path, index=False)
print(f"Predictions saved to {predictions_csv_path}")

# ===== METRICS =====
valid_preds = predictions_df.dropna(subset=["predicted_zero_sum"])
y_true = valid_preds["ground_truth"]
y_pred = valid_preds["predicted_zero_sum"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

metrics_df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1,
    "timestamp": run_timestamp
}])
metrics_csv_path = f"results/claude_zero_shot_metrics_{run_timestamp}.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
print("\n--- Evaluation Metrics ---")
print(metrics_df)
print(f"Metrics saved to {metrics_csv_path}")

# ===== NORMALIZED CONFUSION MATRIX =====
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Not Zero-Sum", "Zero-Sum"],
            yticklabels=["Not Zero-Sum", "Zero-Sum"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Normalized Confusion Matrix Zero-Shot: Claude")
conf_matrix_path = f"results/zero_shot_confusion_matrix_{run_timestamp}.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved to {conf_matrix_path}")
