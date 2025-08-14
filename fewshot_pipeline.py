#few shot pipeline that includes balanced examples

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import OpenAI
import math
import json

# ===== CONFIG =====
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0
DATA_PATH = "data/groundtruth_cleaned_anon.xlsx"
RESULTS_PATH = "results/predictions.csv"
SEED = 42
K_FEW_SHOT = 8      # Number of few-shot examples
MAX_PROMPT_TOKENS = None  # Cap prompt length if needed

# ===== CREATE RESULTS FOLDER =====
os.makedirs("results", exist_ok=True)

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
    return max(1, math.ceil(len(s) / 4))  # crude heuristic

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
            ex_str = f'Comment: "{r["text"]}"\nOutput: {{"zero_sum": {int(r["zero_sum"])}, "justification": "{r["Annotation Justifications Combined"]}"}}\n'
            t = approx_tokens(ex_str)
            if used + t > max_tokens:
                break
            chosen.append(r)
            used += t
        few_shot_df = pd.DataFrame(chosen)

    return few_shot_df

FEW_SHOT_SET = sample_few_shot(train, k=K_FEW_SHOT, seed=SEED, max_tokens=MAX_PROMPT_TOKENS)

# ===== Prompting =====
def build_prompt(text, examples_df):
    examples_str = ""
    for _, row in examples_df.iterrows():
        examples_str += f'Comment: "{row["text"]}"\nOutput: {{"zero_sum": {int(row["zero_sum"])}, "justification": "{row["Annotation Justifications Combined"]}"}}\n\n'

    return f"""
You are an expert linguistic researcher. 
You are tasked with reviewing Reddit comments on controversial opinions 
to determine whether the text contains a specific linguistic feature 
called the illusion of a zero-sum game.

The definition of the illusion of a zero-sum game is as follows: 
a phenomenon in which people assign a strict gain/loss framework to a given conflict, 
such that any gains accomplished by one side must necessarily be accompanied 
by an equivalent loss on the part of the other. 

This language often appears in discourse on political or controversial topics
wherein people may identify strongly with one group over another.
Please follow the instructions below and think carefully before
generating an output.

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

# ===== CLASSIFICATION FUNCTION =====
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
            "ground_justification": row["Annotation Justifications Combined"],
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
y_true = valid_preds["ground_truth"]
y_pred = valid_preds["predicted_zero_sum"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

