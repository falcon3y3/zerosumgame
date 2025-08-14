# zero_shot_pipeline_with_metrics.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from openai import OpenAI
import json

# ===== CONFIG =====
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0
DATA_PATH = "data/groundtruth_cleaned_anon.xlsx"
RESULTS_PATH = "results/predictions_zero_shot.csv"
SEED = 42

# ===== CREATE RESULTS FOLDER =====
os.makedirs("results", exist_ok=True)

# ===== API KEY SETUP =====
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set! Please set it in your terminal or .env file.")
client = OpenAI(api_key=api_key)

# ===== LOAD DATA =====
df = pd.read_excel(DATA_PATH)
df.rename(columns={
    "Zero-Sum Y/N": "zero_sum",
    "Annotation Justifications Combined": "justification"
}, inplace=True)

# ===== SPLIT DATA 70/15/15 =====
train_val, test = train_test_split(df, test_size=0.15, random_state=SEED, shuffle=True)
train, validate = train_test_split(train_val, test_size=0.1765, random_state=SEED, shuffle=True)
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

# ===== SAVE RESULTS =====
predictions_df.to_csv(RESULTS_PATH, index=False)
print(f"Predictions saved to {RESULTS_PATH}")

# ===== COMPUTE METRICS =====
y_true = predictions_df["ground_truth"].astype(int)
y_pred = predictions_df["llm_zero_sum"].fillna(-1).astype(int)  # fill None with -1

# Filter out invalid predictions (-1)
valid_idx = y_pred.isin([0,1])
y_true = y_true[valid_idx]
y_pred = y_pred[valid_idx]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"\n=== METRICS ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

#PLUS A CHI SQUARED TEST
from scipy.stats import chi2_contingency

# ===== CHI-SQUARED TEST =====
# Build contingency table
contingency_table = pd.crosstab(y_true, y_pred)
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n=== STATISTICAL SIGNIFICANCE ===")
print("Chi-squared statistic:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)
