
#this script uses few-shot prompting with a train/validation/test set to determine LLM ability to 
#identify the illusion of a zero-sum game in Reddit comments

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from openai import OpenAI

# ===== CONFIG =====
MODEL_NAME = "gpt-4.1"  # using GPT-4
TEMPERATURE = 0
DATA_PATH = "data/groundtruth_cleaned_anon.xlsx"
RESULTS_PATH = "results/predictions.csv"
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
# Rename columns for convenience
df.rename(columns={
    "zero_sum": "zero_sum",
    "annotation_justifiication_combo": "justification"
}, inplace=True)

# ===== SPLIT DATA 70/15/15 =====
train_val, test = train_test_split(df, test_size=0.15, random_state=SEED, shuffle=True)
train, validate = train_test_split(train_val, test_size=0.1765, random_state=SEED, shuffle=True)  
# 0.1765 ≈ 15% of total

print(f"Train size: {len(train)}, Validate size: {len(validate)}, Test size: {len(test)}")

# ===== FUNCTION TO CLASSIFY WITH GPT-4 =====
def classify_with_justification(text):
    prompt = f"""
You are an expert linguistic researcher. 
You are tasked with reviewing Reddit comments on controversial opinions 
to determine whether the text contains 
a specific linguistic feature called the illusion of a zero-sum game.
The definition of the illusion of a zero-sum game is as follows: 
a phenomenon in which people assign a strict gain/loss framework to a given conflict, 
such that any gains accomplished by one side must necessarily be accompanied 
by an equivalent loss on the part of the other. 
This language often appears in discourse on political or controversial topics
wherein people may identify strongly with one group over another.
Please follow the instructions below and think carefully before
generating an output.

Comment:
\"\"\"{text}\"\"\"

Rules for output:
1. Respond ONLY with valid JSON.
2. JSON must have exactly two keys:
   - "zero_sum": "Y" if zero-sum framing is present, otherwise "N"
   - "justification": a concise 1–2 sentence reason why this comment is either zero-sum or not
3. Do not include any text outside of the JSON object.
"""
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
        # Optionally, parse response into structured zero_sum / justification
        # Here we just store raw LLM output
        results.append({
            "ID": row["ID"],
            "text": row["text"],
            "ground_truth": row["zero_sum"],
            "ground_justification": row["justification"],
            "llm_output": prediction_text
        })
    return pd.DataFrame(results)

# Example: classify the test set (you can also classify train/validate if needed)
predictions_df = run_classification(test)

# ===== SAVE RESULTS =====
predictions_df.to_csv(RESULTS_PATH, index=False)
print(f"Predictions saved to {RESULTS_PATH}")
