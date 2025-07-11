import pandas as pd
import os

# Customize these paths:
input_dir = "/data/grte4343/thesis_groundtruth/zerosumgame/my_outputs/cmv_csvs"
output_dir = "/data/grte4343/thesis_groundtruth/zerosumgame/my_outputs/filtered_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load utterances CSV
utterances_path = os.path.join(input_dir, "utterances.csv")
utterances = pd.read_csv(utterances_path)

# Define vocab features (keywords) to filter for:
vocab_keywords = ["exampleword1", "exampleword2", "examplephrase"]

# Filter utterances containing any of the vocab keywords (case-insensitive)
pattern = "|".join(vocab_keywords)
filtered_vocab = utterances[utterances['text'].str.contains(pattern, case=False, na=False)]

filtered_vocab.to_csv(os.path.join(output_dir, "filtered_vocab_utterances.csv"), index=False)

print(f"Filtered utterances saved: {os.path.join(output_dir, 'filtered_vocab_utterances.csv')}")

# --- Example: Filtering "failed discourse" examples ---
# (Assuming you have a column 'discourse_status' or similar indicating success/failure)

if 'discourse_status' in utterances.columns:
    failed_discourse = utterances[utterances['discourse_status'] == 'failed']
    failed_discourse.to_csv(os.path.join(output_dir, "failed_discourse_utterances.csv"), index=False)
    print(f"Failed discourse utterances saved: {os.path.join(output_dir, 'failed_discourse_utterances.csv')}")
else:
    print("No 'discourse_status' column found for filtering failed discourse.")

