# download_cmv_csv_virtualenv.py
#make sure to run this script inside a virtual environment and ensure you're in the proper directory using the commands:
#python3 -m venv venv
#source venv/bin/activate
#pip install convokit pandas

import os
import subprocess
import sys
import pandas as pd
from convokit import Corpus, download

# Step 1: Confirm script is running inside a virtual environment
if sys.prefix == sys.base_prefix:
    print("⚠️ Not running inside a virtual environment. Please activate one first.")
    sys.exit(1)

# Step 2: Ensure output directory exists
output_dir = "./data/grte4343/thesis_groundtruth/zerosumgame"
os.makedirs(output_dir, exist_ok=True)

# Step 3: Download and load the CMV corpus
print("📥 Downloading CMV corpus...")
corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
print("✅ Download complete.")

# Step 4: Export utterances
print("📤 Exporting utterances...")
utterances = [utt.meta | {
    "id": utt.id,
    "speaker": utt.speaker.id if utt.speaker else None,
    "text": utt.text,
    "reply_to": utt.reply_to,
    "timestamp": utt.timestamp
} for utt in corpus.iter_utterances()]
pd.DataFrame(utterances).to_csv(os.path.join(output_dir, "utterances.csv"), index=False)

# Step 5: Export users
print("📤 Exporting users...")
users = [{"user_id": user.id, **user.meta} for user in corpus.iter_users()]
pd.DataFrame(users).to_csv(os.path.join(output_dir, "users.csv"), index=False)

# Step 6: Export conversations
print("📤 Exporting conversations...")
convos = [{"conv_id": convo.id, **convo.meta} for convo in corpus.iter_conversations()]
pd.DataFrame(convos).to_csv(os.path.join(output_dir, "conversations.csv"), index=False)

print(f"\n✅ All CSVs saved to: {os.path.abspath(output_dir)}")
