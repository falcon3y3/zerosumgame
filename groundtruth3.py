#in bash
#chmod +x setup_env.sh
#./setup_env.sh

import os
import pandas as pd
from convokit import Corpus, download

# Set custom cache directory
user = os.environ.get("USER") or os.environ.get("LOGNAME")
corpus_cache_dir = f"/scratch/{user}/corpus_cache"
os.makedirs(corpus_cache_dir, exist_ok=True)

# Path to where the CMV corpus will be saved
cmv_corpus_path = os.path.join(corpus_cache_dir, "conversations-gone-awry-cmv-corpus")

# Download corpus only if not already downloaded
if not os.path.exists(cmv_corpus_path):
    print("📥 Downloading CMV corpus to scratch cache...")
    download("conversations-gone-awry-cmv-corpus", data_dir=corpus_cache_dir)
else:
    print("✅ CMV corpus found in cache. Skipping download.")

# Load the corpus
corpus = Corpus(filename=cmv_corpus_path)

# Output directory
output_dir = "./my_outputs/cmv_csvs"
os.makedirs(output_dir, exist_ok=True)

# Export utterances
utterances = [utt.meta | {
    "id": utt.id,
    "speaker": utt.speaker.id if utt.speaker else None,
    "text": utt.text,
    "reply_to": utt.reply_to,
    "timestamp": utt.timestamp
} for utt in corpus.iter_utterances()]
pd.DataFrame(utterances).to_csv(os.path.join(output_dir, "utterances.csv"), index=False)

# Export users
users = [{"user_id": speaker.id, **speaker.meta} for speaker in corpus.iter_speakers()]
pd.DataFrame(users).to_csv(os.path.join(output_dir, "users.csv"), index=False)

# Export conversations
convos = [{"conv_id": convo.id, **convo.meta} for convo in corpus.iter_conversations()]
pd.DataFrame(convos).to_csv(os.path.join(output_dir, "conversations.csv"), index=False)

print(f"\n✅ All CSVs saved to: {os.path.abspath(output_dir)}")
