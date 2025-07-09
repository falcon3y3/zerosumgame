# Creating a data pull from Cornell NLP group's Convokit, which includes a data corpus on CMV (r/changemyview) 
# of "conversations gone awry" aka convos that violated the r/CMV subreddit rules as determined by mods
#this script is adapted to run in vs code, using a cluster and a virtual environment


#run ALL of these commands in terminal first
# Navigate to your project directory
#cd /data/grte4343/thesis_groundtruth/zerosumgame

# Create a virtual environment named 'venv'
#python3 -m venv venv

# Activate the virtual environment
#source venv/bin/activate

#install convokit
#pip install convokit
#install pandas 
#pip install pandas
#install unsloth
#pip install "unsloth[torch,auto]"

#then you can run the script below as normal python script 
# Creating a data pull from Cornell NLP group's Convokit, which includes a data corpus on CMV (r/changemyview)
# of "conversations gone awry" which are convos that violated subreddit rules as determined by mods


import unsloth
import site
import os
from convokit import Corpus, download
import pandas as pd

OUTPUT_DIR = "/data/grte4343/thesis_groundtruth/zerosumgame"

SITE_PACKAGES_PATH = site.getsitepackages()[0]
CONVOKIT_PATH = os.path.dirname(__import__('convokit').__file__)

corpus = Corpus(download("conversations-gone-awry-cmv-corpus"))
corpus.print_summary_stats()
print("Site-packages path:", SITE_PACKAGES_PATH)
print("Convokit path:", CONVOKIT_PATH)

# --- Save corpus as JSON directory ---

corpus.dump(os.path.join(OUTPUT_DIR, "cmv_corpus_json"))
utterances_df.to_csv(os.path.join(OUTPUT_DIR, "cmv_utterances.csv"), index=False)
conversations_df.to_csv(os.path.join(OUTPUT_DIR, "cmv_conversations.csv"), index=False)
unsloth_df.to_csv(os.path.join(OUTPUT_DIR, "cmv_unsloth_train.csv"), index=False)

# --- Save utterances as CSV ---
utterances_df = pd.DataFrame([utt.to_dict() for utt in corpus.iter_utterances()])
utterances_df.to_csv("cmv_utterances.csv", index=False)

# --- Save conversations as CSV ---
conversations_df = pd.DataFrame([conv.to_dict() for conv in corpus.iter_conversations()])
conversations_df.to_csv("cmv_conversations.csv", index=False)

# --- Prepare Unsloth training data (prompt/response pairs) ---
pairs = []
utterance_dict = {utt.id: utt for utt in corpus.iter_utterances()}
for utt in corpus.iter_utterances():
    if utt.reply_to and utt.reply_to in utterance_dict:
        prompt = utterance_dict[utt.reply_to].text
        response = utt.text
        pairs.append({"prompt": prompt, "response": response})

unsloth_df = pd.DataFrame(pairs)
unsloth_df.to_csv("cmv_unsloth_train.csv", index=False)

print("Script finished and files should be saved.")