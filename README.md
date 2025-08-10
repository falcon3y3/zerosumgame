# The Zero-Sum Game Illusion Online
This is a working repository for my MSc thesis on dialogue extinction in the 21st century, with a specific focus on zero-sum game illusions in subreddits and language learning models' potential ability to act as a destructive discourse detectors. 
To begin with, I have generated a ground-truth data set of examples and non-examples of zero-sum language derived from the subreddit r/changemyview. In order to compile this, I reviewed a publicly available dataset published from Convokit (Cornell NLP), known as the conversations-gone-awry-cmv-corpus. This dataset specifically includes CMV conversations pulled via Reddit's API (6,842 conversations containing 42,964 comments) that eventually derailed into antisocial behavior and moderator intervention, as well as exchanges that ended calmly (Chang and Danescu-Niculescu-Mizil, 2019). 

# Annotation Guide
Included in this Github is an annotation guide with basic instructions for human annotaters who examined the subset of selected zero-sum examples and non-examples for validation. 

# Dataset credit: 
The conversations-gone-awry-cmv-corpus dataset can be downloaded via the Convokit tutorial available on the Cornell NLP github (https://github.com/CornellNLP/ConvoKit/tree/master), or via the following link: https://zissou.infosci.cornell.edu/convokit/datasets/

Alternatively, you are welcome to utilize the code written here (groundtruth3.py) which will also download the corpus as csvs. Many thanks to the researchers who made this data publicly available and easily accessible. 

Chang, Jonathan P., and Cristian Danescu-Niculescu-Mizil. ‘Trouble on the Horizon: Forecasting the Derailment of Online Conversations as They Develop’. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), edited by Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan. Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/D19-1481.


Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. 2020. "ConvoKit: A Toolkit for the Analysis of Conversations". Proceedings of SIGDIAL.

