import pandas as pd
from transformers import pipeline

# 1) Load
df = pd.read_csv("Q2_20230202_majority.csv")

# 2) Peek
print(df.head())
#    tweet_id           created_at                        tweet  label_majority   month

# 3) Shape & info
print("Rows,Cols:", df.shape)
df.info()

# 4) Missing check & label balance
print("Missing per column:\n", df.isnull().sum())
print("Label distribution:\n", df["label_majority"].value_counts())

# 5) Prepare the empty prediction column
df["label_pred"] = pd.NA
print("\nAfter adding label_pred column:")
print(df.head())

# 6) Initialize the HuggingFace pipeline on CPU
stance_pipe = pipeline(
    task="text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1
)

# 7) Warm-up a dummy run
out = stance_pipe("Hello world", max_new_tokens=2)
print("Pipeline test OK:", out)
