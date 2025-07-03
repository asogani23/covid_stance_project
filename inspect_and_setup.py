import pandas as pd
from transformers import pipeline

#  Load
df = pd.read_csv("Q2_20230202_majority.csv")

#  Peek
print(df.head())
#    tweet                                      label_true  label_pred

#  Shape & info
print("Rows,Cols:", df.shape)
df.info()

#  Missing check & label balance
print("Missing per column:\n", df.isnull().sum())
print("Label counts:\n", df["label_true"].value_counts())


#  Initialize on CPU
stance_pipe = pipeline(
    task="text-generation",                
    model="google/flan-t5-large",          
    tokenizer="google/flan-t5-large",      
    device=-1                              
)

#  Warm-up a dummy run
out = stance_pipe("Hello world", max_new_tokens=2)
print("Pipeline test OK:", out)
