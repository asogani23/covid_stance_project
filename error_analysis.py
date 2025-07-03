import pandas as pd

df = pd.read_csv("Q2_20230202_majority_predictions.csv")
errs = df[df["label_pred"] != df["label_majority"]]

# 10 random error cases
sample = errs[["tweet","label_majority","label_pred"]].sample(10, random_state=42)
print(sample.to_string(index=False))
