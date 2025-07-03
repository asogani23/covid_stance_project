# Imports
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Load & split the full dataset
CSV_PATH = "Q2_20230202_majority.csv"
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"tweet": "input_text", "label_majority": "target_text"})
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"→ {len(train_df)} train / {len(val_df)} validation samples")

train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)

# Tokenizer + preprocessing with label masking
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    prompts = [
        "What is the stance of the following tweet with respect to COVID-19 vaccine? "
        f"Here is the tweet: “{t}” Please use exactly one word from: "
        "\"in-favor\", \"against\", \"neutral-or-unclear\"."
        for t in examples["input_text"]
    ]
    inputs = tokenizer(prompts, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=16, truncation=True, padding="max_length"
        )
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels["input_ids"]
    ]
    inputs["labels"] = labels["input_ids"]
    return inputs

print(" Tokenizing train…")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)
print(" Tokenizing val…")
tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)
print(" Tokenization complete.")

# Load model + memory tweaks
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()
model.config.use_cache = False
torch.cuda.empty_cache()

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",    
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,
    optim="adafactor",
    eval_accumulation_steps=1,
    generation_max_length=2,
    generation_num_beams=4,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize Trainer & train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(" Starting full fine-tuning (3 epochs)…")
trainer.train()
print(" Fine-tuning complete.")

# Predict, clean, and save
print("🔍 Generating predictions on validation set…")
preds = trainer.predict(tokenized_val)
raw = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)

VALID = {"in-favor","against","neutral-or-unclear"}
def clean_label(r: str) -> str:
    txt = r.strip().lower()
    if txt in VALID: return txt
    if txt.startswith("in"): return "in-favor"
    if txt.startswith("a"):  return "against"
    return "neutral-or-unclear"

cleaned = [clean_label(r) for r in raw]
val_df = val_df.reset_index(drop=True)
val_df["label_pred"] = cleaned
OUT_CSV = "Q2_20230202_majority_predictions_fine_tuned.csv"
val_df.to_csv(OUT_CSV, index=False)
print(f" Saved predictions to {OUT_CSV}")

# Report final metrics
print("\n--- Final Validation Metrics ---")
print(classification_report(val_df["target_text"], val_df["label_pred"]))
print(confusion_matrix(
    val_df["target_text"],
    val_df["label_pred"],
    labels=["in-favor","against","neutral-or-unclear"]
))
