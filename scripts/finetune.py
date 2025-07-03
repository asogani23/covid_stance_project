

# Install dependencies
!pip install -q --upgrade transformers[torch] datasets pandas scikit-learn


#  Imports

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
    Seq2SeqTrainer
)


#  Load CSV & sample small subset
CSV_PATH = "Q2_20230202_majority.csv"
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"tweet": "input_text", "label_majority": "target_text"})

# full split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.sample(n=500, random_state=42).reset_index(drop=True)
val_df   = val_df.sample(n=100, random_state=42).reset_index(drop=True)

print(f"→ Using {len(train_df)} train / {len(val_df)} validation samples (prototype slice)")

train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)


# Tokenizer + Preprocessing (with label masking)
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    prompts = [
        f"What is the stance of the following tweet with respect to COVID-19 vaccine? "
        f"Here is the tweet: \"{t}\" Please use exactly one word from: "
        f"\"in-favor\", \"against\", \"neutral-or-unclear\"."
        for t in examples["input_text"]
    ]
    inputs = tokenizer(prompts, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=16,
            truncation=True,
            padding="max_length"
        )
    # mask pad tokens for loss
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels["input_ids"]
    ]
    inputs["labels"] = labels["input_ids"]
    return inputs

print(" Tokenizing…")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)
tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)
print(" Tokenization complete.")


#  Load model + config adjustments
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()
model.config.use_cache = False
torch.cuda.empty_cache()


# Training arguments (1 epoch, no eval during train)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="no",                 # skip validation pass to save time
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,                  # only one epoch for quick test
    predict_with_generate=True,
    fp16=True,
    optim="adafactor",
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


#  Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,      # still needed for final `.predict()`
    tokenizer=tokenizer,
    data_collator=data_collator,
)


#  Quick Fine-tune
print(" Starting prototype fine-tuning (1 epoch, no mid-eval)…")
trainer.train()
print(" Prototype fine-tuning complete.")


# Predict & Save on small val

print("Generating predictions on small validation set…")
preds = trainer.predict(tokenized_val)
decoded = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)

val_df["label_pred"] = decoded
val_df.to_csv("prototype_predictions.csv", index=False)
print(" Prototype predictions saved.")


# Quick Metrics
print("\n--- Prototype Validation Metrics ---")
print(classification_report(val_df["target_text"], val_df["label_pred"]))
print(confusion_matrix(
    val_df["target_text"],
    val_df["label_pred"],
    labels=["in-favor", "against", "neutral-or-unclear"]
))
