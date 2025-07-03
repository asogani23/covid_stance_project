!pip install -q transformers[torch] datasets pandas scikit-learn

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

#  Load and prepare the data
try:
    df = pd.read_csv("Q2_20230202_majority.csv")
except FileNotFoundError:
    print("Error: 'Q2_20230202_majority.csv' not found. Please upload the file to your Colab environment.")
    exit()

# Rename columns for clarity
df = df.rename(columns={"tweet": "input_text", "label_majority": "target_text"})

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

#  Tokenize the data
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    """Prepares the data for the model by tokenizing the input and target texts."""
    inputs = [f"What is the stance of the following tweet with respect to COVID-19 vaccine? Here is the tweet: \"{text}\" Please use exactly one word from the following 3 categories to label it: \"in-favor\", \"against\", \"neutral-or-unclear\"." for text in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=32, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

#  Fine-tune the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2,   
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Keep mixed-precision on
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# Evaluate the fine-tuned model
print("Evaluating the fine-tuned model...")
predictions = trainer.predict(tokenized_val_dataset)
predicted_labels = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

# Add predictions to the validation DataFrame
val_df['label_pred_fine_tuned'] = predicted_labels

# Save the predictions to a new CSV file
output_filename = "Q2_20230202_majority_predictions_fine_tuned.csv"
val_df.to_csv(output_filename, index=False)
print(f"Fine-tuned predictions saved to {output_filename}")

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report (Fine-Tuned):")
print(classification_report(val_df['target_text'], val_df['label_pred_fine_tuned']))

print("Confusion Matrix (Fine-Tuned):")
print(confusion_matrix(val_df['target_text'], val_df['label_pred_fine_tuned']))
