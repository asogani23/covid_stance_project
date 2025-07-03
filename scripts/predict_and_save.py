import pandas as pd
from transformers import pipeline

# Initialize as text2text (for seq2seq models like T5)
stance_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1, # Use -1 for CPU
    # force greedy decoding
    do_sample=False,
    temperature=0.0,
    top_k=1
)

# Define prompt template
PROMPT = (
    'What is the stance of the following tweet with respect to COVID-19 vaccine? '
    'Here is the tweet: "{tweet}" '
    'Please answer exactly one of: in-favor, against, neutral-or-unclear.'
)

def predict_one(tweet: str) -> str:
    # Replace double quotes to avoid issues with f-string formatting inside the prompt
    safe_tweet = tweet.replace('"', "'")
    inp = PROMPT.format(tweet=safe_tweet)

    # Generate the output
    out = stance_pipe(inp, max_new_tokens=5)[0]["generated_text"].strip() # Increased max_new_tokens for robustness

    # Convert to lowercase for consistent matching
    out_lower = out.lower()

    # Exact-match against our labels first
    for label in ["in-favor", "against", "neutral-or-unclear"]:
        if out_lower == label:
            return label

    
    # This is useful if the model adds extra words like "Stance: in-favor"
    for label in ["in-favor", "against", "neutral-or-unclear"]:
        if label in out_lower:
            return label

    # If none of the above, return neutral-or-unclear as a default or for unexpected outputs
    return "neutral-or-unclear"


# Load CSV file
try:
    df = pd.read_csv("Q2_20230202_majority.csv")
except FileNotFoundError:
    print("Error: 'Q2_20230202_majority.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Create a new column for predictions
df['label_pred'] = ""

print("Starting predictions for the entire dataset...")
# Iterate through each row and make a prediction
for index, row in df.iterrows():
    tweet = row['tweet']
    predicted_label = predict_one(tweet)
    df.at[index, 'label_pred'] = predicted_label

    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} tweets.")

print("All tweets processed.")

# Save the updated DataFrame to a new CSV file
output_filename = "Q2_20230202_majority_predictions.csv"
df.to_csv(output_filename, index=False)
print(f"Predictions saved to {output_filename}")
