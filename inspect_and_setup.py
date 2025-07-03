from transformers import pipeline

#  Initialize as text2text (for seq2seq models like T5)
stance_pipe = pipeline(
    task="text2text-generation",      # ← changed here
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1
)

#  Define your prompt template
PROMPT = (
    'What is the stance of the following tweet with respect to COVID-19 vaccine? '
    'Here is the tweet: "{tweet}" '
    'Please answer exactly one of: in-favor, against, neutral-or-unclear.'
)

def predict_one(tweet: str) -> str:
    # sanitize quotes to avoid breaking the prompt
    safe = tweet.replace('"', "'")
    inp = PROMPT.format(tweet=safe)
    out = stance_pipe(inp, max_new_tokens=4)[0]["generated_text"]
    # simple extraction of one of our three labels
    for label in ("in-favor", "against", "neutral-or-unclear"):
        if label in out:
            return label
    # fallback
    return "neutral-or-unclear"

#  Quick dry run on a couple of examples
examples = [
    "I had zero side effects, vaccines are great!",
    "These shots are poisoning people.",
    "Not sure what to think about the new booster."
]
for ex in examples:
    print(ex, "→", predict_one(ex))
