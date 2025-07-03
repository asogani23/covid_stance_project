from transformers import pipeline

#  Initialize as text2text (for seq2seq models like T5)
stance_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1,
    # NEW: force greedy decoding
    do_sample=False,
    temperature=0.0,
    top_k=1
)

#  Define prompt template
PROMPT = (
    'What is the stance of the following tweet with respect to COVID-19 vaccine? '
    'Here is the tweet: "{tweet}" '
    'Please answer exactly one of: in-favor, against, neutral-or-unclear.'
)

def predict_one(tweet: str) -> str:
    safe = tweet.replace('"', "'")
    inp = PROMPT.format(tweet=safe)
    out = stance_pipe(inp, max_new_tokens=3)[0]["generated_text"].strip()
    # exact-match against our labels
    for label in ["in-favor","against","neutral-or-unclear"]:
        if out == label:
            return label
    # fallback: pick the first label that *contains* our output
    for label in ["in-favor","against","neutral-or-unclear"]:
        if label in out:
            return label
    return "neutral-or-unclear"


#  Quick dry run on a couple of examples
examples = [
    "I had zero side effects, vaccines are great!",
    "These shots are poisoning people.",
    "Not sure what to think about the new booster."
]
for ex in examples:
    print(ex, "→", predict_one(ex))
