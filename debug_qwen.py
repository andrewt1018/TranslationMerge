import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CHANGE THESE
MODEL_DIR = "models/qwen_en_zh_ft/checkpoint-60000"   # or en_zh checkpoint
LANG_PAIR = "en-zh"                                   # "en-ja" or "en-zh"

# Simple test sentence
src_sentence = "The forum produced business contracts amounting to more than $24 million between Asian and African private companies."
src_sentence = "The Global Programme of Action Coordination Office, with the financial support of Belgium,\xa0is currently assisting Egypt, Nigeria, United Republic of Tanzania, Sri Lanka and Yemen to develop pilot national programmes of action for the protection of the marine environment from land-based activities."

# Language tags (must match training!)
if LANG_PAIR == "en-ja":
    prompt = f"<en> {src_sentence} <ja>"
elif LANG_PAIR == "en-zh":
    prompt = f"<en> {src_sentence} <zh>"
else:
    raise ValueError("Unsupported lang pair")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
model.eval()

# Tokenize prompt
enc = tokenizer(prompt, return_tensors="pt").to(device)
input_len = enc["input_ids"].shape[1]

# Generate
with torch.no_grad():
    out = model.generate(
        **enc,
        max_new_tokens=64,
        num_beams=5,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    
def clean_pred(s: str) -> str:
    s = s.strip()
    # Hard stop at common endings / tag re-appearances
    for stop in ["<en>", "<ja>", "<zh>"]:
        if stop in s:
            s = s.split(stop, 1)[0].strip()
    # Optional: keep only first sentence (good for JA/ZH)
    for sep in ["。", "！", "？", ".", "!", "?"]:
        if sep in s:
            s = s.split(sep, 1)[0] + sep
            break
    return s

# Decode ONLY the continuation
generated_tokens = out[0][input_len:]
decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("PROMPT:")
print(prompt)
print("\nMODEL OUTPUT:")
print(clean_pred(decoded))
