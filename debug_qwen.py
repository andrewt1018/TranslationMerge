import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu, corpus_chrf
from sacrebleu.metrics import BLEU

MODEL_DIR = "models/qwen_ta_merged"
LANG_PAIR = "en-zh"

# Simple test sentence
src_sentence = "I like machine learning."
target = "我喜欢机器学习。"

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
decoded = clean_pred(tokenizer.decode(generated_tokens, skip_special_tokens=True))
if LANG_PAIR == "en-ja":
    score = corpus_chrf([decoded], [target])
else:
    bleu = BLEU(tokenize="zh")
    score = bleu.corpus_score([decoded], [target])

print("PROMPT:")
print(prompt)
print("TARGET:")
print(target)
print("MODEL OUTPUT:")
print(decoded)
print("BLEU:", score.score)
