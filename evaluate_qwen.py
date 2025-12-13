# evaluate_qwen.py
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu
import torch

from config import LANG_COLUMNS, QWEN_LANG_TAGS, QWEN_MAX_SEQ_LEN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to fine-tuned Qwen model dir (e.g. models/qwen_en_ja_ft)")
    p.add_argument("--lang_pair", type=str, required=True,
                   help="Language pair: en-ja or en-zh")
    p.add_argument("--max_samples", type=int, default=1000,
                   help="Max number of test examples to evaluate")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()

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

@torch.no_grad()
def generate_translations(model, tokenizer, prompts, batch_size=8, max_new_tokens=128, device="cuda"):
    all_preds = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        # Tokenize prompts
        enc = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        # input_len = enc["input_ids"].shape[1]
        prompt_len = enc["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        gen_only = out[:, prompt_len:]
        for j in range(gen_only.size(0)):
            decoded = tokenizer.decode(gen_only[j], skip_special_tokens=True)
            all_preds.append(clean_pred(decoded))
            
        # generated_tokens = out[0][input_len:]
        # decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # all_preds.append(clean_pred(decoded))

    return all_preds

def main():
    args = parse_args()

    assert args.lang_pair in LANG_COLUMNS, f"Unsupported lang_pair: {args.lang_pair}"
    src_lang, tgt_lang = LANG_COLUMNS[args.lang_pair]
    tags = QWEN_LANG_TAGS[args.lang_pair]  # {"src": "<en>", "tgt": "<ja>"} or "<zh>"
    print(f"Tags: {tags}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load OPUS-100 test split
    dataset = load_dataset("opus100", args.lang_pair)["test"]
    if args.max_samples is not None:
        n = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(n))

    sources = [ex[src_lang] for ex in dataset["translation"]]
    targets = [ex[tgt_lang] for ex in dataset["translation"]]

    # Build prompts: "<en> source <ja>" or "<en> source <zh>"
    prompts = [f"{tags['src']} {src} {tags['tgt']}" for src in sources]

    # Generate translations from Qwen
    preds = generate_translations(
        model, tokenizer, prompts,
        batch_size=args.batch_size,
        max_new_tokens=128,
        device=device
    )
    
    # Compute BLEU
    bleu = corpus_bleu(preds, [targets])
    print(f"\n=== Evaluation for {args.model_dir} on {args.lang_pair} ===")
    print("BLEU:", bleu.score)


if __name__ == "__main__":
    main()
