# evaluate_qwen.py
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu, corpus_chrf
from sacrebleu.metrics import BLEU
import torch

from config import LANG_COLUMNS, QWEN_LANG_TAGS, QWEN_MAX_SEQ_LEN

FLORES_CODES = {
    "en-ja": ("eng_Latn", "jpn_Jpan"),
    "en-zh": ("eng_Latn", "cmn_Hans"),
}

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
    # Keep only first sentence (good for JA/ZH)
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
   
    return all_preds

def main():
    args = parse_args()

    assert args.lang_pair in LANG_COLUMNS, f"Unsupported lang_pair: {args.lang_pair}"
    src_lang, tgt_lang = LANG_COLUMNS[args.lang_pair]
    tags = QWEN_LANG_TAGS[args.lang_pair]  
    print(f"Tags: {tags}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load OPUS-100 test split
    # dataset = load_dataset("opus100", args.lang_pair)["test"]
    # if args.max_samples is not None:
    #     n = min(args.max_samples, len(dataset))
    #     dataset = dataset.select(range(n))

    # sources = [ex[src_lang] for ex in dataset["translation"]]
    # targets = [ex[tgt_lang] for ex in dataset["translation"]]
    
    src_code, tgt_code = FLORES_CODES[args.lang_pair]

    src_ds = load_dataset("openlanguagedata/flores_plus", src_code, split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_code, split="devtest")

    # Cap samples
    n = min(args.max_samples, len(src_ds), len(tgt_ds)) if args.max_samples else min(len(src_ds), len(tgt_ds))
    src_ds = src_ds.select(range(n))
    tgt_ds = tgt_ds.select(range(n))

    sources = src_ds["text"]
    targets = tgt_ds["text"]
    
    prompts = [f"{tags['src']} {src} {tags['tgt']}" for src in sources]
    # Generate translations from Qwen
    preds = generate_translations(
        model, tokenizer, prompts,
        batch_size=args.batch_size,
        max_new_tokens=128,
        device=device
    )
    
    # Compute BLEU
    if args.lang_pair == "en-ja":
        score = corpus_chrf(preds, [targets])
    else:
        bleu = BLEU(tokenize="zh")
        score = bleu.corpus_score(preds, [targets])
    print(f"\n=== Evaluation for {args.model_dir} on {args.lang_pair} ===")
    print("BLEU:", score.score)


if __name__ == "__main__":
    main()
