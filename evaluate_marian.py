# evaluate_wudi.py
import argparse
from datasets import load_dataset
from transformers import MarianTokenizer, MarianMTModel
from sacrebleu import corpus_bleu, corpus_chrf
from sacrebleu.metrics import BLEU
from config import LANG_TAGS, LANG_COLUMNS

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--base_model", type=str, default="Helsinki-NLP/opus-mt-en-mul")
    p.add_argument("--lang_pair", type=str, required=True)   # "en-zh" or "en-ja"
    p.add_argument("--max_samples", type=int, default=2000)
    return p.parse_args()

FLORES_CODES = {
    "en-ja": ("eng_Latn", "jpn_Jpan"),
    "en-zh": ("eng_Latn", "cmn_Hans"),
}

def main():
    args = parse_args()

    # Load tokenizer from the *base* model (not the merged checkpoint)
    tokenizer = MarianTokenizer.from_pretrained(args.base_model)
    model = MarianMTModel.from_pretrained(args.model_dir)

    # Language info
    src_lang, tgt_lang = LANG_COLUMNS[args.lang_pair]
    lang_tag = LANG_TAGS[args.lang_pair]

    # Load OPUS100 test split
    # raw = load_dataset("opus100", args.lang_pair)["test"]

    # # Optionally truncate for fast evaluation
    # raw = raw.select(range(min(args.max_samples, len(raw))))

    # sources = [ex[src_lang] for ex in raw["translation"]]
    # targets = [ex[tgt_lang] for ex in raw["translation"]]
    
    src_code, tgt_code = FLORES_CODES[args.lang_pair]

    src_ds = load_dataset("openlanguagedata/flores_plus", src_code, split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_code, split="devtest")

    # Cap samples
    n = min(args.max_samples, len(src_ds), len(tgt_ds)) if args.max_samples else min(len(src_ds), len(tgt_ds))
    src_ds = src_ds.select(range(n))
    tgt_ds = tgt_ds.select(range(n))

    sources = src_ds["text"]
    targets = tgt_ds["text"]

    # Add language tag:  ">>zho<< <sentence>"
    src_tagged = [f"{lang_tag} {s}" for s in sources]

    # Tokenize
    batch = tokenizer(
        src_tagged, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    # Move to GPU if available
    device = "cuda" if model.device.type == "cuda" else "cpu"
    model = model.to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Generate translations
    outputs = model.generate(
        **batch,
        max_length=128,
        num_beams=5,
    )
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
