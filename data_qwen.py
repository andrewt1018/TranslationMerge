# data_qwen.py
from typing import Dict, Any, Tuple

from datasets import load_dataset
import torch

from config import (
    LANG_COLUMNS,
    QWEN_LANG_TAGS,
    QWEN_MAX_SOURCE_LENGTH,
    QWEN_MAX_TARGET_LENGTH,
    QWEN_MAX_SEQ_LEN,
)


def load_opus100_qwen(lang_pair: str, tokenizer) -> Tuple[Any, Any]:
    """
    Prepare OPUS-100 for Qwen causal LM finetuning.

    For each example, we build:
      text = "<en> {src} <ja> {tgt}"
    and then create labels that are -100 on the source part and
    normal token ids on the target part.
    """
    assert lang_pair in LANG_COLUMNS, f"Unsupported lang_pair: {lang_pair}"
    src_lang, tgt_lang = LANG_COLUMNS[lang_pair]
    tags = QWEN_LANG_TAGS[lang_pair]  # {"src": "<en>", "tgt": "<ja>"}

    dataset = load_dataset("opus100", lang_pair)

    def preprocess_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        translations = batch["translation"]
        src_texts = [ex[src_lang] for ex in translations]
        tgt_texts = [ex[tgt_lang] for ex in translations]

        # Build raw text sequences
        # e.g. "<en> I like cats. <ja> 猫が好きです。"
        texts = [
            f"{tags['src']} {src} {tags['tgt']} {tgt}"
            for src, tgt in zip(src_texts, tgt_texts)
        ]

        encodings = tokenizer(
            texts,
            max_length=QWEN_MAX_SEQ_LEN,
            truncation=True,
            padding=False,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        labels = []
        for ids in input_ids:
            # We want to ignore loss on everything up to and including the target tag.
            # Strategy:
            #   1. find the position of the target tag token id;
            #   2. set labels[:pos+1] = -100, labels[pos+1:] = input_ids[pos+1:]
            tgt_tag_id = tokenizer.convert_tokens_to_ids(tags["tgt"])
            # find first occurrence
            try:
                pos = ids.index(tgt_tag_id)
            except ValueError:
                # very unlikely, but be safe
                pos = 0
            lab = [-100] * len(ids)
            for j in range(pos + 1, len(ids)):
                lab[j] = ids[j]
            labels.append(lab)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = dataset["train"].map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    valid_dataset = dataset["validation"].map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    # Set dataset format for PyTorch
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    valid_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset, valid_dataset
