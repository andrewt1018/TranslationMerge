from typing import Tuple
from datasets import load_dataset
from transformers import MarianTokenizer

from config import (
    DATASET_NAME,
    DATASET_CONFIGS,
    LANG_KEYS,
    MAX_LENGTH,
    MARIAN_MODELS,
)


def get_tokenizer(lang_pair: str) -> MarianTokenizer:
    """
    Load the tokenizer corresponding to the Marian model for this language pair.
    Even though we use OPUS-100 for fine-tuning, we want the tokenizer to match
    the pre-trained model (vocab, special tokens, etc.).
    """
    model_name = MARIAN_MODELS[lang_pair]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return tokenizer


def load_raw_datasets(lang_pair: str):
    """
    Load OPUS-100 for the given language pair, e.g. 'en-zh' or 'en-ja'.
    Splits: train / validation / test (if available).
    """
    config_name = DATASET_CONFIGS[lang_pair]
    raw = load_dataset(DATASET_NAME, config_name)
    # raw is a DatasetDict with keys like "train", "validation", "test"
    return raw


def make_preprocess_fn(tokenizer: MarianTokenizer, lang_pair: str):
    """
    Build a preprocessing function to map raw OPUS-100 samples -> model inputs.

    OPUS-100 format: each example is:
      {"translation": {"en": "...", "zh": "..."}}
    or  {"translation": {"en": "...", "ja": "..."}}

    We always translate EN -> target (ZH or JA).
    """

    src_key, tgt_key = LANG_KEYS[lang_pair]

    def preprocess(batch):
        translations = batch["translation"]
        src_texts = [ex[src_key] for ex in translations]
        tgt_texts = [ex[tgt_key] for ex in translations]

        # Tokenize source (encoder input)
        model_inputs = tokenizer(
            src_texts,
            max_length=MAX_LENGTH,
            truncation=True,
        )

        # Tokenize targets (decoder labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_texts,
                max_length=MAX_LENGTH,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def get_tokenized_datasets(lang_pair: str):
    """
    High-level helper:
      - load raw OPUS-100 for lang_pair
      - build preprocess function
      - map over train/validation/test splits

    Returns: tokenizer, train_ds, val_ds, test_ds
    """
    tokenizer = get_tokenizer(lang_pair)
    raw = load_raw_datasets(lang_pair)
    preprocess_fn = make_preprocess_fn(tokenizer, lang_pair)

    tokenized = {}
    for split in ["train", "validation", "test"]:
        if split in raw:
            tokenized[split] = raw[split].map(
                preprocess_fn,
                batched=True,
                remove_columns=raw[split].column_names,
            )
        else:
            tokenized[split] = None

    return tokenizer, tokenized["train"], tokenized["validation"], tokenized["test"]
