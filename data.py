# data.py
from datasets import load_dataset
from typing import Dict, Any, Tuple
from config import LANG_COLUMNS, LANG_TAGS, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH

def load_opus100(lang_pair: str, tokenizer) -> Tuple[Any, Any]:
    """
    lang_pair: "en-zh" or "en-ja"
    Returns: (tokenized_train_dataset, tokenized_valid_dataset)
    """
    assert lang_pair in LANG_COLUMNS, f"Unsupported lang_pair: {lang_pair}"

    src_lang, tgt_lang = LANG_COLUMNS[lang_pair]
    tgt_lang_tag = LANG_TAGS[lang_pair]  # ">>zho<<" or ">>jpn<<"

    raw = load_dataset("opus100", lang_pair.replace("-", "-"))  # "en-zh", "en-ja"

    def preprocess_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        translations = batch["translation"]
        src_texts = [ex[src_lang] for ex in translations]
        tgt_texts = [ex[tgt_lang] for ex in translations]

        # IMPORTANT: prepend target language token to the source
        # e.g., ">>zho<< This is a sentence."
        src_with_tag = [f"{tgt_lang_tag} {s}" for s in src_texts]

        model_inputs = tokenizer(
            src_with_tag,
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_texts,
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = raw["train"].map(
        preprocess_batch,
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    tokenized_valid = raw["validation"].map(
        preprocess_batch,
        batched=True,
        remove_columns=raw["validation"].column_names,
    )

    return tokenized_train, tokenized_valid
