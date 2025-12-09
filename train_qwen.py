# train_qwen.py
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from config import (
    QWEN_MODEL_NAME,
    QWEN_BATCH_SIZE,
    QWEN_LR,
    QWEN_NUM_EPOCHS,
    QWEN_WEIGHT_DECAY,
    QWEN_SAVE_STEPS,
    QWEN_EVAL_STEPS,
    QWEN_LOGGING_STEPS,
    QWEN_SAVE_TOTAL_LIMIT,
    QWEN_WARMUP_STEPS,
    QWEN_LANG_TAGS,
)
from data_qwen import load_opus100_qwen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_pair", type=str, required=True, help="en-ja or en-zh")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load tokenizer + add special language tags
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, use_fast=True)

    special_tokens = set()
    for lp, tags in QWEN_LANG_TAGS.items():
        special_tokens.add(tags["src"])
        special_tokens.add(tags["tgt"])
    special_tokens = sorted(list(special_tokens))

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    print(f"Added {num_added} special tokens to tokenizer:", special_tokens)

    # Qwen2.5 base doesn’t always have a pad_token; set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model and resize embeddings for new tokens
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Load datasets
    train_dataset, eval_dataset = load_opus100_qwen(args.lang_pair, tokenizer)

    # 4. Data collator (no MLM – pure causal LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=QWEN_EVAL_STEPS,
        save_steps=QWEN_SAVE_STEPS,
        logging_steps=QWEN_LOGGING_STEPS,
        learning_rate=QWEN_LR,
        warmup_steps=QWEN_WARMUP_STEPS,
        per_device_train_batch_size=QWEN_BATCH_SIZE,
        per_device_eval_batch_size=QWEN_BATCH_SIZE,
        num_train_epochs=QWEN_NUM_EPOCHS,
        weight_decay=QWEN_WEIGHT_DECAY,
        save_total_limit=QWEN_SAVE_TOTAL_LIMIT,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],  # if you’ve been using W&B
        run_name=f"qwen2.5_0.5B_{args.lang_pair}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
