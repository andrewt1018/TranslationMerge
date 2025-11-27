# train_marian.py
import argparse
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from config import (
    BASE_MODEL_NAME,
    BATCH_SIZE,
    LR,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    SAVE_STEPS,
    EVAL_STEPS,
    LOGGING_STEPS,
    SAVE_TOTAL_LIMIT,
    WARMUP_STEPS,
)
from data import load_opus100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_pair", type=str, required=True, help="e.g., en-zh or en-ja")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Shared base model + tokenizer
    tokenizer = MarianTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = MarianMTModel.from_pretrained(BASE_MODEL_NAME)

    # 2. Load tokenized data for the given lang_pair
    train_dataset, eval_dataset = load_opus100(args.lang_pair, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 3. Training arguments (with warmup)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        predict_with_generate=True,
        fp16=True,
        report_to=["wandb"],
        run_name=f"opus_en_mul_{args.lang_pair}",
    )

    trainer = Seq2SeqTrainer(
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
