import argparse
import numpy as np

from transformers import (
    MarianMTModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

from config import (
    MARIAN_MODELS,
    BATCH_SIZE,
    LR,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    EVAL_STEPS,
    SAVE_STEPS,
    LOGGING_STEPS,
    SAVE_TOTAL_LIMIT,
)
from data import get_tokenized_datasets, get_tokenizer


def build_bleu_metric(tokenizer):
    """
    Wrap sacrebleu in a function that Trainer can use.
    """
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]  # sacrebleu expects list-of-lists
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace masked label tokens (-100) with pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_pair",
        type=str,
        required=True,
        choices=["en-zh", "en-ja"],
        help="Which language pair to train on.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the fine-tuned model.",
    )
    args = parser.parse_args()

    # 1. Tokenizer + datasets
    tokenizer = get_tokenizer(args.lang_pair)
    tokenizer, train_ds, val_ds, test_ds = get_tokenized_datasets(args.lang_pair)

    # 2. Load the pre-trained Marian model for this direction
    model_name = MARIAN_MODELS[args.lang_pair]
    model = MarianMTModel.from_pretrained(model_name)

    # 3. Data collator & metrics
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = build_bleu_metric(tokenizer)

    # 4. TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,   # <- keeps disk usage low
        predict_with_generate=True,
        fp16=True,                           # turn off if your GPU doesn't support it
        report_to=["wandb"],  # enable W&B integration
        run_name=f"marian_{args.lang_pair}",  # shows up in W&B UI
    )

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Train and save
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Optional: evaluate on test split
    if test_ds is not None:
        metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
