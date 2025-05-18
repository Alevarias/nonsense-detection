import os
from datasets import load_dataset, DatasetDict
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import torch
import numpy as np
import evaluate
import math

# Check CUDA
print("CUDA available:", torch.cuda.is_available())

# Paths & hyperparameters
TRAIN_PATH      = "multinli_1.0/multinli_1.0_train.jsonl"
OUTPUT_DIR      = "trained_models"
MODEL_NAME      = "t5-small"
MAX_SOURCE_LEN  = 256
MAX_TARGET_LEN  = 32
BATCH_SIZE      = 2
ACCUM_STEPS     = 4
NUM_EPOCHS      = 10
LEARNING_RATE   = 3e-5
LOGGING_STEPS   = 100
EVAL_STEPS      = 10000
SAVE_STEPS      = 10000

# Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_preds):
    pred_ids, label_ids = eval_preds

    # 1) Decode predictions & references to label strings
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = []
    for ref_seq in label_ids:
        valid = [tok for tok in ref_seq if tok != -100]
        decoded_labels.append(
            tokenizer.decode(valid, skip_special_tokens=True)
        )

    # 2) Map strings to class IDs
    label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    y_pred = [label2id.get(text.strip(), -1) for text in decoded_preds]
    y_true = [label2id.get(text.strip(), -1) for text in decoded_labels]

    # 3) Filter out any invalid entries
    pairs = [(p, t) for p, t in zip(y_pred, y_true) if p >= 0 and t >= 0]
    if not pairs:
        return {"accuracy": 0.0, "f1_contradiction": 0.0}
    preds_f, labels_f = zip(*pairs)

    # 4) Compute accuracy
    acc = accuracy_metric.compute(predictions=preds_f, references=labels_f)

    # 5) Compute per-class F1 with average=None (not the string!)
    f1_all = f1_metric.compute(
        predictions=preds_f,
        references=labels_f,
        average=None,       # Python None, not the string "none"
        labels=[0, 1, 2]
    )

    return {
        "accuracy": acc["accuracy"],
        "f1_contradiction": f1_all["f1"][2],
    }

# Preprocessing
def preprocess(batch):
    inputs = [f"mnli premise: {p} hypothesis: {h}"
              for p,h in zip(batch["sentence1"], batch["sentence2"])]
    model_inputs = tokenizer(
        inputs, max_length=MAX_SOURCE_LEN,
        truncation=True, padding="max_length"
    )
    labels = tokenizer(
        batch["gold_label"], max_length=MAX_TARGET_LEN,
        truncation=True, padding="max_length"
    )
    label_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100)
         for tok in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs


if __name__ == "__main__":
    # Load & split dataset
    raw_ds = load_dataset("json", data_files={"train": TRAIN_PATH})
    ds = raw_ds["train"].filter(lambda ex: ex["gold_label"] != "-")
    split = ds.train_test_split(test_size=0.01, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Tokenizer & model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(
        preprocess, batched=True,
        remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    num_train_examples = len(tokenized["train"])
    updates_per_epoch = math.ceil(num_train_examples / (BATCH_SIZE * ACCUM_STEPS))
    half_epoch_updates = updates_per_epoch // 2

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch & accumulation
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=ACCUM_STEPS,

        # Optimization
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=NUM_EPOCHS,

        # Logging
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],

        # Evaluation
        eval_strategy="steps",
        # eval_steps=half_epoch_updates,
        eval_steps=EVAL_STEPS,
        predict_with_generate=True,

        # Checkpointing
        save_strategy="steps",
        # save_steps=half_epoch_updates,
        save_steps=SAVE_STEPS,
        save_total_limit=10,              # keep the last 10

        # Mixed precision & memory tricks
        fp16=True,
        gradient_checkpointing=True,

        # Keep best model by lowest eval loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(OUTPUT_DIR)
