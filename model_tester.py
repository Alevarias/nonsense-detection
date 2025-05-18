import os
import glob
import torch
from datasets import load_dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Constants
CHECKPOINT_ROOT = "trained_models"      # Root directory containing checkpoint-* subfolders
TRAIN_PATH      = "multinli_1.0/multinli_1.0_train.jsonl"
TEST_SIZE       = 0.05                # Fraction of data to hold out as validation
MAX_SOURCE_LEN  = 256                 # Must match training preprocessing
MAX_TARGET_LEN  = 32                  # Expected single-token prediction length
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
correct = 0

# Finding the latest checkpoint
checkpoint_dirs = glob.glob(os.path.join(CHECKPOINT_ROOT, 'checkpoint-*'))
if not checkpoint_dirs:
    raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_ROOT}, please check the path")
latest_ckpt = sorted(
    checkpoint_dirs,
    key=lambda x: int(x.rsplit('-', 1)[-1])
)[-1]
print(f"Loading model from latest checkpoint: {latest_ckpt} on {DEVICE}...")

# Loading the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(latest_ckpt)
model     = T5ForConditionalGeneration.from_pretrained(latest_ckpt)
model.to(DEVICE)
model.eval()

# Loading and splitting the dataset
print("Loading and splitting dataset...")
raw_ds = load_dataset("json", data_files={"train": TRAIN_PATH})
full = raw_ds["train"].filter(lambda ex: ex["gold_label"] != "-")
split = full.train_test_split(test_size=TEST_SIZE, seed=42)
dataset = DatasetDict({
    "train": split["train"],
    "validation": split["test"]
})
val_ds = dataset["validation"]

# Iterating through the validation set
print(f"Evaluating {len(val_ds)} examples...")
for ex in val_ds:
    premise    = ex["sentence1"]
    hypothesis = ex["sentence2"]
    true_label = ex["gold_label"]

    # Preparing input text
    text_in = f"mnli premise: {premise} hypothesis: {hypothesis}"
    inputs = tokenizer(
        text_in,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding=False,
        return_tensors="pt"
    ).to(DEVICE)

    # Generating prediction
    with torch.no_grad():
        pred_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=MAX_TARGET_LEN,
            num_beams=1
        )
    pred_label = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

    # Counts correct predictions
    if pred_label == true_label:
        correct += 1

    # Printing the result
    print("Premise:    ", premise)
    print("Hypothesis: ", hypothesis)
    print("Prediction: ", pred_label)
    print("True Label: ", true_label)
    print("---")

# Variables to hold accuracy and F1 scores

total = len(val_ds)
accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")
print(f"Total examples: {total}")

