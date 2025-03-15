from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments
import evaluate

id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
label2id = {"AGAINST": 0, "POSITIVE": 1, "NEITHER": 2}

model = BertForSequenceClassification.from_pretrained(
    "models/output/f1-55", num_labels=3, label2id=label2id, id2label=id2label
)
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


def tokenize_function(examples):
    texts = [
        text + f" [SEP]  The stance of the aformentioned text to target: {target} is: [MASK]"
        for text, target in zip(examples["text"], examples["target"])
    ]
    return tokenizer(texts, padding="max_length", truncation=True)


datasets = load_dataset("csv", data_files={"train": "data/processed/train.csv", "test": "data/processed/test.csv"})
test_datasets = datasets["test"]
tokenized_test_datasets = test_datasets.map(tokenize_function, batched=True)


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     f1 = f1_score(labels, predictions, average="weighted")
#     return {"f1": f1}

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


training_args = TrainingArguments(output_dir="models/output", eval_strategy="epoch", num_train_epochs=3)


trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_datasets,
    compute_metrics=compute_metrics,
)


eval_loop = 3
result = []

for i in range(eval_loop):
    result.append(trainer.evaluate())
    print(f"Eval {i + 1}/{eval_loop}")
    print(result[-1])

print("Average F1:", sum([r["eval_f1"] for r in result]) / eval_loop)
