import json
import torch
from transformers import AdamW
from transformers.optimization import get_scheduler
from transformers import BertTokenizer
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer


# 处理数据集

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


datasets = load_dataset("csv", data_files={"train": "data/processed/train.csv", "test": "data/processed/test.csv"})
train_datasets = datasets["train"]
test_datasets = datasets["test"]
tokenized_train_datasets = train_datasets.map(tokenize_function, batched=True)
tokenized_test_datasets = test_datasets.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased", num_labels=3)
training_args = TrainingArguments(output_dir="models/output")


metric = evaluate.load("f1")


training_args = TrainingArguments(output_dir="models/output", eval_strategy="epoch")

id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}

label2id = {"AGAINST": 0, "POSITIVE": 1, "NEITHER": 2}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_test_datasets,
)

trainer.train()
