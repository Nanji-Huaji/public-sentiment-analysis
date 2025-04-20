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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(
    "models/output/f1-46", num_labels=3, label2id=label2id, id2label=id2label
).to(device)
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


def tokenize_function(dataset):
    return tokenizer(
        text=dataset["text"],
        text_pair=dataset["target"],
        max_length=256,
        truncation="only_first",  # 优先截断文本（保留完整目标）
        padding="max_length",
        add_special_tokens=True,
        return_attention_mask=True,
    )


datasets = load_dataset(
    "csv", data_files={"train": "data/csv_data/Weibo-SD/train.csv", "test": "data/csv_data/Weibo-SD/test.csv"}
)
test_datasets = datasets["test"]
tokenized_test_datasets = test_datasets.map(tokenize_function, batched=True)

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


eval_loop = 1
result = []

for i in range(eval_loop):
    result.append(trainer.evaluate())
    print(f"Eval {i + 1}/{eval_loop}")
    print(result[-1])

print("Average F1:", sum([r["eval_f1"] for r in result]) / eval_loop)
