import json
import torch
from transformers import AdamW
from transformers.optimization import get_scheduler
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 处理数据集

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


# 数据预处理
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


datasets = load_dataset("csv", data_files={"train": "data/processed/train.csv", "test": "data/processed/test.csv"})

tokenized_datasets = datasets.map(tokenize_function, batched=True)

id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
label2id = {v: k for k, v in id2label.items()}

# 训练模型

id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
label2id = {"AGAINST": 0, "POSITIVE": 1, "NEITHER": 2}

model = AutoModelForSequenceClassification.from_pretrained(
    "models/output/f1-46", num_labels=3, id2label=id2label, label2id=label2id
)


# 评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions, average="weighted"),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def save_error_samples(trainer, dataset, output_file="logs/errors.csv"):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    errors = []
    for i, (pred, label) in enumerate(zip(preds, dataset["label"])):
        if pred != label:
            errors.append(
                {
                    "text": dataset[i]["text"],
                    "target": dataset[i]["target"],
                    "true_label": id2label[label],
                    "pred_label": id2label[pred],
                }
            )
    pd.DataFrame(errors).to_csv(output_file)


# 训练配置
training_args = TrainingArguments(
    output_dir="models/output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)


# 自定义Trainer处理类别不平衡
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 给出类别不平衡的权重
        class_counts = [0.4111, 0.3794, 0.2095]  # 各标签比例
        device = next(model.parameters()).device
        weights = torch.tensor([1 / (c + 1e-5) for c in class_counts], dtype=torch.float32, device=device)  # 逆频率加权
        weights = weights / weights.sum() * len(class_counts)  # 归一化
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# 初始化Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)


trainer.train()

save_error_samples(trainer, tokenized_datasets["test"], output_file="logs/errors.csv")
