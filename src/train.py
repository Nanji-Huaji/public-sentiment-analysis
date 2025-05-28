import json
import torch
from transformers.optimization import get_scheduler
from transformers.models.bert import BertTokenizer
from datasets import load_dataset
import numpy as np
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
import pandas as pd

import argparse

import logging
from datetime import datetime
from logger import Logger
import sys

import os

import shutil


def get_checkpoint_path(file_path) -> str | None:
    """
    从文件路径中提取模型检查点路径
    """
    subdirs = [os.path.join(file_path, d) for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]

    if not subdirs:
        return None  # 如果没有子目录，返回 None

    # 按最后修改时间排序，获取最新的子目录
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir


parser = argparse.ArgumentParser("Description: Train a sentiment analysis model")
parser.add_argument(
    "--class_counts",
    type=float,
    nargs=3,
    default=[0.4152, 0.3953, 0.1895],
    help="The ratio of each class in the dataset",
)
parser.add_argument(
    "--train_epochs",
    type=int,
    default=50,
    help="The number of epochs to train the model",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="google-bert/bert-base-multilingual-cased",
    help="The path to the pre-trained model",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置日志

# 获取当前时间
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 构造日志文件名
log_file_name = f"logs/{args.checkpoint_path.replace('/', '_')}_{args.train_epochs}_{current_time}.log"

# 配置日志记录
logging.basicConfig(
    filename=log_file_name,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

sys.stdout = Logger(log_file_name)


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


datasets = load_dataset(
    "csv", data_files={"train": "data/processed/train.csv", "test": "data/csv_data/Weibo-SD/test.csv"}
)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
label2id = {v: k for k, v in id2label.items()}

# 训练模型

id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
label2id = {"AGAINST": 0, "POSITIVE": 1, "NEITHER": 2}

model = AutoModelForSequenceClassification.from_pretrained(
    args.checkpoint_path, num_labels=3, id2label=id2label, label2id=label2id, local_files_only=True
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
    output_dir="/home/tiantianyi/code/public-sentiment-analysis/models",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=args.train_epochs,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=10,
)


logging.info("Training Arguments:")
for key, value in vars(training_args).items():
    logging.info(f"{key}: {value}")

# 写入训练开始日志
logging.info("Training started...")


# 自定义Trainer处理类别不平衡
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 给出类别不平衡的权重
        class_counts = args.class_counts  # 各标签比例
        device = next(model.parameters()).device
        weights = torch.tensor([1 / (c + 1e-5) for c in class_counts], dtype=torch.float32, device=device)  # 逆频率加权
        weights = weights / weights.sum() * len(class_counts)  # 归一化
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir, _internal_call):
        if output_dir is None:
            raise ValueError("Output directory must be specified.")
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        super().save_model(output_dir, _internal_call)

            # 确保 logs/json 目录存在
        os.makedirs("logs/json", exist_ok=True)

        # 定义 trainer_state.json 的路径
        trainer_state_path = os.path.join(output_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            # 复制 trainer_state.json 到 logs/json 目录
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_path = os.path.join("logs/json", f"trainer_state_{os.path.basename(output_dir)}_{current_time}.json")
            shutil.copy(trainer_state_path, dest_path)
            logging.info(f"已将 {trainer_state_path} 复制到 {dest_path}")


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
