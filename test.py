from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score

dataset = load_dataset("yelp_review_full")
dataset["train"][100]
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)


training_args = TrainingArguments(output_dir="test_trainer")


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)
    f1 = f1_score(labels.cpu().numpy(), predictions, average="weighted")
    loss = 1 - f1  # 损失函数为 1 - F1 分数
    return (loss, outputs) if return_outputs else loss


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
