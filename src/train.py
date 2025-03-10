import json
import torch


from transformers import AdamW
from transformers.optimization import get_scheduler
from transformers import BertTokenizer
import os

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased", num_labels=3)
training_args = TrainingArguments(output_dir="models/output")
