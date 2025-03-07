import json
import torch


from transformers import AdamW
from transformers.optimization import get_scheduler
from transformers import BertTokenizer
import os

os.environ["TRANSFORMERS_CACHE"] = "models/pretrained"

from datasets import load_dataset

ds = load_dataset("strombergnlp/nlpcc-stance")
