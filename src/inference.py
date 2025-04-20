import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import BertTokenizer
import openai
import torch
import pandas as pd
from prompt import *
from transformers import AutoModelForCausalLM


class StanceDetection:
    def __init__(self, model="models/produce/f1-46", tokenizer="google-bert/bert-base-multilingual-cased"):
        self.id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
        self.label2id = {"AGAINST": 0, "POSITIVE": 1, "NEITHER": 2}
        model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=3, id2label=self.id2label, label2id=self.label2id
        )
        self.model = model
        # tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer

    def classify(self, text, target):
        inputs = self.tokenizer(
            text=text, text_pair=target, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs).item()
        return {
            "label": self.id2label[pred_id],
            "score": probs[0][pred_id].item(),
            "details": {self.id2label[i]: probs[0][i].item() for i in range(len(self.id2label))},
        }

    def raw_classify(self, text, target):
        inputs = self.tokenizer(
            text=text, text_pair=target, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs).item()
        return {
            "label": self.id2label[pred_id],
            "score": probs[0][pred_id].item(),
            "details": {self.id2label[i]: probs[0][i].item() for i in range(len(self.id2label))},
        }

    def __call__(self, text, target):
        return self.classify(text, target)

    def process_csv(self, csv_file):
        # 给定csv文件，向这个csv文件中添加一列，列名为stance，内容为对应的stance（0, 1, 2）
        df = pd.read_csv(csv_file)
        stances = list(map(lambda x: self.classify(x[0], x[1]), zip(df["text"], df["target"])))
        df["stance"] = [stance["label"] for stance in stances]
        df.to_csv(csv_file, index=False)
        return df


class LLMInference:
    def __init__(self, model, api_base, api_key):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, api_base=api_base, model=model)

    def inference(self, text):
        response = self.client.chat.Completion.create(engine=self.model, prompt=text)
        return response.choices[0].text

    def __call__(self, text):
        return self.inference(text)

    def analyze(self, summary, target, **kwargs):
        favor_rate = kwargs.get("favor_rate", "未提供")
        neutral_rate = kwargs.get("neutral_rate", "未提供")
        against_rate = kwargs.get("against_rate", "未提供")
        top_words = kwargs.get("top_words", "未提供")
        prompt = analyze_prompt.format(
            summary=summary,
            target=target,
            favor_rate=favor_rate,
            neutral_rate=neutral_rate,
            against_rate=against_rate,
            top_words=top_words,
        )
        analysis = self.inference(prompt)
        return analysis


class SLMInference:
    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def inference(self, prompt):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def __call__(self, prompt):
        return self.inference(prompt)

    def summary(self, favor_text, neutral_text, against_text, target, **kwargs):
        favor_rate = kwargs.get("favor_rate", "未提供")
        neutral_rate = kwargs.get("neutral_rate", "未提供")
        against_rate = kwargs.get("against_rate", "未提供")
        top_words = kwargs.get("top_words", "未提供")
        prompt = summarize_prompt.format(
            favor_text=favor_text,
            neutral_text=neutral_text,
            against_text=against_text,
            target=target,
            favor_rate=favor_rate,
            neutral_rate=neutral_rate,
            against_rate=against_rate,
            top_words=top_words,
        )
        summary = self.inference(prompt)
        return summary
