from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import BertTokenizer
import openai
from prompt import *


class StanceDetection:
    def __init__(self, model="models/produce/checkpoint-825", tokenizer=None):
        id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
        label2id = {"AGAINST": 0, "POSITIVE": 1, "NEITHER": 2}
        model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=3, id2label=id2label, label2id=label2id
        )
        self.model = model
        # tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        self.tokenizer = tokenizer

    def classify(self, text, target):
        classfier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        result = classfier(text + f"[SEP]  The stance of the aformentioned text to target: {target} is [MASK]")
        return result

    def __call__(self, text, target):
        return self.classify(text, target)

    def process_csv(self, csv_file):
        classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)


class LLMInference:
    def __init__(self, model, api_base, api_key):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        openai = openai.OpenAI(api_key=api_key, api_base=api_base, model=model)

    def inference(self, text):
        response = openai.Completion.create(engine=self.model, prompt=text)
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
    def __init__(self, model, api_base, api_key):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        openai = openai.OpenAI(api_key=api_key, api_base=api_base, model=model)

    def inference(self, text):
        response = openai.Completion.create(engine=self.model, prompt=text)
        return response.choices[0].text

    def __call__(self, text):
        return self.inference(text)

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
