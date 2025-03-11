from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import BertTokenizer
import openai


class StanceDetection:
    def __init__(self, model="models/output/checkpoint-825", tokenizer=None):
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
        pass


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


class SLMInference:
    def __init__(self, model, api_base, api_key):
        pass

    def inference(self, text):
        pass

    def __call__(self, text):
        return self.inference(text)

    def summary(self, text):
        pass
