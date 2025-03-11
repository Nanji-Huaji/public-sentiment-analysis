from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class StanceDetection:
    def __init__(self, model, tokenizer):
        id2label = {0: "AGAINST", 1: "POSITIVE", 2: "NEITHER"}
        label2id = {"AGAINST": 0, "POSITIVE": 1, "NEITHER": 2}
        model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=3, id2label=id2label, label2id=label2id
        )
        self.model = model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer

    def classify(self, text, target):
        classfier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        result = classfier(text + f"[SEP]  The stance of the aformentioned text to target: {target} is [MASK]")
        return result

    def __call__(self, text, target):
        return self.classify(text, target)
