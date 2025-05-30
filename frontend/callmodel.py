import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import jieba

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from inference import StanceDetection, LLMInference, SLMInference


class CallModel:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model")
        if self.model in ["GPT-4o", "DeepSeek-r1", "GPT-3.5", "GPT-3"]:
            self.model = LLMInference(model=self.model, api_base=kwargs.get("api_base"), api_key=kwargs.get("api_key"))
        elif self.model in ["Qwen/Qwen2.5-7B-Instruct"]:
            self.model = SLMInference()

    def call_crawler_necessary(self, start_time, end_time):
        # 遍历data/crawler下的所有文件，如果有符合时间范围的文件，则直接返回
        pass

    def call_crawler(self, keywords, platform):
        pass

    def process_csv(self):
        pass

    def call_stance_detection(self, csv_file):
        df = pd.read_csv(csv_file)
        text = df["text"]
        target = df["target"]
        pass

    def call_slm_llm(self):
        pass

    def lcut_with_filter(self, text: str, stop_word_file: str="frontend/data/stopwords.txt") -> list:
        with open(stop_word_file, "r") as f:
            stop_word = set(map(lambda word: word.strip(), f.readlines()))
        words = jieba.lcut(text)
        return list(map(lambda word: word, filter(lambda word: word not in stop_word, words)))

    def top10_word_freq(self, text: str) -> dict:
        words = self.lcut_with_filter(text)
        word_freq = Counter(words)
        return dict(word_freq.most_common(10))

    def word_cloud(self, text):
        wc = WordCloud(font_path="msyh.ttc", width=800, height=600, background_color="white", max_words=200)
        wc.generate(text)
        plt.imshow(wc)
        plt.axis("off")
        return plt
