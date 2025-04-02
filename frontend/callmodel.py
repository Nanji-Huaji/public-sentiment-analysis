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
    def __init__(self, model):
        pass

    def call_crawler_necessary(self, start_time, end_time):
        pass

    def call_crawler(self, keywords, platform):
        pass

    def process_csv(self, csv_file):
        # csv文件中只有text和target两列，用分类器进行分类
        df = pd.read_csv(csv_file)
        df = df[["text", "target"]]

    def call_slm_llm(self):
        pass

    def top10_word_freq(self, text: str) -> dict:
        pass

    def word_cloud(self, text):
        pass
