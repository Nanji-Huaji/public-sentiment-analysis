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
    def __init__(self):
        pass

    def call_crawler_necessary(self, start_time, end_time):
        pass

    def call_crawler(self, keywords, platform):
        pass

    def process_csv(self):
        pass

    def call_stance_detection(self, text, target, csv_file):
        pass

    def call_slm_llm(self):
        pass

    def top10_word_freq(self, text: str) -> dict:
        pass

    def word_cloud(self, text):
        pass
