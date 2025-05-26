import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    draw_pie_chart,
    cut_text,
    cut_text_from_csv,
    get_top_words,
    get_top_words_from_csv,
    draw_wordcloud,
    draw_heatmap,
    merge_csv_files
)

import collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from inference import StanceDetection, LLMInference, SLMInference
from MediaCrawler import MediaCrawler

# 定义全局变量
if "stance_detection" not in st.session_state:
    st.session_state.stance_detection = StanceDetection(
        "models/produce/checkpoint-825", "google-bert/bert-base-multilingual-cased"
    )

# 定义LLM模型配置字典
llm_configs = {
    "gpt_4o": {
        "name": "gpt-4o",
        "api_base": "https://api.bltcy.ai/v1",
        "api_key": os.getenv("OPENAI_PUBLIC_SENTIMENT_SYSTEM_API_KEY"),
    },
    "deepseek_r1": {
        "name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
    "gpt_35": {
        "name": "gpt-3.5",
        "api_base": "https://api.bltcy.ai/v1",
        "api_key": os.getenv("OPENAI_PUBLIC_SENTIMENT_SYSTEM_API_KEY"),
    },
    "gpt_3": {
        "name": "gpt-3",
        "api_base": "https://api.bltcy.ai/v1",
        "api_key": os.getenv("OPENAI_PUBLIC_SENTIMENT_SYSTEM_API_KEY"),
    },
}

# 初始化所有LLM模型
for model_key, config in llm_configs.items():
    if model_key not in st.session_state:
        st.session_state[model_key] = LLMInference(
            model=config["name"], api_base=config["api_base"], api_key=config["api_key"]
        )

if "slm" not in st.session_state:
    st.session_state.slm = SLMInference()


def call_crawler(platform: str, keywords: list[str], max_crawl_note: int = 30) -> str | None:
    media_crawler = MediaCrawler(platform=platform, keywords=keywords, max_crawl_note=max_crawl_note)
    csv = media_crawler.crawl()
    if csv is not None:
        csv = media_crawler.get_valid_csv_file_path(csv)
        if csv is None:
            st.error("爬取数据失败，请检查平台设置或关键词。")
            return None
        csv = merge_csv_files(csv)
        return csv
    else:
        st.error("没有找到相关数据，请检查关键词或平台设置。")
        return None


def call_crawler_test(*args, **kwargs) -> str:
    time.sleep(30)  # 模拟爬取时间
    return "data/analysis/demo.csv"

st.title("公共舆情分析系统")

stance_detection = st.session_state.stance_detection

st.markdown("这是一个简单的演示立场检测的demo。" "给定一个句子和一个目标，我们会告诉您该句子对目标的情感倾向。")

sentence = st.text_input("输入一个句子", "I feel great")
target = st.text_input("输入一个你要检测的目标", "I")

if st.button("分析"):
    st.write(f"分析: {sentence}")
    stance = stance_detection(sentence, target)
    st.write(f"情感: {stance['label']}")
    st.write(f"置信度: {stance['score']}")

st.title("公共舆情监控")

st.markdown(
    "这是一个演示我们的情感监测方法如何工作的简单示例。"
    "给定一个监测主题，我们会为您展示最新社交媒体帖子的情感倾向。"
    "请注意，该部分尚未完全实现。"
)

st.markdown("您可以使用大语言模型来帮助监测公众舆论的情感倾向。")
st.markdown("请注意，大语言模型可能存在偏见和幻觉问题，其生成的内容可能不够可靠。")

topic = st.text_input("输入您想要监测的主题", "胡鑫宇，满江红，泼水节")
target = st.text_input("输入您想要监测的目标", "政府")
keyword_monitoring = st.text_input("输入您想要检索的帖子关键词，用逗号分隔", "胡鑫宇, 满江红, 泼水节")
platform = st.selectbox("选择社交媒体平台", ["Weibo", "RedNote", "Tieba"])
# date_range = st.date_input("选择日期范围", [pd.to_datetime("2023-01-01"), pd.to_datetime("2025-03-10")])
crawler_max_note = st.number_input("设置最大爬取帖子数", min_value=1, max_value=100, value=30, step=1)
llm_used = st.selectbox("选择语言模型", ["不使用LLM", "GPT-4o", "DeepSeek-r1", "GPT-3.5", "GPT-3"])

keywords = [kw.strip() for kw in keyword_monitoring.split(",") if kw.strip()]

if st.button("开始监测"):

    data_file = None
    progress_bar = st.progress(0, text="正在搜索数据……")
    import threading
    import time

    def run_crawler():
        global data_file
        # data_file = call_crawler(keywords=keywords, platform=platform)
        data_file = call_crawler_test(keywords=keywords, platform=platform, max_crawl_note=crawler_max_note)

    thread = threading.Thread(target=run_crawler)
    thread.start()
    progress = 0
    while thread.is_alive():
        progress = min(progress + 5, 95)
        progress_bar.progress(progress, text="正在搜索数据……")
        time.sleep(0.1)
    thread.join()
    progress_bar.progress(100, text="正在搜索数据……")
    # 等待进度条

    if data_file is None:
        data_file = "analysis/demo.csv"
    # 处理 CSV 文件
    stance_detection.process_csv(data_file)
    # 获取process_csv处理后的label列的占比
    df = pd.read_csv(data_file)
    label_counts = df["label"].value_counts(normalize=True)
    label_counts = label_counts.reindex([1, 0, 2], fill_value=0)  # 确保顺序为 POSITIVE, NEGATIVE, NEUTRAL
    pos_ratio = label_counts[1]
    neg_ratio = label_counts[0]
    neu_ratio = label_counts[2]
    # 处理文本
    word_list = cut_text_from_csv(data_file, "text")
    word_count = collections.Counter(word_list)
    word_dict = dict(word_count.most_common(100))
    # 绘制饼图
    st.markdown("### 情感倾向饼图：")
    st.markdown(f"正面情感占比：{pos_ratio:.2%}，负面情感占比：{neg_ratio:.2%}，中立情感占比：{neu_ratio:.2%}")
    pie_fig = draw_pie_chart(pos_ratio, neg_ratio, neu_ratio)
    st.pyplot(pie_fig)
    # 绘制词云
    st.markdown("### 词云图：")
    st.markdown("词云图展示了文本中最常见的单词，单词的大小表示其出现的频率。")
    wordcloud_fig = draw_wordcloud(word_dict)
    st.pyplot(wordcloud_fig)
    # 绘制热力图
    st.markdown("### 热力图：")
    st.markdown("热力图展示了文本中单词之间的共现关系，颜色越深表示共现次数越多。")
    heatmap_fig = draw_heatmap(word_dict)
    st.pyplot(heatmap_fig)
    if llm_used != "不使用LLM":
        st.markdown("### LLM分析:")
        # 将模型名称映射到session_state中的键名
        model_map = {"GPT-4o": "gpt_4o", "DeepSeek-r1": "deepseek_r1", "GPT-3.5": "gpt_35", "GPT-3": "gpt_3"}
        llm = st.session_state[model_map[llm_used]]

        # 调用小模型，抽样总结
        df = pd.read_csv(data_file)
        # 由于target要求保持一致，取出数据集中的第一个target
        target = df["target"].iloc[0]
        # 从 df 中各抽取 20 条标签为 0、1、2 的数据，并存储为字典
        sampled_data_dict = {
            0: df[df["label"] == 0]
            .sample(n=20, random_state=42, replace=True)
            .reset_index(drop=True),  # 标签为 0 的数据
            1: df[df["label"] == 1]
            .sample(n=20, random_state=42, replace=True)
            .reset_index(drop=True),  # 标签为 1 的数据
            2: df[df["label"] == 2]
            .sample(n=20, random_state=42, replace=True)
            .reset_index(drop=True),  # 标签为 2 的数据
        }

        sampled_data_list = [sampled_data_dict[0], sampled_data_dict[1], sampled_data_dict[2]]

        slm_summary = st.session_state.slm.summary(
            favor_text=sampled_data_list[1],
            neutral_text=sampled_data_list[2],
            against_text=sampled_data_list[0],
            target=target,
            favor_rate=pos_ratio,
            against_rate=neg_ratio,
            neutral_rate=neu_ratio,
        )

        # 调用大模型，分析总结
        llm_inference = llm.analyze(
            summary=slm_summary,
            target=target,
            topic=topic,
            favor_rate=pos_ratio,
            against_rate=neg_ratio,
            neutral_rate=neu_ratio,
        )

        st.markdown("### LLM分析摘要：")
        st.markdown(slm_summary)

        st.markdown("### LLM分析结果：")
        st.markdown(llm_inference)
