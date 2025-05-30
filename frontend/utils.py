import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jieba

import collections

import wordcloud

import os


def draw_pie_chart(pos, neg, neu):
    ratio = {"positive": pos, "negative": neg, "neutral": neu}
    labels = list(ratio.keys())
    sizes = [ratio[label] for label in labels]
    colors = ["gold", "yellowgreen", "lightcoral"]
    explode = (0, 0, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=90)
    ax.axis("equal")
    return fig


def cut_text(text: str, stopword: str) -> list:
    """
    对文本进行分词，并去除停用词和空格
    :param text: 输入文本
    :param stopword: 停用词txt文件路径
    :return: 词汇列表
    """
    # 读取停用词
    with open(stopword, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

    # 分词
    words = jieba.cut(text)
    # 去除停用词和空格
    words = list(filter(lambda x: x not in stopwords and x.strip() != "", words))
    return words


def cut_text_from_csv(csv_file: str, text_column: str, stopword: str = "frontend/data/stopwords.txt") -> list[str]:
    """
    从csv文件中读取文本，进行分词，并去除停用词
    :param csv_file: csv文件路径
    :param text_column: 文本列名
    :param stopword: 停用词txt文件路径
    :return: 词汇列表
    """
    df = pd.read_csv(csv_file)
    text = " ".join(df[text_column].astype(str).tolist())
    words = cut_text(text, stopword)
    return words


def get_top_words(word_list: list, top_n: int = 10) -> dict:
    """
    获取词频最高的前n个词
    :param word_list: 词汇列表
    :param top_n: 前n个
    :return: 词频最高的前n个词及其频率的字典
    """
    word_count = collections.Counter(word_list)
    top_words = dict(word_count.most_common(top_n))
    return top_words


def get_top_words_from_csv(
    csv_file: str, text_column: str, stopword: str = "frontend/data/stopwords.txt", top_n: int = 10
) -> dict:
    """
    从csv文件中获取词频最高的前n个词
    :param csv_file: csv文件路径
    :param text_column: 文本列名
    :param stopword: 停用词txt文件路径
    :param top_n: 前n个
    :return: 词频最高的前n个词及其频率的字典
    """
    df = pd.read_csv(csv_file)
    text = " ".join(df[text_column].astype(str).tolist())
    words = cut_text(text, stopword)
    top_words = get_top_words(words, top_n)
    return top_words


def draw_wordcloud(
    word_dict: dict,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    max_words: int = 200,
    font_path: str = "frontend/data/fonts/SourceHanSansHWSC-Regular.otf",
):
    """
    绘制词云图
    :param word_dict: 词频字典，键为词，值为频率
    :param width: 图片宽度
    :param height: 图片高度
    :param background_color: 背景颜色
    :param max_words: 最多显示的词数
    :param font_path: 字体路径，中文词云需要指定中文字体
    :return: matplotlib图像对象
    """
    # 创建词云对象
    wc = wordcloud.WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        font_path=font_path,
        random_state=42,
    )

    # 将字典转换为文本形式
    text = " ".join([f"{word} " * int(freq) for word, freq in word_dict.items()])

    # 使用generate方法替代generate_from_frequencies
    wc.generate(text)

    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    return fig


def draw_heatmap(
    word_dict: dict,
    figsize: tuple = (10, 8),
    cmap: str = "YlOrRd",
    title: str = "词频热力图",
    annot: bool = True,
    font_path: str = "frontend/data/fonts/SourceHanSansHWSC-Regular.otf",
):
    """
    根据词频绘制矩阵形式的热力图
    :param word_dict: 词频字典，键为词，值为频率
    :param figsize: 图像尺寸，(宽, 高)
    :param cmap: 颜色映射，可选如 'YlOrRd', 'Blues', 'Greens', 'Reds' 等
    :param title: 图表标题
    :param annot: 是否在每个单元格上标注数值
    :param font_path: 字体路径，用于显示中文
    :return: matplotlib图像对象
    """
    # 设置中文字体
    from matplotlib.font_manager import FontProperties

    if font_path and os.path.exists(font_path):
        font_prop = FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        # 尝试使用系统中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "SimSun", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

    # 如果词典太大，只取前25个
    if len(word_dict) > 25:
        word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:25])

    # 创建数据
    words = list(word_dict.keys())
    frequencies = list(word_dict.values())

    # 计算矩阵的行列数（尽量接近方形）
    n = len(words)
    grid_size = int(np.ceil(np.sqrt(n)))

    # 创建矩阵并填充词频数据
    matrix = np.zeros((grid_size, grid_size))
    for i in range(n):
        row = i // grid_size
        col = i % grid_size
        matrix[row, col] = frequencies[i]

    # 创建图像
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    im = ax.imshow(matrix, cmap=cmap)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("词频", rotation=-90, va="bottom", fontproperties=font_prop if "font_prop" in locals() else None)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))

    # 准备x和y轴的标签
    x_labels = []
    y_labels = []
    for i in range(grid_size):
        y_row = []
        x_col = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(words):
                x_col.append(words[idx])
                y_row.append(words[idx])
            else:
                x_col.append("")
                y_row.append("")
        # 每行只显示第一个词
        y_labels.append(y_row[0] if y_row[0] else "")
        # 每列只显示第一个词
        x_labels.append(x_col[0] if x_col[0] else "")

    # 设置坐标轴标签
    if "font_prop" in locals():
        ax.set_xticklabels(x_labels, fontproperties=font_prop)
        ax.set_yticklabels(y_labels, fontproperties=font_prop)
    else:
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

    # 在热力图中标注数值和词汇
    if annot:
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(words):
                    word = words[idx]
                    freq = frequencies[idx]
                    # 根据背景色选择文字颜色，确保可读性
                    color = "white" if freq > max(frequencies) * 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{word}\n{freq}",
                        ha="center",
                        va="center",
                        color=color,
                        fontproperties=font_prop if "font_prop" in locals() else None,
                    )

    # 设置标题和调整布局
    if "font_prop" in locals():
        ax.set_title(title, fontproperties=font_prop)
    else:
        ax.set_title(title)

    fig.tight_layout()

    return fig


def merge_csv_files(input_files: list[str]) -> str:
    """
    合并多个CSV文件，并根据输入文件名自动生成输出文件路径
    :param input_files: 输入的CSV文件列表
    :return: 合并后的CSV文件路径
    """
    import re
    # 检查输入参数
    if not input_files:
        raise ValueError("输入文件列表不能为空")
    # 用于存储所有数据框的列表
    dfs = []

    # 读取所有CSV文件
    for file in input_files:
        if os.path.exists(file):
            try:
                convert_csv_column_name(file)
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        else:
            print(f"文件不存在: {file}")

    if not dfs:
        raise ValueError("没有可合并的有效CSV文件")

    # 合并所有数据框
    merged_df = pd.concat(dfs, ignore_index=True)

    # 自动生成输出文件名
    # 假设文件名格式为: {file_count}_{crawler_type}_{store_type}_{date}.csv
    first_file = os.path.basename(input_files[0])
    match = re.match(r"(\d+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([\d-]+)\\.csv", first_file)
    if match:
        file_count, crawler_type, _, date_str = match.groups()
        output_file = f"{file_count}_{crawler_type}_merged_{date_str}.csv"
    else:
        # 如果格式不符，默认合并文件名
        output_file = "merged_output.csv"

    # 输出到与输入文件相同目录
    output_dir = os.path.dirname(input_files[0])
    output_path = os.path.join(output_dir, output_file)

    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存合并后的数据框
    merged_df.to_csv(output_path, index=False)

    print(f"已将 {len(dfs)} 个CSV文件合并保存至 {output_path}")
    return output_path

def convert_csv_column_name(file: str) -> str:
    '''
    将CSV文件中的content列名转化为text列名
    :param file: CSV文件路径
    :return: CSV文件路径
    '''
    import pandas as pd
    df = pd.read_csv(file)
    if 'content' in df.columns:
        df.rename(columns={'content': 'text'}, inplace=True)
        df.to_csv(file, index=False)
        print(f"已将 {file} 中的 'content' 列重命名为 'text'")
    else:
        print(f"{file} 中未找到 'content' 列，无需更改")
    return file
