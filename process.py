import json
import pandas as pd
import os


train_csv = "data/processed/train.csv"
test_csv = "data/processed/test.csv"


"""
立场标签，0=反对，1=支持，2=中立
"""


def process_vast(original_train_file, original_test_file):
    # process train data
    def process_table(file: str):
        df = pd.read_csv(file)
        df = df.dropna(subset=["post", "new_topic", "ori_topic"])  # 跳过标签为空的部分
        processed_df = pd.DataFrame()
        processed_df["text"] = df["post"]
        # 如果 new_topic 非空，取 new_topic 作为 target 列，否则取 ori_topic 作为 target
        processed_df["target"] = df.apply(
            lambda row: row["new_topic"] if pd.notna(row["new_topic"]) else row["ori_topic"], axis=1
        )

        # 将处理后的 DataFrame 追加到 train_csv 所对应的 CSV 文件中
        save_file = train_csv if "train" in file else test_csv
        processed_df.to_csv(save_file, mode="a", header=False, index=False)

    for file in [original_train_file, original_test_file]:
        process_table(file)


def process_weibo_sd(original_train_file, original_test_file):
    def process_json_file(file: str):
        with open(file, "r") as f:
            data = json.load(f)
        # 过滤掉标签为空的部分
        data = [item for item in data if item["label"] is not None]
        # 只选择需要的列
        filtered_data = {
            "text": [item["text"] for item in data],
            "target": [item["target"] for item in data],
            "label": [0 if item["label"] == 1 else 1 if item["label"] == 0 else 2 for item in data],
        }
        processed_df = pd.DataFrame(filtered_data)
        save_file = train_csv if "train" in file else test_csv
        processed_df.to_csv(save_file, mode="a", header=False, index=False)

    for file in [original_train_file, original_test_file]:
        process_json_file(file)


def process_nlpcc(nlpcc_file, split_ratio):
    df = pd.read_csv(nlpcc_file)
    df = df.dropna(subset=["text", "target", "stance"])  # 跳过标签为空的部分
    df = df.sample(frac=1).reset_index(drop=True)
    # 计算分割点
    split_point = int(len(df) * split_ratio)

    # 分割数据
    train_df = df[:split_point]
    test_df = df[split_point:]

    processed_train_df = pd.DataFrame()
    processed_test_df = pd.DataFrame()

    # 将数据写入 CSV 文件
    processed_train_df["text"] = train_df["text"]
    processed_train_df["target"] = train_df["target"]
    processed_train_df["label"] = train_df["stance"]

    processed_test_df["text"] = test_df["text"]
    processed_test_df["target"] = test_df["target"]
    processed_test_df["label"] = test_df["stance"]

    processed_train_df.to_csv(train_csv, mode="a", header=False, index=False)
    processed_test_df.to_csv(test_csv, mode="a", header=False, index=False)


def gen_train_test_val(nlpcc_file):
    pass


def main():
    col_name = ["text", "target", "label"]
    # 写入表头
    pd.DataFrame(columns=col_name).to_csv(train_csv, index=False)
    pd.DataFrame(columns=col_name).to_csv(test_csv, index=False)
    # 处理 VAST 数据集
    # process_vast("data/raw/vast/VAST/vast_train.csv", "data/raw/vast/VAST/vast_test.csv")
    # 处理 Weibo SD 数据集
    process_weibo_sd("data/raw/Weibo-SD/train.json", "data/raw/Weibo-SD/test.json")
    # 处理 NLPCC 数据集
    process_nlpcc("data/raw/nlpcc/train.csv", 0.8)


if __name__ == "__main__":
    main()
