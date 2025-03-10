import json
import pandas as pd
import os


train_csv = "data/processed/train.csv"
test_csv = "data/processed/test.csv"


"""
立场标签，0=反对，1=支持，2=中立
"""


def process_vast(data_dir):
    # process train data
    def process_table(file: str):
        df = pd.read_csv(os.path.join(data_dir, file))
        processed_df = pd.DataFrame()
        processed_df["text"] = df["post"]
        # 如果 new_topic 非空，取 new_topic 作为 target 列，否则取 ori_topic 作为 target
        processed_df["target"] = df.apply(
            lambda row: row["new_topic"] if pd.notna(row["new_topic"]) else row["ori_topic"], axis=1
        )
        # 取 label 作为 label 列
        processed_df["label"] = df["label"]
        # 将处理后的 DataFrame 写入到 train_csv 所对应的 CSV 文件中
        save_file = train_csv if "train" in file else test_csv
        processed_df.to_csv(save_file, index=False)

    for file in ["train.csv", "test.csv"]:
        process_table(data_dir + "/" + file)


def process_weibo_sd(data_dir):
    def process_json_file(file: str):
        with open(file, "w") as f:
            data = json.load(f)
        processed_df = pd.DataFrame(data)
        processed_df["label"] = processed_df["label"].apply(lambda x: 0 if x == 1 else 1 if x == 0 else 2)
        save_file = train_csv if "train" in file else test_csv
        processed_df.to_csv(save_file, index=False)

    for file in ["train.json", "test.json"]:
        process_json_file(data_dir + "/" + file)


def process_nlpcc(nlpcc_file, split_ratio):
    df = pd.read_csv(nlpcc_file)
    processed_df = pd.DataFrame()
    df = df.sample(frac=1).reset_index(drop=True)
    # 计算分割点
    split_point = int(len(df) * split_ratio)

    # 分割数据
    train_df = df[:split_point]
    test_df = df[split_point:]

    # 将数据写入 CSV 文件
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)


def gen_train_test_val(nlpcc_file):
    pass


def main():
    col_name = ["text", "target", "label"]
    # 写入表头
    pd.DataFrame(columns=col_name).to_csv(train_csv, index=False)
    pd.DataFrame(columns=col_name).to_csv(test_csv, index=False)
    # 处理 VAST 数据集
    data_dir = "data/raw/vast/"
    process_vast(data_dir)
    # 处理 Weibo-SD 数据集
    data_dir = "data/raw/Weibo-SD"
    process_weibo_sd(data_dir)
    # 处理 NLPCC 数据集
    nlpcc_file = "data/processed/train.csv"
    process_nlpcc(nlpcc_file, 0.8)


if __name__ == "__main__":
    main()
