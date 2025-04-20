import json
import subprocess
import os
import math
import pandas as pd

DATASET_INCREASE_ROUND = 20  # 数据集增量轮数
MODEL_TRAIN_EPOCHS = 20  # 模型训练轮数
MAX_DATASET_SIZE = 100000000  # 数据集最大行数


def process_dataset(
    raw_path: str, data_ratio_begin: float, data_ratio_end: float, processed_path: str, cover: bool = False
):
    """
    从原始数据集中截取指定比例范围的数据，并分别处理 train.csv 和 test.csv，
    将结果写入到目标目录中的 train.csv 和 test.csv。

    Args:
        raw_path (str): 原始数据集所在目录路径。
        data_ratio_begin (float): 起始比例（0.0 到 1.0）。
        data_ratio_end (float): 结束比例（0.0 到 1.0）。
        processed_path (str): 处理后数据集所在目录路径。

    Returns:
        None
    """
    # 确保目标目录存在
    os.makedirs(processed_path, exist_ok=True)

    # 定义需要处理的文件
    files_to_process = ["train.csv"]

    for file_name in files_to_process:
        raw_file_path = os.path.join(raw_path, file_name)
        processed_file_path = os.path.join(processed_path, file_name)

        # 检查原始文件是否存在
        if not os.path.exists(raw_file_path):
            print(f"文件 {raw_file_path} 不存在，跳过处理。")
            continue

        # 读取原始数据集
        raw_data = pd.read_csv(raw_file_path)

        # 确保比例范围合法
        if not (0.0 <= data_ratio_begin < data_ratio_end <= 1.0):
            raise ValueError(
                "data_ratio_begin 和 data_ratio_end 必须在 0.0 到 1.0 之间，且 data_ratio_begin < data_ratio_end"
            )

        # 计算起始和结束行索引
        total_rows = len(raw_data)
        start_idx = math.floor(total_rows * data_ratio_begin)
        end_idx = math.ceil(total_rows * data_ratio_end)

        # 截取指定范围的数据
        subset_data = raw_data.iloc[start_idx:end_idx]

        # 将数据写入目标文件
        try:
            # 如果目标文件已存在，则追加；否则创建新文件
            if not cover:
                subset_data.to_csv(
                    processed_file_path, mode="a", index=False, header=not os.path.exists(processed_file_path)
                )
                print(f"文件 {file_name} 已成功处理并写入到 {processed_file_path}")
            else:
                subset_data.to_csv(
                    processed_file_path, mode="w", index=False, header=["text", "target", "label"]  # 添加表头
                )
                print(f"文件 {file_name} 已成功覆盖写入到 {processed_file_path}")
        except Exception as e:
            print(f"写入文件 {processed_file_path} 时出错: {e}")


def get_data_length(file_path: str) -> int:
    """
    获取指定文件的行数（不包括标题行）。

    Args:
        file_path (str): 文件路径。

    Returns:
        int: 文件的行数（不包括标题行）。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return len(lines) - 1  # 减去标题行
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
        return 0
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0


def calculate_label_proportions(file_path="data/processed/train.csv"):
    df = pd.read_csv(file_path, on_bad_lines="warn")
    label_counts = df["label"].value_counts(normalize=True)
    label_proportions = label_counts.to_dict()

    for label, proportion in label_proportions.items():
        print(f"Label {label}: {proportion:.2f}")
    return label_proportions


def train_model(data_ratio: list, train_epochs: int, checkpoint_path: str, conda_env: str = "nlp"):
    """
    训练模型，使用指定的 Conda 环境。

    Args:
        data_ratio (list): 数据比例列表，包含三个介于 0 和 1 之间的浮点数。
        train_epochs (int): 训练轮数。
        checkpoint_path (str): 检查点路径。
        conda_env (str): Conda 环境名称，默认为 "nlp"。
    """

    print("开始训练模型...")

    # 验证 data_ratio 是否包含三个介于 0 和 1 之间的浮点数
    if len(data_ratio) != 3 or not all(0.0 <= r <= 1.0 for r in data_ratio):
        raise ValueError("data_ratio 必须是包含三个介于 0 和 1 之间的浮点数的列表。")

    # 调用 subprocess.run 执行训练脚本
    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            conda_env,  # 指定 Conda 环境
            "python",
            "src/train.py",
            "--class_counts",
            str(data_ratio[0]),  # 第一个比例
            str(data_ratio[1]),  # 第二个比例
            str(data_ratio[2]),  # 第三个比例
            "--train_epochs",
            str(train_epochs),  # 确保是字符串
            "--checkpoint_path",
            checkpoint_path,  # checkpoint_path 应该是字符串
        ]
    )


def get_checkpoint_path(file_path) -> str:
    """
    从文件路径中提取模型检查点路径
    """
    subdirs = [os.path.join(file_path, d) for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]

    if not subdirs:
        return None  # 如果没有子目录，返回 None

    # 按最后修改时间排序，获取最新的子目录
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir


def get_checkpoint_path_f1_highest(file_path) -> str:
    """
    从 file_path 下所有子目录的 trainer_state.json 文件中提取 eval_f1 值，
    返回 eval_f1 最大的目录路径。

    Args:
        file_path (str): 父目录路径。

    Returns:
        str: eval_f1 最大的子目录路径。如果没有找到有效的目录，则返回 None。
    """
    best_f1 = float("-inf")  # 初始化为负无穷
    best_dir = None

    # 遍历 file_path 下的所有子目录
    for subdir in [
        os.path.join(file_path, d) for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))
    ]:
        trainer_state_path = os.path.join(subdir, "trainer_state.json")

        # 检查 trainer_state.json 是否存在
        if not os.path.exists(trainer_state_path):
            continue

        try:
            # 读取 trainer_state.json 文件
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)

            # 提取 log_history 列表的最后一个元素
            log_history = trainer_state.get("log_history", [])
            if not log_history:
                continue  # 如果 log_history 为空，跳过该目录

            last_log = log_history[-1]  # 获取最后一个元素
            eval_f1 = last_log.get("eval_f1")  # 提取 eval_f1 值

            # 更新最佳目录
            if eval_f1 is not None and eval_f1 > best_f1:
                best_f1 = eval_f1
                best_dir = subdir
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析 {trainer_state_path} 时出错: {e}")
            continue

    return best_dir


def get_best_checkpoint_from_latest_ten(file_path) -> str:
    """
    从 file_path 下最新的十个子目录的 trainer_state.json 文件中提取 eval_f1 值，
    返回 eval_f1 最大的目录路径。

    Args:
        file_path (str): 父目录路径。

    Returns:
        str: eval_f1 最大的子目录路径。如果没有找到有效的目录，则返回 None。
    """
    best_f1 = float("-inf")  # 初始化为负无穷
    best_dir = None

    # 获取所有子目录，并按最后修改时间排序（从新到旧）
    subdirs = sorted(
        [os.path.join(file_path, d) for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))],
        key=os.path.getmtime,
        reverse=True,
    )

    # 取最新的十个子目录
    latest_five_subdirs = subdirs[:10]

    # 遍历最新的十个子目录
    for subdir in latest_five_subdirs:
        trainer_state_path = os.path.join(subdir, "trainer_state.json")

        # 检查 trainer_state.json 是否存在
        if not os.path.exists(trainer_state_path):
            continue

        try:
            # 读取 trainer_state.json 文件
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)

            # 提取 log_history 列表的最后一个元素
            log_history = trainer_state.get("log_history", [])
            if not log_history:
                continue  # 如果 log_history 为空，跳过该目录

            last_log = log_history[-1]  # 获取最后一个元素
            eval_f1 = last_log.get("eval_f1")  # 提取 eval_f1 值

            # 更新最佳目录
            if eval_f1 is not None and eval_f1 > best_f1:
                best_f1 = eval_f1
                best_dir = subdir
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析 {trainer_state_path} 时出错: {e}")
            continue

    return best_dir


def main():
    """
    在运行第一次后删除is_initial_train = True
    """
    is_initial_train = True
    with open("data/processed/train.csv", "w", encoding="utf-8") as f:
        f.write("text,target,label\n")  # 写入表头
    for dataset in ["nlpcc", "Weibo-SD", "C-Stance"]:
        raw_path = f"data/csv_data/{dataset}"
        processed_path = f"data/processed"
        if is_initial_train:
            checkpoint_path = get_checkpoint_path("models/output")
        else:
            checkpoint_path = get_best_checkpoint_from_latest_ten("models/output")
        print(f"checkpoint_path: {checkpoint_path}")
        for i in range(DATASET_INCREASE_ROUND):
            cover = False if (get_data_length("data/processed/train.csv") < MAX_DATASET_SIZE) else True
            data_ratio = [i * 1 / DATASET_INCREASE_ROUND, (i + 1) * 1 / DATASET_INCREASE_ROUND]
            process_dataset(raw_path, data_ratio[0], data_ratio[1], processed_path, cover=cover)
            label_proportions = calculate_label_proportions("data/processed/train.csv")
            train_model(
                [label_proportions[0], label_proportions[1], label_proportions[2]], MODEL_TRAIN_EPOCHS, checkpoint_path
            )
            is_initial_train = False


if __name__ == "__main__":
    main()
