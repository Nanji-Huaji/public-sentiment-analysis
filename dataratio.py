import pandas as pd


def calculate_label_proportions(file_path):
    df = pd.read_csv(file_path)
    label_counts = df["label"].value_counts(normalize=True) * 100
    label_proportions = label_counts.to_dict()

    for label, proportion in label_proportions.items():
        print(f"Label {label}: {proportion:.2f}%")
    return label_proportions.values()


if __name__ == "__main__":
    file_path = "data/processed/train.csv"
    calculate_label_proportions(file_path)
