import torch

import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, model_name):
        self.data = data
        self.targets = targets
        self.model_name = model_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass

    def collect_fn(self, batch):
        pass

    def get_data(self):
        return self.data
