import torch
from torch.utils.data import Dataset

from utils.loading import load_labelled_data


class LabelledDataSet(Dataset):

    def __init__(self):
        labels = load_labelled_data()
        self.x = []
        self.y = torch.from_numpy(labels.to_numpy()).float()

        self.len = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
