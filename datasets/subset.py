from torch.utils.data import Dataset


class DataSubSet(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
