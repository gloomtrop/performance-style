import torch

from datasets.labelling import LabelledDataSet
from torch.utils.data import DataLoader

dataset = LabelledDataSet(scaled=True)
dataloader = DataLoader(dataset.all)

training_ys = torch.stack(dataset.train.y)
means = torch.mean(training_ys, dim=0)
mins = torch.min(training_ys, dim=0)
maxs = torch.max(training_ys, dim=0)
print(training_ys)
print(means)
print(mins)
print(maxs)
