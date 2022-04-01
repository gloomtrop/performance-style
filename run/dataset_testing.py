from datasets.labelling import LabelledDataSet
from torch.utils.data import DataLoader

dataset = LabelledDataSet()
dataloader = DataLoader(dataset.all)

for x, y in dataloader:
    print(x.shape, y.shape)
