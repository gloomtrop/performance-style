from datasets.labelling import LabelledDataSet
from torch.utils.data import DataLoader

dataset = LabelledDataSet()
dataloader = DataLoader(dataset)

for x, y in dataloader:
    print(x.shape, y.shape)
