from torch.nn import L1Loss
from torch.utils.data import DataLoader

from datasets.labelling import LabelledDataSet
import torch

dataset = LabelledDataSet()

training_ys = torch.stack(dataset.train.y)
means = torch.mean(training_ys, dim=0)

CRITERION = L1Loss()
validation_loader = DataLoader(dataset.validation, shuffle=False)
test_loader = DataLoader(dataset.test, shuffle=False)

total_validation_loss = 0
for i, data in enumerate(validation_loader):
    inputs, labels = data
    total_validation_loss += CRITERION(means, labels)
avg_validation_loss = total_validation_loss / len(validation_loader)

total_test_loss = 0
for i, data in enumerate(test_loader):
    inputs, labels = data
    total_test_loss += CRITERION(means, labels)
avg_test_loss = total_test_loss / len(validation_loader)

print(f'Validation loss: {avg_validation_loss}')
print(f'Test loss:       {avg_test_loss}')
