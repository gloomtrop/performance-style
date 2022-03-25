import torch
from torch.utils.data import DataLoader

from utils.loading import DeviationDataset, load_model
from utils.loading import performers_full_name_list
from utils.paths import path_from_root
from utils.plotting import plot_confusion_matrix

MODEL_NAME = 'LSTM_int.pkl'
MODEL_PATH = path_from_root('models', 'saved', MODEL_NAME)
BATCH_SIZE = 25

model = load_model(MODEL_PATH)
model.eval()

dataset = DeviationDataset()
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

y_true = []
y_pred = []

print('Start modelling')
print(len(test_loader))
for i, data in enumerate(test_loader):
    inputs, label = data
    outputs = model(inputs)
    y_true += torch.argmax(label, dim=1)
    y_pred += torch.argmax(outputs, dim=1)

plot_confusion_matrix(y_true, y_pred, [torch.tensor(i) for i in range(11)], performers_full_name_list())
