import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.rnn import RNN
from utils.loading import DeviationDataset
from utils.testing import compute_accuracy

INPUT_SIZE = 7
HIDDEN_SIZE = 50
NUM_LAYERS = 2
NUM_CLASSES = 11
SEQUENCE_LENGTH = 50
SEQUENCE_OFFSET = 10
BATCH_SIZE = 25
LEARNING_RATE = 0.001
EPOCHS = 10

model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
# model = GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
# model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()

dataset = DeviationDataset()
train, val = dataset.split()
trainloader = DataLoader(train, batch_size=BATCH_SIZE,
                         shuffle=True)
valloader = DataLoader(val, batch_size=BATCH_SIZE,
                       shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    y_true = []
    y_pred = []
    for i, data in enumerate(valloader):
        inputs, label = data
        outputs = model(inputs)
        y_true += torch.argmax(label, dim=1)
        y_pred += torch.argmax(outputs, dim=1)
    accuracy = float(compute_accuracy(y_true, y_pred))

print('Finished Training')
