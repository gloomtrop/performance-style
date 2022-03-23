from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.rnn import RNN
from utils.loading import DeviationDataset

INPUT_SIZE = 7
HIDDEN_SIZE = 50
NUM_LAYERS = 2
NUM_CLASSES = 11
SEQUENCE_LENGTH = 50
SEQUENCE_OFFSET = 10
BATCH_SIZE = 25
LEARNING_RATE = 0.001
EPOCHS = 100

model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
# model = GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
# model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()

dataset = DeviationDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=1)


def train():
    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(epoch)
    print('Finished Training')


if __name__ == "__main__":
    train()
