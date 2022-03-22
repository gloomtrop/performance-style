import numpy as np
import torch

from models.rnn import RNN, GRU, LSTM
from utils.loading import load_split
from utils.preprocessing import chunker

INPUT_SIZE = 7
HIDDEN_SIZE = 50
NUM_LAYERS = 2
NUM_CLASSES = 11
SEQUENCE_LENGTH = 50
SEQUENCE_OFFSET = 10

rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
gru = GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

train, test = load_split()
for chunk in list(chunker(train, SEQUENCE_LENGTH, SEQUENCE_OFFSET))[:1]:
    X = chunk.drop(columns=['performer']).to_numpy()
    batch = torch.tensor(np.expand_dims(X, axis=0)).float()
    print(batch.shape)
    print(rnn(batch))
    print(gru(batch))
    print(lstm(batch))
