import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.input_arguments = locals()
        del self.input_arguments['self']
        del self.input_arguments['__class__']

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = self.rnn_type(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    @property
    def rnn_type(self):
        return nn.RNN

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]

        return self.fc(out)

    def save(self, path):
        save_dict = {
            'state_dict': self.state_dict(),
            'class': self.__class__,
            'input_arguments': self.input_arguments
        }
        torch.save(save_dict, path)


class GRU(RNN):

    @property
    def rnn_type(self):
        return nn.GRU


class LSTM(RNN):

    @property
    def rnn_type(self):
        return nn.LSTM

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, (h0, c0))
        out = out[:, -1, :]

        return self.fc(out)
