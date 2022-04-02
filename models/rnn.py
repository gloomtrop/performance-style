import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0):
        super(RNN, self).__init__()

        self.input_arguments = locals()
        del self.input_arguments['self']
        del self.input_arguments['__class__']

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = self.rnn_type(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    @property
    def rnn_type(self):
        return nn.RNN

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.dropout(out)

        return self.fc(out)

    def save(self, path, **kwargs):
        save_dict = {
            'state_dict': self.state_dict(),
            'class': self.__class__,
            'input_arguments': self.input_arguments
        }
        save_dict.update(kwargs)
        torch.save(save_dict, path)
        print(f'Model was saved')


class RNNJoint(RNN):
    def __init__(self, input_size, hidden_size, num_layers, num_mid, num_classes):
        super(RNN, self).__init__()

        self.input_arguments = locals()
        del self.input_arguments['self']
        del self.input_arguments['__class__']

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = self.rnn_type(input_size, hidden_size, num_layers, batch_first=True)
        self.mid_fc = nn.Linear(hidden_size, num_mid)
        self.class_fc = nn.Linear(num_mid, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        mid = self.mid_fc(out)
        classes = self.class_fc(mid)
        return mid, classes


class GRU(RNN):

    @property
    def rnn_type(self):
        return nn.GRU


class GRUJoint(RNNJoint):

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


class LSTMJoint(RNNJoint):

    @property
    def rnn_type(self):
        return nn.LSTM

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, (h0, c0))
        out = out[:, -1, :]
        mid = self.mid_fc(out)
        classes = self.class_fc(mid)
        return mid, classes
