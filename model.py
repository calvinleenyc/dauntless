import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

lr_rate = 0.001

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2I = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2O = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2C = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        forget = F.sigmoid(self.i2f(combined))
        I = F.sigmoid(self.i2I(combined))
        O = F.sigmoid(self.i2O(combined))
        C = forget * cell + I * F.tanh(self.i2C(combined))
        H = O * F.tanh(C)
        return H, C

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def initCell(self):
        return Variable(torch.zeros(1, self.hidden_size))

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(StackedLSTM, self).__init__()
        self.lstm1 = LSTM(input_size, hidden_size1)
        self.lstm2 = LSTM(hidden_size1, hidden_size2) # try a skip connection later
        self.to_out = nn.Linear(hidden_size2, output_size)
        
    def forward(self, input, hiddens, cells):
        hidden1, cell1 = self.lstm1(input, hiddens[0], cells[0])
        hidden2, cell2 = self.lstm2(hidden1, hiddens[1], cells[1])
        out = F.log_softmax(self.to_out(hidden2))
        return out, [hidden1, hidden2], [cell1, cell2]

    def initHidden(self, batch_size = 1):
        return [Variable(torch.zeros(batch_size, self.lstm1.hidden_size)),
                Variable(torch.zeros(batch_size, self.lstm2.hidden_size))]

    def initCell(self, batch_size = 1):
        return [Variable(torch.zeros(batch_size, self.lstm1.hidden_size)),
                Variable(torch.zeros(batch_size, self.lstm2.hidden_size))]
