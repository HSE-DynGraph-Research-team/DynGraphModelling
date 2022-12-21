import torch
import torch.nn as nn

from models.TI_DC_GNN.functions.function import Function


class TransformFn(Function):
    def __init__(self):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

class CellNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, input, hidden):
        pass

    def get_output_dim(self):
        pass


class GRUCell(CellNN):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, input, hidden):
        return self.cell(input, hidden)

    def get_output_dim(self):
        return self.hidden_dim


class RNNCell(CellNN):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        self.cell = nn.RNNCell(input_dim, hidden_dim)

    def forward(self, input, hidden):
        return self.cell(input, hidden)

    def get_output_dim(self):
        return self.hidden_dim


class ConcatCell(CellNN):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, input, hidden):
        torch.cat([input, hidden], dim=2)

    def get_output_dim(self):
        return self.input_dim + self.hidden_dim


class MeanInputCell(CellNN):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, input, hidden):
        torch.mean(input, dim=1)

    def get_output_dim(self):
        return self.input_dim


class LinearInputCell(CellNN):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, input, hidden):
        return self.fc(input)

    def get_output_dim(self):
        return self.hidden_dim


def get_cell(cell_name):
    if cell_name is None:
        return None
    if cell_name == 'RNN':
        return RNNCell
    if cell_name == 'GRU':
        return GRUCell
    if cell_name == 'concat':
        return ConcatCell
