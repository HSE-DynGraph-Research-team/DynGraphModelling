import torch
import torch.nn as nn


class AggregationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, input, hidden):
        pass

    def get_output_dim(self):
        pass


class MeanInputAggregarion(AggregationNN):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, input, hidden):
        # input: message_num x message_dim
        # hidden: memory_dim
        torch.mean(input, dim=0)

    def get_output_dim(self):
        return  self.input_dim + self.hidden_dim


def get_aggregator(agg_name):
    if agg_name is None:
        return None
    if agg_name == 'mean':
        return MeanInputAggregarion