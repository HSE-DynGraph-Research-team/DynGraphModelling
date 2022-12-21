import torch.nn as nn

from models.TI_DC_GNN.functions.function import Function


class ActFn(Function):
    def __init__(self):
        super().__init__()
        self.function = None

    def forward(self, x):
        return self.function(x)

    def get_output_dim(self, input_dim):
        return input_dim


class IdActFn(ActFn):
    def __init__(self):
        super().__init__()
        self.function = lambda x: x


class RelUActFn(ActFn):
    def __init__(self):
        super().__init__()
        self.function = nn.ReLU


class SigmoidActFn(ActFn):
    def __init__(self):
        super().__init__()
        self.function = nn.Sigmoid


class LeakyRelUActFn(ActFn):
    def __init__(self):
        super().__init__()
        self.function = nn.Sigmoid


def get_act_fn(act_name):
    if act_name == 'id':
        return IdActFn()
    if act_name == 'RelU':
        return RelUActFn()
    if act_name == 'Sigmoid':
        return SigmoidActFn()
    if act_name == 'LeakyRelU':
        return LeakyRelUActFn()
