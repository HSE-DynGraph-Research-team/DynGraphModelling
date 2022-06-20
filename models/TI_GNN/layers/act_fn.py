import torch.nn as nn


class ActFn:
    def __init__(self):
        self.function = None

    def __call__(self, x):
        return self.function(x)


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


def get_act_fn(act_name):
    if act_name == 'id':
        return IdActFn()
    if act_name == 'relU':
        return RelUActFn()
    if act_name == 'sigmoid':
        return SigmoidActFn()
