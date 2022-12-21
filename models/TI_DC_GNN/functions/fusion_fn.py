from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from models.TI_DC_GNN.functions.transform_fn import TransformFn


class FusionFn(TransformFn, ABC):
    def __init__(self, act_fn_name):
        super(FusionFn, self).__init__(act_fn_name)
        self.function = None

    def _forward(self, arg1, arg2):
        return self.function(arg1, arg2)

    @abstractmethod
    def get_output_dim(self, arg1, arg2):
        pass



# ------------Arithmetic--------------

class ArithmeticFusionFn(FusionFn):
    def get_output_dim(self, input_dim1, input_dim2):
        if input_dim1 != input_dim2:
            raise RuntimeError('Dimensions must be equal')
        return input_dim1


class PlusFusionFn(ArithmeticFusionFn):
    def __init__(self, act_fn_name):
        super().__init__(act_fn_name)
        self.function = lambda x,y: x+y


class MinusFusionFn(ArithmeticFusionFn):
    def __init__(self, act_fn_name):
        super().__init__(act_fn_name)
        self.function = lambda x,y: x-y


class MultFusionFn(ArithmeticFusionFn):
    def __init__(self, act_fn_name):
        super().__init__(act_fn_name)
        self.function = lambda x,y: x*y


# ------------Concat/Select--------------


class ConcatFusionFn(FusionFn):
    def __init__(self, act_fn_name):
        super().__init__(act_fn_name)
        self.function = lambda x,y: torch.cat([x, y], dim=1)

    def get_output_dim(self, input_dim1, input_dim2):
        return input_dim1 + input_dim2


class FirstFusionFn(FusionFn):
    def __init__(self, act_fn_name):
        super().__init__(act_fn_name)
        self.function = lambda x,y: x

    def get_output_dim(self, input_dim1, input_dim2):
        return input_dim1


class SecondFusionFn(FusionFn):
    def __init__(self, act_fn_name):
        super().__init__(act_fn_name)
        self.function = lambda x,y: y

    def get_output_dim(self, input_dim1, input_dim2):
        return input_dim2


# ------------RNN--------------
class RNNCellFn(TransformFn):
    def __init__(self, act_fn_name, input_dim, hidden_dim, cell_name):
        super().__init__(act_fn_name)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.function = getattr(nn, cell_name)(input_dim, hidden_dim)  # GRUCell, RNNCell, LSTMCell

    def get_output_dim(self, input_dim, hidden_dim):
        return self.hidden_dim


def get_fusion_fn(fn_name, act_fn_name, input_dim, hidden_dim):
    if fn_name is None:
        return None
    if fn_name == 'plus':
        return PlusFusionFn(act_fn_name)
    if fn_name == 'minus':
        return MinusFusionFn(act_fn_name)
    if fn_name == 'mult':
        return MultFusionFn(act_fn_name)
    if fn_name == 'concat':
        return ConcatFusionFn(act_fn_name)
    if fn_name == 'first':
        return FirstFusionFn(act_fn_name)
    if fn_name == 'second':
        return SecondFusionFn(act_fn_name)
    return get_rnn_fusion_fn(fn_name, act_fn_name, input_dim, hidden_dim)


def get_rnn_fusion_fn(fn_name, act_fn_name, input_dim, hidden_dim):
    return RNNCellFn(act_fn_name, input_dim, hidden_dim, fn_name+'Cell')

