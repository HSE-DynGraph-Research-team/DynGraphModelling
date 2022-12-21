from abc import abstractmethod
import torch.nn as nn

from models.TI_DC_GNN.functions.transform_fn import TransformFn


class SingleTransformFn(TransformFn):
    def __init__(self, act_fn_name, input_dim, *args):
        super().__init__(act_fn_name)
        self.input_dim = input_dim

    @abstractmethod
    def _forward(self, input):
        pass

    @abstractmethod
    def get_output_dim(self):
        pass


class LinearTransfromFn(SingleTransformFn):
    def __init__(self, act_fn_name, input_dim, output_dim):
        super().__init__(act_fn_name, input_dim)
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def get_output_dim(self):
        return self.output_dim

    def _forward(self, x):
        return self.fc(x)


class IdTransfromFn(SingleTransformFn):
    def get_output_dim(self):
        return self.input_dim

    def _forward(self, x):
        return x


def get_single_transform_fn(fn_name, act_fn_name, input_dim, output_dim):
    if fn_name == 'linear':
        return LinearTransfromFn(act_fn_name, input_dim, output_dim)
    if fn_name == 'id':
        return IdTransfromFn(act_fn_name, input_dim)
