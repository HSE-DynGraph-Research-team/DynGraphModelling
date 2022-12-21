from models.TI_DC_GNN.functions.act_fn import get_act_fn
from models.TI_DC_GNN.functions.function import Function
from abc import abstractmethod


class TransformFn(Function):
    def __init__(self, act_fn_name, *args):
        super().__init__()
        self.act_fn = get_act_fn(act_fn_name)

    @abstractmethod
    def _forward(self, *args):
        pass

    def forward(self, *args):
        outputs = self._forward(*args)
        return self.act_fn(outputs)

    @abstractmethod
    def get_output_dim(self, *args):
        pass